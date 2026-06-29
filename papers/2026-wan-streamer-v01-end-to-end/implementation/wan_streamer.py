"""
Wan-Streamer v0.1 — Core architecture implementation.
========================================================
Implements the five core ideas from the paper in a self-contained,
numpy-only toy:

  1. MultimodalTokenStream   — interleaved {t_in, a_in, v_in, t_out, a_out, v_out}
  2. BlockCausalAttention    — the key attention pattern (Sec 2.1)
  3. StreamingTransformer    — the unified Transformer with flow-matching heads
  4. ThinkerPerformerPipeline — the two-device inference split (Sec 2.4)
  5. FlowMatchingSolver      — conditional flow matching for latents (Eq 2-3)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ========================================================================= #
#  Constants & Enums                                                        #
# ========================================================================= #

# Modality token type IDs — each token in the interleaved sequence has a type.
MOD_TEXT_IN  = 0   # user language tokens (discrete)
MOD_AUDIO_IN = 1  # user audio tokens (discrete codec or continuous)
MOD_VIDEO_IN = 2  # user video tokens (discrete or continuous)
MOD_TEXT_OUT = 3  # agent language tokens (discrete, autoregressive)
MOD_AUDIO_OUT = 4 # agent audio latent (continuous, flow-matched)
MOD_VIDEO_OUT = 5 # agent video latent (continuous, flow-matched)

MOD_NAMES = {
    MOD_TEXT_IN: "text_in", MOD_AUDIO_IN: "audio_in", MOD_VIDEO_IN: "video_in",
    MOD_TEXT_OUT: "text_out", MOD_AUDIO_OUT: "audio_out", MOD_VIDEO_OUT: "video_out",
}

# Each streaming unit is 160ms at 25fps = 4 frames.
# In the toy, we represent each unit as a fixed number of tokens.
STREAMING_UNIT_DURATION_MS = 160  # ms per streaming unit
FPS = 25


# ========================================================================= #
#  Multimodal Token Stream                                                  #
# ========================================================================= #

@dataclass
class StreamingUnit:
    """
    One 160ms chunk of the interaction stream (Eq 1 in the paper).
    
    u_k = (u_k^t, u_k^a, u_k^v)  — user observations
    y_k = (y_k^t, y_k^a, y_k^v)  — agent response
    """
    step: int
    user_text: np.ndarray    # shape (n_text_in, d_model)
    user_audio: np.ndarray   # shape (n_audio_in, d_model)
    user_video: np.ndarray   # shape (n_video_in, d_model)
    agent_text: np.ndarray   # shape (n_text_out, d_model)
    agent_audio: np.ndarray   # shape (n_audio_out, d_model)  — latent
    agent_video: np.ndarray   # shape (n_video_out, d_model)  — latent
    
    @property
    def n_user_tokens(self) -> int:
        return (self.user_text.shape[0] + self.user_audio.shape[0] 
                + self.user_video.shape[0])
    
    @property
    def n_agent_tokens(self) -> int:
        return (self.agent_text.shape[0] + self.agent_audio.shape[0]
                + self.agent_video.shape[0])
    
    @property
    def total_tokens(self) -> int:
        return self.n_user_tokens + self.n_agent_tokens
    
    def get_interleaved_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the full interleaved token sequence and modality type array.
        
        Order: user_text, user_audio, user_video, agent_text, agent_audio, agent_video
        This matches the paper's interleaving scheme.
        """
        tokens_list = []
        types_list = []
        for arr, mod in [(self.user_text, MOD_TEXT_IN),
                         (self.user_audio, MOD_AUDIO_IN),
                         (self.user_video, MOD_VIDEO_IN),
                         (self.agent_text, MOD_TEXT_OUT),
                         (self.agent_audio, MOD_AUDIO_OUT),
                         (self.agent_video, MOD_VIDEO_OUT)]:
            if arr.shape[0] > 0:
                tokens_list.append(arr)
                types_list.append(np.full(arr.shape[0], mod, dtype=np.int32))
        return np.concatenate(tokens_list, axis=0), np.concatenate(types_list)


@dataclass
class MultimodalTokenStream:
    """
    Full conversation stream: a list of streaming units that together form
    the complete causal context c_k = {u_{<=k}^t, u_{<=k}^a, u_{<=k}^v,
                                       y_{<k}^t, y_{<k}^a, y_{<k}^v}.
    """
    units: List[StreamingUnit] = field(default_factory=list)
    d_model: int = 64
    
    def append(self, unit: StreamingUnit):
        self.units.append(unit)
    
    def get_full_sequence(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Concatenate all streaming units into one flat token sequence.
        Returns: (tokens [N, d], types [N], block_boundaries [n_blocks+1])
        """
        all_tokens = []
        all_types = []
        boundaries = [0]
        for unit in self.units:
            t, tp = unit.get_interleaved_sequence()
            all_tokens.append(t)
            all_types.append(tp)
            boundaries.append(boundaries[-1] + t.shape[0])
        return (np.concatenate(all_tokens) if all_tokens else np.zeros((0, self.d_model)),
                np.concatenate(all_types) if all_types else np.zeros(0, dtype=np.int32),
                boundaries)
    
    def n_blocks(self) -> int:
        return len(self.units)


# ========================================================================= #
#  Block-Causal Attention Mask                                              #
# ========================================================================= #

def build_block_causal_mask(
    block_sizes: List[int],
    n_output_per_block: List[int],
) -> np.ndarray:
    """
    Build the block-causal attention mask (Sec 2.1, Fig 1).
    
    Rules:
      - Input tokens within a block are FULLY VISIBLE (bidirectional within block).
      - Output tokens within a block are CAUSAL (can only see themselves and earlier).
      - Each token can attend to ALL tokens in ALL preceding blocks.
    
    This is the key architectural innovation: user observations within a 160ms
    chunk are processed bidirectionally (full self-attention within the block),
    while the agent response is generated autoregressively within the block.
    Both can see the complete past history across blocks.
    
    Args:
        block_sizes: number of tokens in each streaming unit block.
        n_output_per_block: number of output (agent) tokens per block.
    
    Returns:
        attention mask of shape (N, N) where N = sum(block_sizes).
        1.0 = allowed to attend, 0.0 = masked.
    """
    N = sum(block_sizes)
    mask = np.zeros((N, N), dtype=np.float32)
    
    pos = 0
    for b_idx, (b_size, n_out) in enumerate(zip(block_sizes, n_output_per_block)):
        n_in = b_size - n_out
        # Positions within this block
        block_start = pos
        block_end = pos + b_size
        input_end = pos + n_in  # input tokens end
        output_start = pos + n_in  # output tokens start
        
        # --- Row j (query) in this block ---
        for j in range(b_size):
            abs_j = block_start + j
            if j < n_in:
                # INPUT token: can see ALL past blocks + ALL tokens in current block
                # (bidirectional within block for input)
                for k in range(b_size):  # all in current block
                    mask[abs_j, block_start + k] = 1.0
            else:
                # OUTPUT token: can see ALL past blocks + all input in current block
                # + previous output tokens in current block (causal)
                for k in range(n_in):  # all input in current block
                    mask[abs_j, block_start + k] = 1.0
                for k in range(n_in, j + 1):  # output tokens up to and including self
                    mask[abs_j, block_start + k] = 1.0
        
        pos = block_end
    
    # Fill in cross-block attention: all tokens can see all past blocks
    pos = 0
    for b_idx, b_size in enumerate(block_sizes):
        block_start = pos
        block_end = pos + b_size
        # This block can attend to all previous blocks
        prev_end = 0
        for prev_idx in range(b_idx):
            prev_size = block_sizes[prev_idx]
            prev_start = prev_end
            prev_end = prev_start + prev_size
            mask[block_start:block_end, prev_start:prev_end] = 1.0
        pos = block_end
    
    return mask


# ========================================================================= #
#  Neural Network Building Blocks (numpy-only)                               #
# ========================================================================= #

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-8)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear projection: x @ w.T + b."""
    return x @ w.T + b


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Scaled dot-product attention.
    q: (N, d_k), k: (N, d_k), v: (N, d_v)
    mask: (N, N) broadcastable, 1=attend, 0=mask
    """
    d_k = q.shape[-1]
    scores = q @ k.T / math.sqrt(d_k)
    if mask is not None:
        scores = scores * mask + (1.0 - mask) * (-1e9)
    weights = softmax(scores, axis=-1)
    return weights @ v


class MultiHeadAttention:
    """Multi-head self-attention with block-causal mask support."""
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Xavier initialization
        scale = 1.0 / math.sqrt(d_model)
        self.w_q = np.random.randn(n_heads, d_model, self.d_k).astype(np.float32) * scale
        self.w_k = np.random.randn(n_heads, d_model, self.d_k).astype(np.float32) * scale
        self.w_v = np.random.randn(n_heads, d_model, self.d_k).astype(np.float32) * scale
        self.w_o = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.b_o = np.zeros(d_model, dtype=np.float32)
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        N = x.shape[0]
        # Project to multi-head Q, K, V
        qs = np.stack([x @ self.w_q[h] for h in range(self.n_heads)], axis=0)  # (H, N, d_k)
        ks = np.stack([x @ self.w_k[h] for h in range(self.n_heads)], axis=0)
        vs = np.stack([x @ self.w_v[h] for h in range(self.n_heads)], axis=0)
        
        # Apply attention per head
        head_outs = []
        for h in range(self.n_heads):
            out = attention(qs[h], ks[h], vs[h], mask)
            head_outs.append(out)
        
        # Concatenate heads and project
        concat = np.concatenate(head_outs, axis=-1)  # (N, d_model)
        return concat @ self.w_o + self.b_o


class FeedForward:
    """Position-wise feed-forward network (2-layer MLP with GELU)."""
    
    def __init__(self, d_model: int, d_ff: int):
        scale = 1.0 / math.sqrt(d_model)
        self.w1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.w2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale
        self.b2 = np.zeros(d_model, dtype=np.float32)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = gelu(x @ self.w1 + self.b1)  # (N, d_ff)
        return h @ self.w2 + self.b2  # (N, d_model)


class TransformerBlock:
    """Pre-norm Transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        scale = 1.0 / math.sqrt(d_model)
        self.ln1_w = np.ones(d_model, dtype=np.float32)
        self.ln1_b = np.zeros(d_model, dtype=np.float32)
        self.ln2_w = np.ones(d_model, dtype=np.float32)
        self.ln2_b = np.zeros(d_model, dtype=np.float32)
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = x + self.attn(layer_norm(x, self.ln1_w, self.ln1_b), mask)
        x = x + self.ffn(layer_norm(x, self.ln2_w, self.ln2_b))
        return x


class StreamingTransformer:
    """
    The unified Transformer backbone of Wan-Streamer.
    
    Takes the full interleaved multimodal sequence with a block-causal
    attention mask and produces:
      - Text output logits (for next-token prediction, cross-entropy loss)
      - Audio/video velocity predictions (for flow matching, Eq 3)
    
    Paper Sec 2.1:
      p_theta(y_{1:K} | u_{1:K}) = prod_k p_theta(y_k^t, y_k^a, y_k^v |
          u_{<=k}^t, u_{<=k}^a, u_{<=k}^v, y_{<k}^t, y_{<k}^a, y_{<k}^v)
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 256, vocab_size: int = 128):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        
        # Token embedding (shared across all modalities — in reality each
        # modality has its own encoder, but in this toy we use one embedding)
        scale = 1.0 / math.sqrt(d_model)
        self.token_emb = np.random.randn(vocab_size, d_model).astype(np.float32) * scale
        # Modality type embedding
        self.mod_emb = np.random.randn(6, d_model).astype(np.float32) * scale
        # Positional embedding (max 1024 positions)
        self.pos_emb = np.random.randn(1024, d_model).astype(np.float32) * scale
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        
        # Output heads
        # Text: logits for next-token prediction
        self.text_head_w = np.random.randn(d_model, vocab_size).astype(np.float32) * scale
        self.text_head_b = np.zeros(vocab_size, dtype=np.float32)
        # Audio/Video: velocity prediction for flow matching (Eq 3)
        self.audio_vel_w = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.audio_vel_b = np.zeros(d_model, dtype=np.float32)
        self.video_vel_w = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.video_vel_b = np.zeros(d_model, dtype=np.float32)
        
        # Final layer norm
        self.ln_w = np.ones(d_model, dtype=np.float32)
        self.ln_b = np.zeros(d_model, dtype=np.float32)
    
    def embed_tokens(self, token_ids: np.ndarray, mod_types: np.ndarray,
                     positions: np.ndarray) -> np.ndarray:
        """Embed tokens = token_emb + modality_emb + position_emb."""
        emb = self.token_emb[token_ids.astype(int)]  # (N, d)
        emb = emb + self.mod_emb[mod_types.astype(int)]
        emb = emb + self.pos_emb[positions.astype(int) % 1024]
        return emb
    
    def forward(self, token_ids: np.ndarray, mod_types: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Full forward pass.
        
        Args:
            token_ids: (N,) integer token IDs
            mod_types: (N,) modality type IDs
            mask: (N, N) block-causal attention mask
        
        Returns:
            dict with:
              'hidden': (N, d_model) final hidden states
              'text_logits': (N_text_out, vocab_size) for text output positions
              'audio_velocity': (N_audio_out, d_model) velocity prediction
              'video_velocity': (N_video_out, d_model) velocity prediction
        """
        N = len(token_ids)
        positions = np.arange(N, dtype=np.int32)
        
        x = self.embed_tokens(token_ids, mod_types, positions)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = layer_norm(x, self.ln_w, self.ln_b)
        
        # Extract outputs by modality
        text_out_mask = (mod_types == MOD_TEXT_OUT)
        audio_out_mask = (mod_types == MOD_AUDIO_OUT)
        video_out_mask = (mod_types == MOD_VIDEO_OUT)
        
        result = {'hidden': x}
        
        if np.any(text_out_mask):
            result['text_logits'] = x[text_out_mask] @ self.text_head_w + self.text_head_b
        if np.any(audio_out_mask):
            result['audio_velocity'] = x[audio_out_mask] @ self.audio_vel_w + self.audio_vel_b
        if np.any(video_out_mask):
            result['video_velocity'] = x[video_out_mask] @ self.video_vel_w + self.video_vel_b
        
        return result


# ========================================================================= #
#  Flow Matching Solver (Paper Eq 2-3)                                     #
# ========================================================================= #

class FlowMatchingSolver:
    """
    Conditional flow matching for audio/video latent generation.
    
    Paper Eq 2: z_tau^m = (1 - tau) * z_0^m + tau * epsilon^m
               dz_tau^m / d_tau = epsilon^m - z_0^m
    
    Paper Eq 3: L_FM^m = E_epsilon || f_theta(z_tau^a, z_tau^v, c_k, tau)
                                - dz_tau^m / d_tau ||_2^2
    
    At inference, we denoise from tau=1 (noise) to tau=0 (clean).
    """
    
    def __init__(self, n_steps: int = 5):
        self.n_steps = n_steps
    
    def add_noise(self, z0: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct noisy latent at flow time tau.
        Returns: (z_tau, epsilon)
        """
        eps = np.random.randn(*z0.shape).astype(np.float32)
        z_tau = (1.0 - tau) * z0 + tau * eps
        return z_tau, eps
    
    def compute_target_velocity(self, z0: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        """dz_tau / d_tau = epsilon - z0"""
        return epsilon - z0
    
    def training_loss(self, predicted_velocity: np.ndarray,
                      target_velocity: np.ndarray) -> float:
        """L_FM = || predicted - target ||_2^2"""
        return float(np.mean((predicted_velocity - target_velocity) ** 2))
    
    def denoise(self, model_fn, z_noisy: np.ndarray, context: np.ndarray,
                tau_schedule: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Euler-step denoising from tau=1 to tau=0.
        
        Args:
            model_fn: function(z_tau_a, z_tau_v, context) -> (vel_a, vel_v)
            z_noisy: (n_tokens, d_model) initial noisy latent
            context: (d_model,) conditioning context
            tau_schedule: array of tau values from 1->0
        
        Returns:
            z_clean: (n_tokens, d_model) denoised latent
        """
        if tau_schedule is None:
            tau_schedule = np.linspace(1.0, 0.0, self.n_steps + 1)
        
        z = z_noisy.copy()
        for i in range(len(tau_schedule) - 1):
            tau_curr = tau_schedule[i]
            tau_next = tau_schedule[i + 1]
            dt = tau_next - tau_curr
            
            # Predict velocity at current noise level
            vel = model_fn(z, tau_curr, context)
            
            # Euler step: z_{tau_next} = z_{tau_curr} + dt * dz/dtau
            z = z + dt * vel
        
        return z


# ========================================================================= #
#  Thinker-Performer Pipeline (Paper Sec 2.4)                               #
# ========================================================================= #

class Thinker:
    """
    Thinker GPU (Paper Sec 2.4):
      - Hosts causal audio/video encoders (in our toy: token embedding)
      - Runs the short token-causal Transformer path for language prediction
      - Constructs the KV-cache slice for the current streaming unit
      - Decodes audio/video latents (in toy: just identity)
      - Receives clean latents from performer
    
    At step k:
      1. Encode u_k (user observations) → token embeddings
      2. Run Transformer forward pass → text logits + state update
      3. Build KV slice for this block
      4. Send KV slice to performer
      5. Receive clean latents from performer (from step k-1)
      6. Decode clean latents → emit audio/video output
    """
    
    def __init__(self, model: StreamingTransformer):
        self.model = model
        self.latent_buffer = None  # received from performer
    
    def encode_observations(self, unit: StreamingUnit, token_ids: np.ndarray,
                            mod_types: np.ndarray) -> np.ndarray:
        """Embed the user observations for this streaming unit."""
        N = len(token_ids)
        positions = np.arange(N, dtype=np.int32)
        return self.model.embed_tokens(token_ids, mod_types, positions)
    
    def update_state(self, hidden: np.ndarray) -> np.ndarray:
        """Extract the KV-relevant hidden states for this block."""
        return hidden.copy()
    
    def decode_latents(self, audio_latent: np.ndarray,
                       video_latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode clean latents to audio/video. (Toy: identity + small transform.)"""
        # In the real model, this uses causal audio/video decoders.
        # Here we just add a learned (but random) affine transform.
        return audio_latent, video_latent
    
    def receive_from_performer(self, audio_latent: np.ndarray,
                                video_latent: np.ndarray):
        """Receive clean latents from the performer (from previous step)."""
        self.latent_buffer = (audio_latent, video_latent)


class Performer:
    """
    Performer GPU (Paper Sec 2.4):
      - Hosts only the latent generation path (flow-matching solver)
      - Receives KV-cache slice from thinker
      - Runs flow matching to denoise audio/video latents
      - Sends clean latents back to thinker at next step
    
    The performer is the throughput bottleneck: real-time operation requires
    performer_time + KV/latent_communication < 160ms streaming unit.
    """
    
    def __init__(self, model: StreamingTransformer, fm_solver: FlowMatchingSolver):
        self.model = model
        self.fm_solver = fm_solver
        self.kv_cache = None
    
    def receive_kv_slice(self, kv_slice: np.ndarray):
        """Append KV slice from thinker into full-history cache."""
        if self.kv_cache is None:
            self.kv_cache = kv_slice
        else:
            self.kv_cache = np.concatenate([self.kv_cache, kv_slice], axis=0)
    
    def generate_latents(self, context: np.ndarray, n_audio: int,
                         n_video: int, d_model: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run flow matching to generate clean audio and video latents.
        In the real model this is the expensive path (multiple solver steps).
        """
        # Start from noise
        z_audio = np.random.randn(n_audio, d_model).astype(np.float32)
        z_video = np.random.randn(n_video, d_model).astype(np.float32)
        
        # Simple model function using the transformer's velocity heads
        def model_fn(z: np.ndarray, tau: float, ctx: np.ndarray) -> np.ndarray:
            # Use the audio/video velocity heads conditioned on context
            # (simplified: just a linear combination)
            vel_a = z @ self.model.audio_vel_w.T + self.model.audio_vel_b
            vel_v = z @ self.model.video_vel_w.T + self.model.video_vel_b
            return vel_a  # return one modality at a time
        
        # Denoise
        z_audio_clean = self.fm_solver.denoise(
            model_fn, z_audio, context, np.linspace(1.0, 0.0, self.fm_solver.n_steps + 1))
        z_video_clean = self.fm_solver.denoise(
            model_fn, z_video, context, np.linspace(1.0, 0.0, self.fm_solver.n_steps + 1))
        
        return z_audio_clean, z_video_clean


class ThinkerPerformerPipeline:
    """
    Full thinker-performer inference pipeline (Paper Sec 2.4, Fig 2).
    
    Pipelines current-frame perception + state update (thinker),
    previous-frame audio/video decoding (thinker),
    KV/latent communication,
    and next-frame latent denoising (performer).
    """
    
    def __init__(self, model: StreamingTransformer, fm_solver: FlowMatchingSolver):
        self.thinker = Thinker(model)
        self.performer = Performer(model, fm_solver)
        self.model = model
        self.step_count = 0
        self.latency_log: List[float] = []
        # Accumulate committed streaming units for cross-block attention
        self.committed_units: List[StreamingUnit] = []
    
    def process_streaming_unit(self, unit: StreamingUnit) -> Dict:
        """
        Process one streaming unit through the full pipeline.
        Accumulates all previous units so the Transformer sees the full
        cross-block causal history (the key Wan-Streamer design).
        """
        import time
        step = self.step_count
        self.step_count += 1
        
        # Build the full sequence: ALL committed past units + current unit
        stream = MultimodalTokenStream(d_model=self.model.d_model)
        for prev_unit in self.committed_units:
            stream.append(prev_unit)
        stream.append(unit)
        
        tokens, types, boundaries = stream.get_full_sequence()
        N = len(tokens)
        
        if N == 0:
            return {'step': step, 'error': 'empty sequence'}
        
        # --- THINKER: Encode + State Update ---
        t0 = time.perf_counter()
        
        # Build block-causal attention mask across ALL blocks
        block_sizes = [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]
        n_output_per_block = [bu.n_agent_tokens for bu in stream.units]
        
        mask = build_block_causal_mask(block_sizes, n_output_per_block)
        
        # Fake token IDs (in real model these come from encoders/codecs)
        token_ids = np.random.randint(0, self.model.vocab_size, N).astype(np.int32)
        
        # Forward pass through unified Transformer
        outputs = self.model.forward(token_ids, types, mask)
        hidden = outputs['hidden']
        
        t_thinker = time.perf_counter() - t0
        
        # --- THINKER: Build KV slice for performer ---
        kv_slice = self.thinker.update_state(hidden)
        
        # --- THINKER: Decode previous latents (received from performer) ---
        if self.thinker.latent_buffer is not None:
            prev_audio, prev_video = self.thinker.latent_buffer
            decoded_audio, decoded_video = self.thinker.decode_latents(
                prev_audio, prev_video)
        else:
            decoded_audio = decoded_video = None
        
        # --- PERFORMER: Receive KV + Generate latents ---
        t1 = time.perf_counter()
        self.performer.receive_kv_slice(kv_slice)
        
        # Generate latents for next audio/video response
        n_audio_tokens = max(1, unit.agent_audio.shape[0])
        n_video_tokens = max(1, unit.agent_video.shape[0])
        context = np.mean(hidden, axis=0)  # mean-pool context
        audio_latent, video_latent = self.performer.generate_latents(
            context, n_audio_tokens, n_video_tokens, self.model.d_model)
        
        t_performer = time.perf_counter() - t1
        
        # --- COMMUNICATION: Send latents to thinker ---
        self.thinker.receive_from_performer(audio_latent, video_latent)
        
        # Commit this unit to history for next step's cross-block attention
        self.committed_units.append(unit)
        
        # --- METRICS ---
        total_latency_ms = (t_thinker + t_performer) * 1000
        
        return {
            'step': step,
            'total_tokens': N,
            'n_blocks': len(block_sizes),
            'thinker_ms': t_thinker * 1000,
            'performer_ms': t_performer * 1000,
            'total_ms': total_latency_ms,
            'decoded_audio': decoded_audio,
            'decoded_video': decoded_video,
            'audio_latent_shape': audio_latent.shape,
            'video_latent_shape': video_latent.shape,
            'hidden_norm': float(np.mean(np.linalg.norm(hidden, axis=-1))),
        }
