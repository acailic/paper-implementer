"""
PhysisForcing Core Model Implementation.

Implements:
- Small DiT for video generation via flow matching
- Lightweight frozen video encoder (V-JEPA-style)
- Physics-informative region mask extraction
- Pixel-level trajectory alignment loss (L_pix)
- Semantic-level relational alignment loss (L_sem)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
#  Helpers
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ============================================================================
#  DiT (Diffusion Transformer) for Video Generation
# ============================================================================

class DiTBlock(nn.Module):
    """Single DiT transformer block with self-attention."""
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # x: (B, N, C) — N = T*H*W tokens
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class SmallDiT(nn.Module):
    """
    Small Diffusion Transformer for video generation via flow matching.
    
    Generates video of shape (T, C_img, H, W) from Gaussian noise.
    Uses patch-based tokenization and self-attention over space-time tokens.
    """
    def __init__(
        self,
        img_size=64,
        patch_size=4,
        in_channels=3,
        out_channels=3,
        dim=192,
        n_blocks=8,
        n_heads=6,
        n_frames=16,
        mlp_ratio=4.0,
        middle_block_idx=None,  # which block to extract features from
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.n_frames = n_frames
        self.n_patches = (img_size // patch_size) ** 2
        self.n_tokens = n_frames * self.n_patches

        # Middle block index (default: half-way)
        if middle_block_idx is None:
            self.middle_block_idx = n_blocks // 2
        else:
            self.middle_block_idx = middle_block_idx

        # Patchify
        self.patchify = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

        # Spatial + temporal positional embeddings
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.n_tokens, dim) * 0.02
        )

        # Timestep embedding
        self.time_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, n_heads, mlp_ratio) for _ in range(n_blocks)
        ])

        # Final norm and head
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * patch_size * patch_size)

    def forward(self, x_noisy, t):
        """
        Args:
            x_noisy: (B, T, C, H, W) noisy video
            t: (B,) diffusion timestep
            
        Returns:
            v_pred: (B, T, C, H, W) predicted velocity (for flow matching)
            middle_features: (B, N, D) features from middle block
        """
        B, T, C, H, W = x_noisy.shape

        # Merge batch and time for patchification
        x = rearrange(x_noisy, 'b t c h w -> (b t) c h w')
        x = self.patchify(x)  # (B*T, D, h, w)
        x = rearrange(x, '(b t) d h w -> b (t h w) d', b=B, t=T)

        # Add positional embeddings
        x = x + self.pos_emb

        # Timestep conditioning (additive)
        t_emb = self.time_mlp(self.time_emb(t))  # (B, D)
        x = x + t_emb[:, None, :]

        # DiT blocks — extract middle features
        middle_features = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.middle_block_idx:
                middle_features = x.clone()

        # Final norm and predict velocity
        x = self.final_norm(x)
        x = self.head(x)  # (B, N, C*p*p)
        p = self.patch_size
        h = w = self.img_size // p
        x = rearrange(x, 'b (t h w) (c p1 p2) -> b t c (h p1) (w p2)',
                       t=T, h=h, w=w, c=self.out_channels, p1=p, p2=p)

        return x, middle_features


# ============================================================================
#  Frozen Video Encoder (V-JEPA-style)
# ============================================================================

class FrozenVideoEncoder(nn.Module):
    """
    Lightweight spatial-patch video encoder.
    Processes each frame independently with a shared ViT encoder,
    producing spatio-temporal token features.
    Acts as the frozen Φ_u in the paper.
    """
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        dim=128,
        n_blocks=4,
        n_heads=4,
        mlp_ratio=4.0,
        n_frames=16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_frames = n_frames
        self.n_spatial_patches = (img_size // patch_size) ** 2

        self.patchify = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.n_spatial_patches, dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            DiTBlock(dim, n_heads, mlp_ratio) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W)
        Returns:
            features: (B, T, n_spatial_patches, D)
        """
        B, T, C, H, W = video.shape
        x = rearrange(video, 'b t c h w -> (b t) c h w')
        x = self.patchify(x)
        x = rearrange(x, '(b t) d h w -> b t (h w) d', b=B, t=T)
        x = x + self.pos_emb[None, None, :, :]  # broadcast over B, T

        for block in self.blocks:
            x_flat = rearrange(x, 'b t n d -> (b t) n d')
            x_flat = block(x_flat)
            x = rearrange(x_flat, '(b t) n d -> b t n d', b=B, t=T)

        x = self.norm(x)
        return x  # (B, T, N_spatial, D)


# ============================================================================
#  Physics-Informative Region Mask
# ============================================================================

def extract_physics_mask(
    trajectories,  # (B, N_pts, T, 2) — ground-truth point trajectories
    depth_map,      # (B, H, W) — depth map of first frame
    H=64, W=64,
):
    """
    Extract physics-informative region mask following the paper's method.

    Steps:
    1. Motion score: a_i = Σ_t ||p_i^{t+1} - p_i^t||_2
    2. Foreground weight: r_i = 1 / (D_0(p_i^0) + eps)
    3. Physics score: q_i = a_i * r_i
    4. Adaptive threshold: M_i = I(q_i >= mean(q))
    5. Rasterize to spatiotemporal mask

    Args:
        trajectories: (B, N_pts, T, 2) float — point tracks
        depth_map: (B, H, W) float — depth (0=near, 1=far)
        H, W: spatial resolution

    Returns:
        mask: (B, T, H, W) binary — physics-informative regions
    """
    B, N_pts, T, _ = trajectories.shape
    device = trajectories.device

    # Step 1: Motion score
    displacements = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]  # (B, N, T-1, 2)
    motion_scores = displacements.norm(dim=-1).sum(dim=-1)  # (B, N_pts)

    # Step 2: Foreground weight from depth at initial positions
    # trajectories[:, :, 0, :] gives (x, y) coordinates in [0, H-1] x [0, W-1]
    init_pts = trajectories[:, :, 0, :]  # (B, N_pts, 2) — (x, y)
    x_coords = init_pts[:, :, 0].long().clamp(0, W - 1)  # (B, N_pts)
    y_coords = init_pts[:, :, 1].long().clamp(0, H - 1)  # (B, N_pts)

    # Sample depth at initial positions
    depth_at_pts = torch.stack([
        depth_map[b, y_coords[b], x_coords[b]] for b in range(B)
    ], dim=0)  # (B, N_pts)

    eps = 1e-4
    fg_weights = 1.0 / (depth_at_pts + eps)  # (B, N_pts)

    # Step 3: Physics score
    physics_scores = motion_scores * fg_weights  # (B, N_pts)

    # Step 4: Adaptive threshold
    threshold = physics_scores.mean(dim=-1, keepdim=True)  # (B, 1)
    point_mask = (physics_scores >= threshold).float()  # (B, N_pts)

    # Step 5: Rasterize to spatiotemporal mask
    # For each frame, mark the pixels near tracked points that are active
    mask = torch.zeros(B, T, H, W, device=device)
    for b in range(B):
        for i in range(N_pts):
            if point_mask[b, i] > 0.5:
                for t in range(T):
                    px = int(trajectories[b, i, t, 0].item())
                    py = int(trajectories[b, i, t, 1].item())
                    # Mark a small region around the point (±2 pixels for robustness)
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                mask[b, t, ny, nx] = 1.0

    return mask  # (B, T, H, W)


# ============================================================================
#  Trajectory Prediction MLP (φ for L_pix)
# ============================================================================

class TrajectoryMLP(nn.Module):
    """
    MLP φ(·) that refines DiT middle-block features into per-frame spatial features
    for trajectory prediction via cross-frame similarity.
    """
    def __init__(self, in_dim, hidden_dim=256, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
#  Semantic Alignment MLP (ψ for L_sem)
# ============================================================================

class SemanticMLP(nn.Module):
    """
    MLP ψ(·) that projects DiT middle-block features to match the frozen encoder
    feature dimension for relational alignment.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
#  Pixel-Level Trajectory Alignment Loss
# ============================================================================

def compute_pixel_trajectory_loss(
    dit_features,    # (B, N_tokens, D) from middle DiT block
    trajectory_mlp,   # nn.Module
    trajectories_gt,  # (B, N_pts, T, 2) ground-truth point tracks
    physics_mask,     # (B, T, H, W) binary physics mask
    T, H, W,
    dit_patch_size=4,
    n_query_points=64,  # number of query points to sample
):
    """
    Compute L_pix — pixel-level trajectory alignment loss.

    1. Refine DiT features with MLP φ(·)
    2. Reshape to (B, T, C, h_patch, w_patch)
    3. Use first-frame features as queries, remaining frames as keys
    4. Compute similarity maps → softmax → coordinate expectation → predicted positions
    5. Masked MSE against ground-truth trajectories

    Args:
        dit_features: (B, N_tokens, D) middle block features
        trajectory_mlp: MLP φ(·)
        trajectories_gt: (B, N_pts, T, 2) GT tracks
        physics_mask: (B, T, H, W)
        T: number of frames
        H, W: spatial resolution
        dit_patch_size: DiT patch size
        n_query_points: number of query points

    Returns:
        loss: scalar
        pred_trajectories: (B, N_pts, T, 2) predicted trajectories
    """
    B = dit_features.shape[0]
    device = dit_features.device
    h_patch = H // dit_patch_size
    w_patch = W // dit_patch_size
    D = dit_features.shape[-1]

    # Refine features
    refined = trajectory_mlp(dit_features)  # (B, N_tokens, D)

    # Reshape to (B, T, D, h_patch, w_patch)
    n_spatial = h_patch * w_patch
    refined = rearrange(refined, 'b (t n) d -> b t d n', t=T, n=n_spatial)
    refined = rearrange(refined, 'b t d n -> b t d h w', h=h_patch, w=w_patch)

    # Normalize features for similarity computation
    refined = F.normalize(refined, dim=2)

    # Query from first frame, keys from remaining frames
    query = refined[:, 0:1]  # (B, 1, D, h_patch, w_patch)
    keys = refined[:, 1:]   # (B, T-1, D, h_patch, w_patch)

    # Sample query points from physics mask (frame 0)
    pred_trajectories = torch.zeros(B, n_query_points, T, 2, device=device)
    gt_points = trajectories_gt[:, :n_query_points]  # (B, N_pts, T, 2)

    total_loss = 0.0
    n_valid = 0

    for t_idx in range(T - 1):
        key_t = keys[:, t_idx]  # (B, D, h_patch, w_patch)
        query_t = query[:, 0]   # (B, D, h_patch, w_patch)

        # Compute similarity map: (B, h_patch*w_patch, h_patch*w_patch)
        q_flat = rearrange(query_t, 'b d h w -> b d (h w)')  # (B, D, N)
        k_flat = rearrange(key_t, 'b d h w -> b d (h w)')    # (B, D, N)
        sim = torch.bmm(q_flat.transpose(1, 2), k_flat) / math.sqrt(D)  # (B, N, N)

        # Softmax over spatial locations → predicted coordinates
        sim_softmax = F.softmax(sim, dim=-1)  # (B, N_q, N_k)

        # Coordinate grid in pixel space (center of each patch)
        y_grid = torch.arange(h_patch, device=device).float() * dit_patch_size + dit_patch_size / 2
        x_grid = torch.arange(w_patch, device=device).float() * dit_patch_size + dit_patch_size / 2
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        coord_grid = torch.stack([xx, yy], dim=-1)  # (h_patch, w_patch, 2)

        # Predicted position = weighted sum of coordinates
        pred_pos = torch.einsum('bnm,mxy->bny', sim_softmax,
                                coord_grid.reshape(-1, 2))  # (B, N_q, 2)

        pred_trajectories[:, :, t_idx + 1] = pred_pos

        # GT positions for frame t_idx + 1
        gt_pos = gt_points[:, :, t_idx + 1]  # (B, N_pts, 2)

        # Mask: only count points where the physics mask is active
        for b_idx in range(B):
            # Check if GT points are within bounds
            gt_x = gt_pos[b_idx, :, 0].clamp(0, W - 1).long()
            gt_y = gt_pos[b_idx, :, 1].clamp(0, H - 1).long()
            mask_vals = physics_mask[b_idx, t_idx + 1, gt_y, gt_x]  # (N_pts,)
            if mask_vals.sum() > 0:
                diff = (pred_pos[b_idx] - gt_pos[b_idx]).norm(dim=-1)  # (N_pts,)
                total_loss += (mask_vals * diff ** 2).sum()
                n_valid += mask_vals.sum().item()

        # Also set frame 0 predicted positions to GT (trivially correct)
    pred_trajectories[:, :, 0] = gt_points[:, :, 0]

    if n_valid > 0:
        loss = total_loss / n_valid
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss, pred_trajectories


# ============================================================================
#  Semantic-Level Relational Alignment Loss
# ============================================================================

def compute_semantic_relational_loss(
    dit_features,     # (B, N_tokens, D_dit) from middle DiT block
    semantic_mlp,     # nn.Module ψ(·)
    encoder_features, # (B, T, N_spatial, D_enc) from frozen encoder
    physics_mask,     # (B, T, H, W)
    T, H, W,
    dit_patch_size=4,
    encoder_patch_size=8,
    max_tokens=512,
):
    """
    Compute L_sem — semantic-level relational alignment loss.

    1. Project DiT features with MLP ψ(·) to match encoder dimension
    2. Resample DiT tokens to match encoder's spatial layout
    3. Select K tokens based on physics mask
    4. Compute pairwise cosine similarity matrices for both sets
    5. L1 loss between matrices

    Args:
        dit_features: (B, N_tokens, D_dit)
        semantic_mlp: MLP ψ(·) mapping D_dit → D_enc
        encoder_features: (B, T, N_spatial_enc, D_enc)
        physics_mask: (B, T, H, W)
        T: frames
        H, W: spatial resolution
        dit_patch_size: DiT patch size
        encoder_patch_size: encoder patch size
        max_tokens: max number of selected tokens K

    Returns:
        loss: scalar
    """
    B = dit_features.shape[0]
    device = dit_features.device

    # Project DiT features to encoder dimension
    dit_proj = semantic_mlp(dit_features)  # (B, N_tokens, D_enc)

    # Reshape DiT features to (B, T, n_dit_patches, D_enc)
    h_dit = H // dit_patch_size
    w_dit = W // dit_patch_size
    n_dit_spatial = h_dit * w_dit
    dit_proj = rearrange(dit_proj, 'b (t n) d -> b t n d', t=T, n=n_dit_spatial)

    # Resample DiT spatial patches to match encoder spatial layout
    h_enc = H // encoder_patch_size
    w_enc = W // encoder_patch_size

    # Reshape to spatial grid and interpolate
    dit_proj = rearrange(dit_proj, 'b t (h w) d -> b t h w d',
                          h=h_dit, w=w_dit)
    dit_proj = rearrange(dit_proj, 'b t h w d -> (b t) d h w')
    dit_proj = F.interpolate(dit_proj, size=(h_enc, w_enc), mode='bilinear',
                             align_corners=False)
    dit_proj = rearrange(dit_proj, '(b t) d h w -> b t (h w) d', b=B, t=T)

    # Now dit_proj and encoder_features have same shape: (B, T, N_enc_spatial, D_enc)
    dit_proj = dit_proj[:, :, :encoder_features.shape[2], :]

    # Build physics mask at encoder spatial resolution
    # Resize physics_mask from (B, T, H, W) to (B, T, h_enc, w_enc)
    mask_resized = rearrange(physics_mask, 'b t h w -> (b t) 1 h w')
    mask_resized = F.interpolate(mask_resized.float(), size=(h_enc, w_enc),
                                  mode='nearest')
    mask_resized = rearrange(mask_resized, '(b t) 1 h w -> b t (h w)', b=B, t=T)

    # Select K tokens per sample based on physics mask
    all_losses = []
    for b_idx in range(B):
        mask_t = mask_resized[b_idx]  # (T, N_enc)
        dit_t = dit_proj[b_idx]       # (T, N_enc, D)
        enc_t = encoder_features[b_idx]  # (T, N_enc, D)

        # Flatten time dimension: (T*N_enc, D)
        mask_flat = mask_t.reshape(-1)  # (T*N_enc,)
        dit_flat = dit_t.reshape(-1, dit_t.shape[-1])  # (T*N_enc, D)
        enc_flat = enc_t.reshape(-1, enc_t.shape[-1])  # (T*N_enc, D)

        # Select top-K tokens by mask score
        n_total = mask_flat.shape[0]
        n_select = min(max_tokens, mask_flat.sum().long().item())
        if n_select < 2:
            continue

        # Get indices where mask is active
        active_idx = mask_flat.nonzero(as_tuple=True)[0]
        if active_idx.shape[0] > max_tokens:
            perm = torch.randperm(active_idx.shape[0], device=device)[:max_tokens]
            active_idx = active_idx[perm]

        dit_selected = dit_flat[active_idx]  # (K, D)
        enc_selected = enc_flat[active_idx]  # (K, D)

        # Normalize
        dit_selected = F.normalize(dit_selected, dim=-1)
        enc_selected = F.normalize(enc_selected, dim=-1)

        # Pairwise cosine similarity matrices
        R_hat = torch.mm(dit_selected, dit_selected.t())  # (K, K)
        R = torch.mm(enc_selected, enc_selected.t())      # (K, K)

        # L1 loss
        loss_b = (R_hat - R).abs().mean()
        all_losses.append(loss_b)

    if all_losses:
        loss = torch.stack(all_losses).mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


# ============================================================================
#  Flow Matching Utilities
# ============================================================================

def compute_flow_matching_loss(model, video, condition=None):
    """
    Compute L_FM — flow matching loss.
    Uses rectified flow: x_t = (1-t) * x_0 + t * noise, v = noise - x_0.
    
    Args:
        model: SmallDiT
        video: (B, T, C, H, W) clean video
        condition: optional conditioning (unused in this simplified version)
    
    Returns:
        loss: scalar flow matching loss
        v_pred: predicted velocity
        v_target: target velocity
        t_sampled: sampled timesteps
    """
    B, T, C, H, W = video.shape
    device = video.device

    # Sample timesteps uniformly in [0, 1]
    t = torch.rand(B, device=device)

    # Sample noise
    noise = torch.randn_like(video)

    # Interpolate: x_t = (1-t) * x_0 + t * noise
    t_expand = t.view(B, 1, 1, 1, 1)
    x_t = (1 - t_expand) * video + t_expand * noise

    # Target velocity: v = noise - x_0 (rectified flow)
    v_target = noise - video

    # Predict velocity
    v_pred, middle_features = model(x_t, t)

    # MSE loss on velocity prediction
    loss = F.mse_loss(v_pred, v_target)

    return loss, v_pred, v_target, t, middle_features


# ============================================================================
#  Main PhysisForcing Model Wrapper
# ============================================================================

class PhysisForcingModel(nn.Module):
    """
    Complete PhysisForcing training model combining:
    - DiT with flow matching (L_FM)
    - Physics-informative region mask
    - Pixel-level trajectory alignment (L_pix)
    - Semantic-level relational alignment (L_sem)
    """
    def __init__(
        self,
        img_size=64,
        dit_patch_size=4,
        encoder_patch_size=8,
        in_channels=3,
        dit_dim=192,
        dit_n_heads=6,
        dit_n_blocks=8,
        encoder_dim=128,
        encoder_n_heads=4,
        encoder_n_blocks=4,
        n_frames=16,
        mlp_hidden=256,
        n_query_points=64,
        max_sem_tokens=512,
        lambda_pix=1.0,
        lambda_sem=0.5,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.img_size = img_size
        self.dit_patch_size = dit_patch_size
        self.encoder_patch_size = encoder_patch_size
        self.n_query_points = n_query_points
        self.max_sem_tokens = max_sem_tokens
        self.lambda_pix = lambda_pix
        self.lambda_sem = lambda_sem

        # DiT (trainable)
        self.dit = SmallDiT(
            img_size=img_size,
            patch_size=dit_patch_size,
            in_channels=in_channels,
            out_channels=in_channels,
            dim=dit_dim,
            n_heads=dit_n_heads,
            n_blocks=dit_n_blocks,
            n_frames=n_frames,
        )

        # Frozen video encoder
        self.encoder = FrozenVideoEncoder(
            img_size=img_size,
            patch_size=encoder_patch_size,
            in_channels=in_channels,
            dim=encoder_dim,
            n_heads=encoder_n_heads,
            n_blocks=encoder_n_blocks,
            n_frames=n_frames,
        )
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Alignment MLPs
        self.trajectory_mlp = TrajectoryMLP(dit_dim, hidden_dim=mlp_hidden)
        self.semantic_mlp = SemanticMLP(dit_dim, encoder_dim, hidden_dim=mlp_hidden)

        # Encoder feature dimension
        self.encoder_dim = encoder_dim

    def compute_total_loss(self, video, trajectories, depth_map):
        """
        Compute total training loss: L = L_FM + λ_pix * L_pix + λ_sem * L_sem

        Args:
            video: (B, T, C, H, W) clean video
            trajectories: (B, N_pts, T, 2) ground-truth point tracks
            depth_map: (B, H, W) depth map

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
            middle_features: (B, N, D) from middle DiT block
        """
        B, T, C, H, W = video.shape

        # --- Flow matching loss ---
        loss_fm, v_pred, v_target, t_sampled, middle_features = \
            compute_flow_matching_loss(self.dit, video)

        # --- Physics mask ---
        with torch.no_grad():
            physics_mask = extract_physics_mask(
                trajectories, depth_map, H=H, W=W
            )

        # --- Pixel-level trajectory alignment ---
        loss_pix, pred_trajs = compute_pixel_trajectory_loss(
            dit_features=middle_features,
            trajectory_mlp=self.trajectory_mlp,
            trajectories_gt=trajectories,
            physics_mask=physics_mask,
            T=T, H=H, W=W,
            dit_patch_size=self.dit_patch_size,
            n_query_points=self.n_query_points,
        )

        # --- Semantic-level relational alignment ---
        with torch.no_grad():
            encoder_features = self.encoder(video)  # (B, T, N_spatial, D_enc)

        loss_sem = compute_semantic_relational_loss(
            dit_features=middle_features,
            semantic_mlp=self.semantic_mlp,
            encoder_features=encoder_features,
            physics_mask=physics_mask,
            T=T, H=H, W=W,
            dit_patch_size=self.dit_patch_size,
            encoder_patch_size=self.encoder_patch_size,
            max_tokens=self.max_sem_tokens,
        )

        # --- Total loss ---
        total_loss = (
            loss_fm
            + self.lambda_pix * loss_pix
            + self.lambda_sem * loss_sem
        )

        loss_dict = {
            'total': total_loss.item(),
            'fm': loss_fm.item(),
            'pix': loss_pix.item(),
            'sem': loss_sem.item(),
        }

        return total_loss, loss_dict, middle_features

    @torch.no_grad()
    def generate(self, n_samples=4, n_steps=10, guidance_scale=1.0):
        """
        Generate video samples via Euler integration of the flow ODE.
        
        Args:
            n_samples: number of videos to generate
            n_steps: number of Euler steps
            guidance_scale: classifier-free guidance scale (unused here)
        
        Returns:
            video: (n_samples, T, C, H, W)
        """
        device = next(self.dit.parameters()).device
        T, H, W = self.n_frames, self.img_size, self.img_size

        # Start from pure noise
        x = torch.randn(n_samples, T, 3, H, W, device=device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((n_samples,), step * dt, device=device)
            v_pred, _ = self.dit(x, t)
            x = x + dt * v_pred

        return x.clamp(-1, 1)

    def trainables(self):
        """Return list of trainable parameters (DiT + alignment MLPs)."""
        params = list(self.dit.parameters())
        params += list(self.trajectory_mlp.parameters())
        params += list(self.semantic_mlp.parameters())
        return params

    def n_params(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
