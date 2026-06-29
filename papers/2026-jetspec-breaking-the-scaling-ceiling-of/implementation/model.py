"""
JetSpec: Causal Parallel Draft Head for Speculative Decoding.

Implements:
  - TargetModel: small autoregressive transformer (frozen during draft training)
  - DraftHead: causal-parallel draft head with tree-causal attention mask
  - Tree construction: best-first expansion with priority queue
  - Verification: greedy acceptance of draft tokens against target model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import heapq


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TargetConfig:
    vocab_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1

@dataclass
class DraftConfig:
    vocab_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1
    # Tree construction params
    max_depth: int = 8
    branching_width: int = 4
    node_budget: int = 31  # total nodes in tree (including root)
    # Layers from target to fuse (indices)
    fuse_layers: tuple = (0, 1, 2, 3)
    # Distillation temperature
    temp_kd: float = 1.5


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: TargetConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Target Model (autoregressive transformer)
# ---------------------------------------------------------------------------

class TargetModel(nn.Module):
    """Small autoregressive transformer used as the target model."""

    def __init__(self, config: TargetConfig):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._causal_mask = None

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) != T or self._causal_mask.device != device:
            mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
            self._causal_mask = mask
        return self._causal_mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token ids
        Returns:
            logits: (B, T, V)
            all_hidden: list of (B, T, D) from each layer
        """
        B, T = input_ids.shape
        device = input_ids.device
        h = self.tok_embed(input_ids)
        h = self.pos_embed(h)

        causal_mask = self._get_causal_mask(T, device)
        all_hidden = []
        for layer in self.layers:
            h = layer(h, attn_mask=causal_mask)
            all_hidden.append(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, all_hidden

    def forward_with_hidden(self, input_ids: torch.Tensor, layer_indices: Optional[List[int]] = None):
        """Forward pass returning hidden states at specified layers."""
        B, T = input_ids.shape
        device = input_ids.device
        h = self.tok_embed(input_ids)
        h = self.pos_embed(h)

        causal_mask = self._get_causal_mask(T, device)
        selected_hidden = []
        for i, layer in enumerate(self.layers):
            h = layer(h, attn_mask=causal_mask)
            if layer_indices is not None and i in layer_indices:
                selected_hidden.append(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, selected_hidden


# ---------------------------------------------------------------------------
# Feature Fusion (from target model hidden states)
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    """
    Fuse hidden states from selected target model layers via concatenation + projection.
    h_x^o = LayerNorm(W_proj @ concat(h^{(l1)}, h^{(l2)}, ...))
    """

    def __init__(self, d_model: int, fuse_layers: tuple, n_fuse_layers: int):
        super().__init__()
        self.fuse_layers = fuse_layers
        self.n_fuse_layers = n_fuse_layers
        self.proj = nn.Linear(d_model * n_fuse_layers, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_list: list of (B, T, D) from selected target layers
        Returns:
            fused: (B, T, D)
        """
        # hidden_list has length = len(fuse_layers), already in order
        concatenated = torch.cat(hidden_list, dim=-1)  # (B, T, D * n_layers)
        fused = self.proj(concatenated)
        fused = self.norm(fused)
        return fused


# ---------------------------------------------------------------------------
# Draft Head Layer (with optional cross-attention to fused features)
# ---------------------------------------------------------------------------

class DraftHeadLayer(nn.Module):
    """Single layer of the draft head with tree-causal attention."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ln3 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        fused_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) draft token embeddings
            attn_mask: (T, T) tree-causal attention mask (0 or -inf)
            fused_features: (B, T_prefix, D) fused target features (cross-attend)
        Returns:
            out: (B, T, D)
        """
        # Self-attention with tree-causal mask
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out

        # Cross-attention to fused target features
        residual = x
        x = self.ln2(x)
        cross_out, _ = self.cross_attn(x, fused_features, fused_features)
        x = residual + cross_out

        # FFN
        x = x + self.ff(self.ln3(x))
        return x


# ---------------------------------------------------------------------------
# Draft Head
# ---------------------------------------------------------------------------

class DraftHead(nn.Module):
    """
    Causal-parallel draft head with tree-causal attention mask.
    All tree nodes are predicted in a single forward pass.
    """

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        n_fuse_layers = len(config.fuse_layers)
        self.fusion = FeatureFusion(config.d_model, config.fuse_layers, n_fuse_layers)
        self.layers = nn.ModuleList([DraftHeadLayer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        draft_tokens: torch.Tensor,
        tree_causal_mask: torch.Tensor,
        target_hidden_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            draft_tokens: (B, T_draft) draft token ids (tree nodes in order)
            tree_causal_mask: (T_draft + T_prefix, T_draft + T_prefix) combined mask
                              rows for draft tokens have tree-causal structure
            target_hidden_list: list of (B, T_prefix, D) hidden states from selected target layers
        Returns:
            logits: (B, T_draft, V) logits for each draft position
            draft_probs: (B, T_draft, V) softmax probabilities
        """
        B, T_draft = draft_tokens.shape
        device = draft_tokens.device

        # Fuse target features
        fused = self.fusion(target_hidden_list)  # (B, T_prefix, D)

        # Embed draft tokens
        h = self.tok_embed(draft_tokens)  # (B, T_draft, D)
        h = self.pos_embed(h)

        # Concatenate prefix fused features as context (for combined mask to work)
        # Actually, we keep them separate: cross-attention to fused, self-attention among draft nodes
        # The tree-causal mask applies to self-attention over draft nodes only
        # For the self-attention mask: draft nodes attend to prefix (fused) + ancestors
        # We implement this by including prefix positions in the self-attention
        # with full bidirectional mask among prefix, and tree-causal from draft to prefix+ancestors

        T_prefix = fused.size(1)
        T_total = T_prefix + T_draft

        # Build self-attention input: [prefix features | draft embeddings]
        self_attn_input = torch.cat([fused, h], dim=1)  # (B, T_total, D)

        # Build mask: (T_total, T_total)
        # - prefix can attend to all prefix (bidirectional among prefix)
        # - draft nodes attend to: all prefix + ancestors only (tree-causal)
        # tree_causal_mask should be (T_total, T_total) already
        # We'll build it here from the node structure

        # For each draft head layer
        for layer in self.layers:
            self_attn_input = layer(self_attn_input, attn_mask=tree_causal_mask, fused_features=fused)

        # Extract draft positions only
        h_draft = self_attn_input[:, T_prefix:]  # (B, T_draft, D)
        h_draft = self.ln_f(h_draft)
        logits = self.head(h_draft)  # (B, T_draft, V)

        # Temperature-scaled softmax for draft probabilities
        temp = self.config.temp_kd
        draft_probs = F.softmax(logits / temp, dim=-1)

        return logits, draft_probs


# ---------------------------------------------------------------------------
# Tree-Causal Attention Mask Builder
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """Node in the draft tree."""
    token_id: int = 0
    parent_idx: int = -1  # -1 for root
    depth: int = 0
    log_prob: float = 0.0  # accumulated log prob of this branch
    draft_pos: int = -1   # position in the flat draft sequence
    children: List[int] = field(default_factory=list)

    # For priority queue: compare by -log_prob (max-heap via min-heap with negation)
    def __lt__(self, other):
        return self.log_prob > other.log_prob  # higher log prob = higher priority


def build_tree_causal_mask(
    tree_nodes: List[TreeNode],
    n_prefix: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the tree-causal attention mask.

    Layout: [prefix_tokens | tree_nodes]
    - Prefix tokens attend bidirectionally to all prefix tokens
    - Prefix tokens CANNOT attend to draft tokens (they don't need to)
    - Draft token at position i attends to: all prefix tokens + ancestors in tree

    Returns: (T_total, T_total) float mask (0.0 or -inf)
    """
    n_draft = len(tree_nodes)
    T_total = n_prefix + n_draft
    mask = torch.full((T_total, T_total), float("-inf"), device=device)

    # Prefix tokens attend to all prefix tokens (bidirectional among prefix)
    mask[:n_prefix, :n_prefix] = 0.0
    # Draft tokens attend to all prefix tokens
    mask[n_prefix:, :n_prefix] = 0.0

    # For each draft node, allow it to attend to its ancestors
    for i, node in enumerate(tree_nodes):
        pos_i = n_prefix + i
        # Self-attention
        mask[pos_i, pos_i] = 0.0
        # Walk up the tree to find ancestors
        ancestor_idx = node.parent_idx
        visited = set()
        while ancestor_idx >= 0 and ancestor_idx not in visited:
            visited.add(ancestor_idx)
            ancestor_pos = n_prefix + ancestor_idx
            mask[pos_i, ancestor_pos] = 0.0
            ancestor_node = tree_nodes[ancestor_idx]
            ancestor_idx = ancestor_node.parent_idx

    return mask


def flatten_tree_to_tokens(tree_nodes: List[TreeNode]) -> List[int]:
    """Get the flat sequence of draft tokens in tree order."""
    return [node.token_id for node in tree_nodes]


def get_ancestor_indices(tree_nodes: List[TreeNode], node_idx: int) -> List[int]:
    """Get indices of ancestors of a tree node (not including self)."""
    ancestors = []
    current = tree_nodes[node_idx].parent_idx
    while current >= 0:
        ancestors.append(current)
        current = tree_nodes[current].parent_idx
    return ancestors


# ---------------------------------------------------------------------------
# Best-First Tree Construction
# ---------------------------------------------------------------------------

def construct_draft_tree(
    draft_head: DraftHead,
    target_model: TargetModel,
    prefix_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    tree_nodes: List[TreeNode],
    config: DraftConfig,
    device: torch.device,
) -> Tuple[List[TreeNode], List[TreeNode]]:
    """
    Best-first tree construction using a priority queue.

    Args:
        draft_head: the draft head model
        target_model: the target model
        prefix_ids: (1, T_prefix) token ids for the prefix
        draft_logits: (1, T_draft, V) logits from draft head
        draft_probs: (1, T_draft, V) probabilities from draft head
        tree_nodes: list of current TreeNode objects
        config: DraftConfig
        device: torch device

    Returns:
        tree_nodes: expanded tree
        leaves: list of leaf node indices (expandable nodes)
    """
    # This function is called iteratively: we use draft head outputs
    # to expand the tree one level at a time.

    # Implementation: simple iterative best-first search
    # We do multiple rounds of: pop best node, get its children, add to tree,
    # until budget exhausted or max depth reached.

    # Priority queue: (neg_log_prob, node_idx)  -- min-heap on neg_log_prob
    heap = []
    n_prefix = prefix_ids.size(1)

    # Initialize: root's children are the first-level tokens
    # tree_nodes[0] is the root (depth 0)
    if len(tree_nodes) <= 1:
        # Get top-W tokens from the draft head at the root position
        root_probs = draft_probs[0, 0]  # (V,)
        top_probs, top_ids = torch.topk(root_probs, config.branching_width)

        for k in range(min(config.branching_width, top_ids.size(0))):
            tok = top_ids[k].item()
            log_p = torch.log(top_probs[k] + 1e-10).item()
            child = TreeNode(
                token_id=tok,
                parent_idx=0,
                depth=1,
                log_prob=log_p,
                draft_pos=len(tree_nodes),
            )
            tree_nodes[0].children.append(len(tree_nodes))
            tree_nodes.append(child)
            heapq.heappush(heap, (-child.log_prob, len(tree_nodes) - 1))

    # Expand using priority queue until budget reached
    while len(tree_nodes) < config.node_budget and heap:
        neg_lp, node_idx = heapq.heappop(heap)
        node = tree_nodes[node_idx]

        if node.depth >= config.max_depth:
            continue

        # Get draft probabilities at this node's position
        if node.draft_pos >= 0 and node.draft_pos < draft_probs.size(1):
            probs = draft_probs[0, node.draft_pos]  # (V,)
        else:
            # Fallback: use uniform
            probs = torch.ones(draft_probs.size(-1), device=device) / draft_probs.size(-1)

        top_probs, top_ids = torch.topk(probs, config.branching_width)

        for k in range(min(config.branching_width, top_ids.size(0))):
            if len(tree_nodes) >= config.node_budget:
                break

            tok = top_ids[k].item()
            log_p = torch.log(top_probs[k] + 1e-10).item()
            child = TreeNode(
                token_id=tok,
                parent_idx=node_idx,
                depth=node.depth + 1,
                log_prob=node.log_prob + log_p,
                draft_pos=len(tree_nodes),
            )
            node.children.append(len(tree_nodes))
            tree_nodes.append(child)
            heapq.heappush(heap, (-child.log_prob, len(tree_nodes) - 1))

    return tree_nodes


def full_tree_construction(
    draft_head: DraftHead,
    target_model: TargetModel,
    prefix_ids: torch.Tensor,
    config: DraftConfig,
    device: torch.device,
) -> List[TreeNode]:
    """
    Perform full tree construction with iterative draft head inference.

    1. Build initial tree from draft head on prefix
    2. Iteratively expand using best-first search
    3. Each expansion round: pop best node, run draft head, get children

    Returns:
        tree_nodes: complete tree
    """
    B = prefix_ids.size(0)
    n_prefix = prefix_ids.size(1)

    # Get target model hidden states for the prefix
    target_model.eval()
    with torch.no_grad():
        target_logits, all_target_hidden = target_model(prefix_ids)
        # Select layers for fusion
        selected_hidden = [all_target_hidden[i] for i in config.fuse_layers]

    # Initialize tree with root node
    root = TreeNode(token_id=0, parent_idx=-1, depth=0, log_prob=0.0, draft_pos=0)
    tree_nodes = [root]

    # Iterative expansion
    while len(tree_nodes) < config.node_budget:
        # Find best expandable node (highest log_prob, not at max depth)
        best_idx = -1
        best_lp = float("-inf")
        for i, node in enumerate(tree_nodes):
            if node.depth < config.max_depth and node.log_prob > best_lp:
                if i > 0:  # skip root
                    best_lp = node.log_prob
                    best_idx = i

        if best_idx < 0:
            break

        parent_node = tree_nodes[best_idx]

        # Build the path from root to this node (the branch prefix)
        branch_tokens = []
        current = best_idx
        while current > 0:
            branch_tokens.append(tree_nodes[current].token_id)
            current = tree_nodes[current].parent_idx
        branch_tokens.reverse()

        # Build input to draft head: prefix + branch tokens (excluding parent, which is the last in branch)
        # Actually: prefix + tokens along the branch (the sequence the parent node was generated from)
        if branch_tokens:
            draft_input_branch = torch.tensor([branch_tokens], dtype=torch.long, device=device)
        else:
            draft_input_branch = torch.empty(0, dtype=torch.long, device=device).unsqueeze(0)

        # We need to feed the prefix + ancestor tokens to the draft head
        # The draft head takes prefix (via cross-attention to fused features)
        # and the ancestor tokens as the self-attention sequence
        full_draft_input = draft_input_branch  # ancestor tokens in order
        # Also need a dummy for root expansion
        if full_draft_input.size(1) == 0:
            # For root's children: draft head takes prefix as context
            # We use a special [BOS]-like approach: feed the prefix as the self-attention context
            full_draft_input = prefix_ids  # use prefix tokens

        # Build tree-causal mask for this sequence
        # For root expansion: prefix tokens + root
        # For deeper: prefix tokens + branch tokens
        n_draft_nodes = full_draft_input.size(1)
        T_total = n_prefix + n_draft_nodes

        # Simple causal mask for this branch (all nodes attend to prefix + ancestors)
        # Since branch tokens are sequential along one path, standard causal works
        branch_mask = torch.triu(
            torch.ones(T_total, T_total, device=device) * float("-inf"), diagonal=1
        )
        # Allow draft positions to see all prefix positions
        branch_mask[n_prefix:, :n_prefix] = 0.0

        # Run draft head
        draft_head.eval()
        with torch.no_grad():
            logits, probs = draft_head(full_draft_input, branch_mask, selected_hidden)

        # Get top-W candidates for the child
        # The last position's logits predict the next token
        last_pos_probs = probs[0, -1]  # (V,)
        top_probs, top_ids = torch.topk(last_pos_probs, config.branching_width)

        for k in range(min(config.branching_width, top_ids.size(0))):
            if len(tree_nodes) >= config.node_budget:
                break

            tok = top_ids[k].item()
            log_p = torch.log(top_probs[k] + 1e-10).item()
            child = TreeNode(
                token_id=tok,
                parent_idx=best_idx,
                depth=parent_node.depth + 1,
                log_prob=parent_node.log_prob + log_p,
                draft_pos=len(tree_nodes),
            )
            parent_node.children.append(len(tree_nodes))
            tree_nodes.append(child)

    return tree_nodes


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_tree_greedy(
    target_model: TargetModel,
    prefix_ids: torch.Tensor,
    tree_nodes: List[TreeNode],
    config: DraftConfig,
    device: torch.device,
) -> Tuple[List[int], int, int]:
    """
    Greedy verification of the draft tree against the target model.

    For each branch in the tree, feed prefix + branch tokens to the target model
    and check how many consecutive tokens match the target's greedy prediction.

    Args:
        target_model: the frozen target model
        prefix_ids: (1, T_prefix) verified prefix tokens
        tree_nodes: list of TreeNode objects
        config: DraftConfig
        device: torch device

    Returns:
        accepted_tokens: list of accepted token ids
        n_accepted: number of accepted draft tokens
        n_proposed: total draft tokens proposed
    """
    target_model.eval()
    B = prefix_ids.size(0)
    n_prefix = prefix_ids.size(1)

    # Collect all root-to-leaf paths (branches)
    branches = []  # list of (path_token_ids, [node_indices])
    leaf_indices = [i for i, n in enumerate(tree_nodes) if len(n.children) == 0 and i > 0]

    # Also include all non-root, non-leaf nodes (partial branches)
    all_endpoints = list(set(leaf_indices + [i for i in range(1, len(tree_nodes))]))

    for end_idx in all_endpoints:
        path = []
        path_indices = []
        current = end_idx
        while current > 0:
            path.append(tree_nodes[current].token_id)
            path_indices.append(current)
            current = tree_nodes[current].parent_idx
        path.reverse()
        path_indices.reverse()
        branches.append((path, path_indices))

    # Verify each branch
    best_branch = []
    best_length = 0

    with torch.no_grad():
        target_logits_prefix, _ = target_model(prefix_ids)
        # Get target model's predictions at each position
        # For verification: we need target p(y_t | x, y_{<t})

        for branch_tokens, branch_indices in branches:
            if not branch_tokens:
                continue

            # Build full sequence: prefix + branch tokens
            branch_tensor = torch.tensor([branch_tokens], dtype=torch.long, device=device)
            full_seq = torch.cat([prefix_ids, branch_tensor], dim=1)

            # Get target logits
            target_logits, _ = target_model(full_seq)

            # Check greedy match for each branch token
            accepted_count = 0
            for t in range(len(branch_tokens)):
                # Target predicts token at position n_prefix + t given prefix + tokens[:t]
                # Greedy: does target's argmax at that position match the draft token?
                pred = target_logits[0, n_prefix + t - 1].argmax().item() if t > 0 \
                    else target_logits_prefix[0, n_prefix - 1].argmax().item()

                if t == 0:
                    # First branch token: predicted from prefix
                    pred = target_logits_prefix[0, -1].argmax().item()

                if pred == branch_tokens[t]:
                    accepted_count += 1
                else:
                    break

            if accepted_count > best_length:
                best_length = accepted_count
                best_branch = branch_tokens[:accepted_count]

    return best_branch, best_length, len(tree_nodes) - 1


def speculative_decode_step(
    draft_head: DraftHead,
    target_model: TargetModel,
    prefix_ids: torch.Tensor,
    config: DraftConfig,
    device: torch.device,
) -> Tuple[List[int], float, int, int]:
    """
    One step of speculative decoding:
    1. Construct draft tree
    2. Verify against target model
    3. Return accepted tokens and metrics

    Returns:
        accepted_tokens: list of accepted token ids
        acceptance_rate: fraction of draft tokens accepted
        n_accepted: number accepted
        n_proposed: number proposed
    """
    # Build draft tree
    tree_nodes = full_tree_construction(draft_head, target_model, prefix_ids, config, device)

    # Verify
    accepted_tokens, n_accepted, n_proposed = verify_tree_greedy(
        target_model, prefix_ids, tree_nodes, config, device
    )

    acceptance_rate = n_accepted / max(n_proposed, 1)

    # Generate a correction token from target if any tokens were accepted
    if n_accepted > 0:
        with torch.no_grad():
            # Feed prefix + accepted tokens to get next prediction
            acc_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=device)
            full_seq = torch.cat([prefix_ids, acc_tensor], dim=1)
            target_logits, _ = target_model(full_seq)
            correction = target_logits[0, -1].argmax().item()
            accepted_tokens.append(correction)

    return accepted_tokens, acceptance_rate, n_accepted, n_proposed


# ---------------------------------------------------------------------------
# Distillation Loss (Forward KL)
# ---------------------------------------------------------------------------

def forward_kl_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Forward KL divergence loss: D_KL(target || draft)

    L = (1/T^2) * sum_m [ w_m * KL(p_m || q_m) ] / sum_m w_m

    Args:
        draft_logits: (B, T, V) draft head logits
        target_logits: (B, T, V) target model logits
        temperature: distillation temperature
        ignore_index: token id to ignore in loss

    Returns:
        loss: scalar
    """
    # Temperature-scaled softmax
    draft_probs = F.softmax(draft_logits / temperature, dim=-1)
    target_probs = F.softmax(target_logits / temperature, dim=-1)

    # KL(p || q) = sum p * log(p/q) = sum p * (log p - log q)
    target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)
    draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)

    kl_per_token = target_probs * (target_log_probs - draft_log_probs)
    kl_per_token = kl_per_token.sum(dim=-1)  # (B, T)

    # Average over positions and batch, with temperature correction
    loss = kl_per_token.mean() / (temperature ** 2)

    return loss
