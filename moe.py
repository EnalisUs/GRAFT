import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageMoE(nn.Module):
    """
    Image Mixture-of-Experts conditioned on question features.

    Mechanism:
        - Gate   : Receives text_feat (question embedding) -> routes visual feature to experts.
        - Experts: Each expert receives visual_feat and learns distinct visual representations
                   (e.g., color patterns, leaf shape, disease lesions, severity levels).

    Training  : Gradients flow exclusively through selected experts -> autonomous specialization.
    Inference : Given a question, gate selects the optimal expert for that question type.

    Args:
        visual_dim  : Dimension of visual features (default: 2048).
        text_dim    : Dimension of text features (default: 2048).
        num_experts : Total number of expert networks.
        top_k       : Number of activated experts per sample (sparse routing).
        hidden      : Hidden dimension of each expert MLP.
        dropout     : Dropout probability within expert MLPs.
    """

    def __init__(
        self,
        visual_dim: int = 2048,
        text_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        hidden: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not (1 <= top_k <= num_experts):
            raise ValueError(f"top_k={top_k} must be within [1, {num_experts}]")

        print(
            f"[ImageMoE] num_experts={num_experts}, top_k={top_k}, "
            f"visual_dim={visual_dim}, text_dim={text_dim}"
        )

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # ── Gate: Question -> routing weights ─────────────────────────────────
        # Project text to visual space for scale stability
        self.gate_proj = nn.Linear(text_dim, visual_dim)
        self.gate = nn.Linear(visual_dim, num_experts)

        # ── Experts: Each expert is an MLP transforming visual features ───────
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, visual_dim),
            )
            for _ in range(num_experts)
        ])

        self.norm = nn.LayerNorm(visual_dim)

        # Cache metrics to compute auxiliary losses after forward pass
        self.last_gate_probs = None   # [B, E]
        self.last_expert_mask = None  # [B, E]

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse top-k routing.

        Args:
            visual_feat: [B, visual_dim] - Visual features from vision encoder.
            text_feat  : [B, text_dim]   - Question embeddings.
        Returns:
            [B, visual_dim] - Question-conditioned visual representations.
        """
        if visual_feat.dim() != 2 or visual_feat.size(1) != self.visual_dim:
            raise RuntimeError(
                f"visual_feat: Expected [B, {self.visual_dim}], got {tuple(visual_feat.shape)}"
            )
        if text_feat.dim() != 2 or text_feat.size(1) != self.text_dim:
            raise RuntimeError(
                f"text_feat: Expected [B, {self.text_dim}], got {tuple(text_feat.shape)}"
            )

        B = visual_feat.size(0)

        # ── Step 1: Gate routing based on question semantics ──────────────────
        gate_h = F.gelu(self.gate_proj(text_feat))   # [B, visual_dim]
        gate_logits = self.gate(gate_h)              # [B, E]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B, E] - cached for loss

        # Top-k routing
        topv, topi = torch.topk(gate_logits, k=self.top_k, dim=-1)   # [B, K]
        w = F.softmax(topv, dim=-1)                                   # [B, K]

        # ── Step 2: Sparse routing - each expert processes assigned samples ───
        out = torch.zeros_like(visual_feat)
        expert_mask = torch.zeros(B, self.num_experts, device=visual_feat.device)

        for j in range(self.top_k):
            ids = topi[:, j]               # [B] - expert index for slot j
            wj = w[:, j].unsqueeze(-1)     # [B, 1] - gating weight

            for e in range(self.num_experts):
                m = (ids == e)             # boolean mask [B]
                if m.any():
                    # Expert e extracts specialized visual features
                    out[m] += wj[m] * self.experts[e](visual_feat[m])
                    expert_mask[m, e] = 1.0

        # Residual + LayerNorm
        out = self.norm(out)

        # Cache metrics for external loss computation
        self.last_gate_probs = gate_probs
        self.last_expert_mask = expert_mask

        return visual_feat + out  # Residual connection: preserve base info + add expert features

    # ──────────────────────────────────────────────────────────────────────────
    def compute_load_balance_loss(self) -> torch.Tensor:
        """
        Auxiliary loss (Switch Transformer style).
        Ensures balanced expert routing to prevent expert collapse.
        """
        if self.last_gate_probs is None or self.last_expert_mask is None:
            return torch.tensor(0.0, device=self.gate.weight.device)

        P = self.last_gate_probs.mean(dim=0)    # [E] - average gate probability
        f = self.last_expert_mask.mean(dim=0)   # [E] - average usage fraction
        return self.num_experts * torch.sum(P * f)

    # ──────────────────────────────────────────────────────────────────────────
    def compute_expert_diversity_loss(self) -> torch.Tensor:
        """
        Diversity loss: Penalizes high cosine similarity between expert weight matrices.
        Forces distinct experts to learn orthogonal/diverse feature spaces.
        """
        # First layer weight matrix of each expert: [E, hidden, visual_dim]
        W = torch.stack([self.experts[e][0].weight for e in range(self.num_experts)])
        W_flat = W.view(self.num_experts, -1)               # [E, hidden * visual_dim]
        W_norm = F.normalize(W_flat, dim=-1)                # Normalize to unit vectors

        sim = W_norm @ W_norm.T                             # [E, E] cosine similarity matrix
        mask = 1 - torch.eye(self.num_experts, device=W.device)  # Mask out diagonal

        # Mean squared off-diagonal similarity
        diversity_loss = (sim * mask).pow(2).sum() / (self.num_experts * (self.num_experts - 1))
        return diversity_loss

    # ──────────────────────────────────────────────────────────────────────────
    def get_expert_usage(self) -> torch.Tensor:
        """
        Returns the expert usage proportion from the most recent forward pass.
        Useful for monitoring expert specialization during evaluation.

        Returns:
            [E] tensor, normalized sum = 1.0
        """
        if self.last_expert_mask is None:
            return torch.zeros(self.num_experts)
        usage = self.last_expert_mask.mean(dim=0).cpu()
        return usage / usage.sum().clamp(min=1e-8)
