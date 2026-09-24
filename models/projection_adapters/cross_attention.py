import torch
from torch import nn

from .base import ProjectionAdapter


class CrossAttentionAdapter(ProjectionAdapter):
    """P4 -- Cross-Attention Adapter (single layer, not a Q-Former).

    A fixed bank of learnable query tokens attends to the DINOv2 vision
    patch tokens through one multi-head cross-attention layer, followed by
    an FFN and LayerNorm, then projected to the decoder's 768-dim space.
    Only one cross-attention layer is used and there is no iterative query
    refinement, unlike BLIP-2's Q-Former.

    Input:  (B, N, 1024) vision patch tokens
    Output: (B, num_queries, 768) -- the only adapter that changes the
        token count, since it replaces the patch-token sequence with a
        fixed-size learned query sequence.
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, num_queries: int = 32,
                 num_heads: int = 8, ffn_dim: int = None, dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.hidden_dim = input_dim
        ffn_dim = ffn_dim or input_dim * 4

        self.query = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
        )
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        batch_size = image_features.shape[0]
        query = self.query.expand(batch_size, -1, -1)

        attn_out, _ = self.cross_attn(query, image_features, image_features)
        x = self.attn_norm(query + attn_out)
        x = self.ffn_norm(x + self.ffn(x))
        return self.out_proj(x)
