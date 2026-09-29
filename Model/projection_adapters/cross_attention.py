import torch
from torch import nn

from .base import ProjectionAdapter


class CrossAttentionAdapter(ProjectionAdapter):
    """P4 -- Cross-Attention Adapter (single layer, not a Q-Former).

    A fixed bank of learnable query tokens attends to the vision encoder's
    patch tokens through one multi-head cross-attention layer, followed by
    an FFN and LayerNorm, then projected to the decoder's 768-dim space.
    Only one cross-attention layer is used and there is no iterative query
    refinement, unlike BLIP-2's Q-Former.

    Sized and normalized to match BLIP-2's Q-Former (Salesforce LAVIS,
    lavis/models/blip2_models/{blip2,Qformer}.py), not the raw encoder
    width, per two evidence-backed corrections
    (KANIT_projeksiyon_analizi/, P1_P3_P7_PROJECTION_HATA_ANALIZI.md):

    1. Queries live in the *decoder's* 768-dim hidden space, like
       Q-Former's `query_tokens = nn.Parameter(torch.zeros(1,
       num_query_token, hidden_size))`; keys/values are separately
       projected from the vision width (`nn.Linear(config.encoder_width,
       ...)` when is_cross_attention) rather than running the whole block
       at the raw encoder width. The previous version kept queries and
       the entire attention+FFN block at the raw encoder width
       (1280 for MobileCLIP-S2) and only projected to 768 at the very end.
    2. Both the key/value input and the queries are LayerNorm'd before
       attention -- `image_embeds = self.ln_vision(self.visual_encoder(image))`
       for the vision side, and `embeddings = self.LayerNorm(embeddings)`
       unconditionally on query embeddings in `BertEmbeddings.forward`.
       Measured on the real mobileclip_s2 checkpoint
       (corrected_design_probe.json, p4_query_ln_probe.json): with queries
       NOT LayerNorm'd, the adapter's own output tokens collapse to ~0.997
       average pairwise cosine similarity (near-total token collapse,
       already present before any training); adding query LayerNorm drops
       this to ~0.13.

    Input:  (B, N, encoder_dim) vision patch tokens
    Output: (B, num_queries, 768) -- the only adapter that changes the
        token count, since it replaces the patch-token sequence with a
        fixed-size learned query sequence.
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, num_queries: int = 32,
                 num_heads: int = 8, ffn_dim: int = None, dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.hidden_dim = output_dim
        ffn_dim = ffn_dim or output_dim * 4

        self.norm_kv = nn.LayerNorm(input_dim)

        self.query = nn.Parameter(torch.zeros(1, num_queries, output_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(output_dim)

        self.cross_attn = nn.MultiheadAttention(output_dim, num_heads, kdim=input_dim, vdim=input_dim,
                                                dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, output_dim),
        )
        self.ffn_norm = nn.LayerNorm(output_dim)

    def forward(self, image_features):
        batch_size = image_features.shape[0]
        kv = self.norm_kv(image_features)
        query = self.query_norm(self.query.expand(batch_size, -1, -1))

        attn_out, _ = self.cross_attn(query, kv, kv)
        x = self.attn_norm(query + attn_out)
        x = self.ffn_norm(x + self.ffn(x))
        return x
