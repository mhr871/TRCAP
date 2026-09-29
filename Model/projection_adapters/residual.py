from torch import nn

from .base import ProjectionAdapter


class ResidualMLPAdapter(ProjectionAdapter):
    """P3 -- Residual MLP.

    x -> LayerNorm -> [MLP(encoder_dim->encoder_dim), zero-init up layer]
      -> Residual Add -> LayerNorm -> Linear(encoder_dim->768)

    Two corrections versus a naive residual MLP, both evidence-backed
    (see KANIT_projeksiyon_analizi/, P1_P3_P7_PROJECTION_HATA_ANALIZI.md):

    1. The branch is computed from a LayerNorm'd input (pre-norm), not the
       raw MobileCLIP-S2 feature -- matching AdaptFormer's own formulation
       (Chen et al. 2022, arXiv:2205.13535, Eq. 3: "x~ = ReLU(LN(x'))*W_down*W_up").
    2. The branch's last (up) layer is zero-initialized, so the module
       starts as an identity function on top of the LayerNorm'd input.
       This is the "near-identity initialization" principle both
       AdaptFormer ("the weights of the up-projection layers are
       configured with zero initialization") and Houlsby et al. 2019
       (arXiv:1902.00751: "if the initialization deviates too far from
       the identity function, the model may fail to train") describe for
       adapter-style residual branches; the previous version used a
       standard (non-zero) nn.Linear init here, so at t=0 it already
       injected non-trivial, untrained noise into the residual stream.

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim or input_dim
        self.norm_in = nn.LayerNorm(input_dim)
        up = nn.Linear(self.hidden_dim, input_dim)
        nn.init.zeros_(up.weight)
        nn.init.zeros_(up.bias)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            up,
        )
        self.norm_out = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        x = self.norm_in(image_features)
        x = x + self.mlp(x)
        x = self.norm_out(x)
        return self.out_proj(x)
