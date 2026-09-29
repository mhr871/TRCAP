from torch import nn

from .base import ProjectionAdapter


class BottleneckAdapter(ProjectionAdapter):
    """P7 -- Bottleneck Adapter (AdaptFormer-style).

    LayerNorm -> Down(encoder_dim->256) -> GELU -> Up(256->encoder_dim),
    scaled and added back as a residual, then Linear(encoder_dim->768).

    Two corrections versus a naive bottleneck, both evidence-backed
    (see KANIT_projeksiyon_analizi/, P1_P3_P7_PROJECTION_HATA_ANALIZI.md),
    matching AdaptFormer (Chen et al. 2022, arXiv:2205.13535), which is
    this adapter's own cited source ("AdapterFormer" in the experiment
    plan document):

    1. The up-projection is zero-initialized ("the weights of the
       up-projection layers are configured with zero initialization"),
       so the module is an identity function at t=0 -- the previous
       version used a standard (non-zero) init and measurably injected
       branch-scale noise comparable to the input's own energy
       (branch_rms/input_rms ~0.26 in adapter_evidence.json) before any
       training.
    2. The branch is scaled by `s=0.1` before being added back
       ("we choose s=0.10 as a default setting"), and the branch is
       computed from a LayerNorm'd input (same normalization fix every
       adapter here needs, since MobileCLIP-S2's raw features are not
       unit-scale).

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, bottleneck_dim: int = 256, scale: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = bottleneck_dim
        self.scale = scale
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck_dim)
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        x = self.norm(image_features)
        branch = self.up(self.act(self.down(x)))
        x = x + self.scale * branch
        return self.out_proj(x)
