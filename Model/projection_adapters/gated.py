import torch
from torch import nn

from .base import ProjectionAdapter


class GatedProjectionAdapter(ProjectionAdapter):
    """P5 -- Gated Projection (GLU-family connector).

    LayerNorm -> Projection(x) x Sigmoid(Gate(x))

    This is a Gated Linear Unit (Dauphin et al. 2017, "Language Modeling
    with Gated Convolutional Networks", arXiv:1612.08083: GLU(a,b) = a (x)
    sigma(b)), the same family as GEGLU/SwiGLU (Shazeer 2020,
    arXiv:2002.05202). The experiment plan document's original "PaLI /
    CoCa" citation for this adapter does not hold up against those papers'
    primary sources: PaLI (arXiv:2209.06794) uses no projection or gate at
    all ("No pooling is applied to the output of the Vision Transformer"
    before cross-attention), and CoCa's (arXiv:2205.01917) attentional
    pooler is a learnable-query multi-head attention layer, i.e. the P4
    family, not a sigmoid gate.

    The LayerNorm is the same fix as every other adapter here needs
    (see P1_P3_P7_PROJECTION_HATA_ANALIZI.md): MobileCLIP-S2's raw,
    un-normalized feature scale is a poor match for a freshly initialized
    Linear layer.

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        x = self.norm(image_features)
        return self.projection(x) * torch.sigmoid(self.gate(x))
