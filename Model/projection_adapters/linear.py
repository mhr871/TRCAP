from torch import nn

from .base import ProjectionAdapter


class LinearAdapter(ProjectionAdapter):
    """P1 -- Linear Projection.

    LayerNorm -> Linear. No activation.

    The LayerNorm is not decorative: MobileCLIPEncoder.forward()
    (Model/mobileclip/mobileclip_encoder.py) returns the raw pre-pooling
    conv_exp feature map with no built-in normalization, unlike e.g.
    DINOv2's wrapper which already returns LayerNorm'd patch tokens.
    Measured on the real mobileclip_s2 checkpoint
    (KANIT_projeksiyon_analizi/adapter_evidence.json, raw_encoder):
    per-channel std varies up to ~10.9x around the median channel and the
    per-image output scale (std) varies up to ~1.36x across images without
    this normalization -- exactly the failure mode this project's own
    working reference connector (Model/TRCaptionNet.py's MlpProj:
    LayerNorm -> Linear -> GELU -> Linear) already avoids by normalizing
    first. A fresh nn.Linear is initialized assuming roughly unit-variance
    input, which the raw encoder output is not.

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        return self.proj(self.norm(image_features))
