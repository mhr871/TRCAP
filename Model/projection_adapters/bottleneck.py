from torch import nn

from .base import ProjectionAdapter


class BottleneckAdapter(ProjectionAdapter):
    """P7 -- Bottleneck Adapter.

    encoder_dim -> 256 -> encoder_dim -> 768, with a residual connection around the
    down/up bottleneck (AdapterFormer-style).

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, bottleneck_dim: int = 256):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = bottleneck_dim
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        x = self.up(self.act(self.down(image_features)))
        x = image_features + x
        return self.out_proj(x)
