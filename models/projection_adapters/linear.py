from torch import nn

from .base import ProjectionAdapter


class LinearAdapter(ProjectionAdapter):
    """P1 -- Linear Projection.

    Single Linear layer, no activation, no normalization. Baseline adapter.

    Input:  (B, N, 1024)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__(input_dim, output_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        return self.proj(image_features)
