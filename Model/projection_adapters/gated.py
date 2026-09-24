import torch
from torch import nn

from .base import ProjectionAdapter


class GatedProjectionAdapter(ProjectionAdapter):
    """P5 -- Gated Projection.

    Projection x Sigmoid(Gate)

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__(input_dim, output_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        return self.projection(image_features) * torch.sigmoid(self.gate(image_features))
