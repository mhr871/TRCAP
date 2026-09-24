from torch import nn

from .base import ProjectionAdapter


class FiLMAdapter(ProjectionAdapter):
    """P6 -- FiLM Adapter.

    gamma, beta = Linear(x), Linear(x)
    x' = gamma * x + beta, then projected to 768.

    Input:  (B, N, 1024)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__(input_dim, output_dim)
        self.to_gamma = nn.Linear(input_dim, input_dim)
        self.to_beta = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        gamma = self.to_gamma(image_features)
        beta = self.to_beta(image_features)
        modulated = gamma * image_features + beta
        return self.out_proj(modulated)
