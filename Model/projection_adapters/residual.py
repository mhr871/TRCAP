from torch import nn

from .base import ProjectionAdapter


class ResidualMLPAdapter(ProjectionAdapter):
    """P3 -- Residual MLP.

    x -> MLP(encoder_dim->encoder_dim) -> Residual Add -> LayerNorm -> Linear(encoder_dim->768)

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim or input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        x = image_features + self.mlp(image_features)
        x = self.norm(x)
        return self.out_proj(x)
