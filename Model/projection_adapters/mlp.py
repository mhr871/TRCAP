from torch import nn

from .base import ProjectionAdapter


class MLPAdapter(ProjectionAdapter):
    """P2 -- Two-Layer MLP.

    Linear -> GELU -> Dropout(0.1) -> Linear

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, image_features):
        return self.net(image_features)
