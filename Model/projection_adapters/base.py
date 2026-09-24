"""Common interface every projection adapter implements.

Every adapter bridges the frozen vision encoder's patch-token features
(B, N, encoder_dim) to the 768-dim space BERTurk's cross-attention expects. New
adapters only need to subclass ProjectionAdapter and implement `forward`;
parameter/shape bookkeeping (`summary`) is shared here so
runs/<name>/adapter_summary.json always has the same schema regardless of
which adapter produced it.
"""
import torch
from torch import nn


class ProjectionAdapter(nn.Module):

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "adapter_name": type(self).__name__,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": getattr(self, "hidden_dim", None),
            "architecture": repr(self),
        }
