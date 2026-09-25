from .base import ProjectionAdapter
from .linear import LinearAdapter
from .mlp import MLPAdapter
from .residual import ResidualMLPAdapter
from .cross_attention import CrossAttentionAdapter
from .gated import GatedProjectionAdapter
from .film import FiLMAdapter
from .bottleneck import BottleneckAdapter

# Registry: config's `proj_type` value -> adapter class. Model/TRCaptionNet.py
# only ever looks this up (one `elif self.proj_type in ADAPTERS:` branch
# alongside its existing "mlp2" / plain-linear / transformer proj_type
# options) -- so adding a P8 adapter is a two-file change (a new module
# here + one registry entry).
ADAPTERS = {
    "linear": LinearAdapter,
    "mlp": MLPAdapter,
    "residual": ResidualMLPAdapter,
    "cross_attention": CrossAttentionAdapter,
    "gated": GatedProjectionAdapter,
    "film": FiLMAdapter,
    "bottleneck": BottleneckAdapter,
}

# Deney dokümanındaki (TRCaptionNet_Projection_Adapter_Deney_Dokumani.docx)
# P1-P7 kodlarıyla eşleşir; runs/<code>_<name>/ klasör adlandırmasında kullanılır.
ADAPTER_CODES = {
    "linear": "P1",
    "mlp": "P2",
    "residual": "P3",
    "cross_attention": "P4",
    "gated": "P5",
    "film": "P6",
    "bottleneck": "P7",
}


def build_projection_adapter(name: str, input_dim: int = 1024, output_dim: int = 768,
                              **kwargs) -> ProjectionAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"Unknown projection_adapter '{name}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](input_dim=input_dim, output_dim=output_dim, **kwargs)
