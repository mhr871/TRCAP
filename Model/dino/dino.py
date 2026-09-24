import numpy
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from pathlib import Path

preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])


class DinoV2(nn.Module):
    """Frozen DINOv2 vision encoder wrapper (ViT-L/14 by default).

    Unlike Program/hibrit mimariler/hibrit_0's DinoV2, which loads
    pretrained=False because that project restores encoder weights from its
    own full TRCaptionNet++ checkpoint afterwards, this wrapper loads
    torch.hub's real DINOv2 pretrained weights directly: in the Projection
    Adapter ablation, the encoder is never covered by a training checkpoint
    of its own -- only the projection adapter and the ELECTRA decoder are
    trained -- so it must carry real pretrained weights from construction.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        try:
            self.vision_encoder = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        except TypeError as exc:
            if "unsupported operand type(s) for |" not in str(exc):
                raise
            _patch_dinov2_cache_for_python39()
            self.vision_encoder = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        self.vision_encoder = self.vision_encoder.eval()
        return

    def forward(self, x):
        return self.vision_encoder.forward_features(x)['x_norm_patchtokens']

    def get_output_dim(self):
        with torch.no_grad():
            dummy_input_image = preprocess(Image.fromarray(numpy.zeros((512, 512, 3), dtype=numpy.uint8))).to(
                next(self.parameters()).device)
            encoder_output_size = self.vision_encoder(dummy_input_image.unsqueeze(0)).shape[-1]
        return encoder_output_size


def _patch_dinov2_cache_for_python39():
    hub_dir = Path(torch.hub.get_dir())
    for dinov2_dir in hub_dir.glob("facebookresearch_dinov2*/dinov2"):
        for py_file in dinov2_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "from __future__ import annotations" in text:
                continue
            if " | None" not in text and "| None" not in text:
                continue
            py_file.write_text("from __future__ import annotations\n" + text, encoding="utf-8")
