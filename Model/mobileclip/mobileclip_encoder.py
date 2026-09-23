from pathlib import Path

import torch
from torch import nn

_MODEL_NAMES = {"mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"}
# MobileCLIP2 is distributed for OpenCLIP (apple/ml-mobileclip README): the
# image tower is a timm FastVit ("fastvit_mci0"/"fastvit_mci2"), so it is
# loaded through open_clip instead of the `mobileclip` package. Its official
# pretrained config (open_clip dfndr2b tag) uses the same 256px, bilinear,
# shortest-edge resize and identity mean/std as MobileCLIP v1, so
# Datasets/dataset_utils.getMobileCLIPTransforms applies to both unchanged.
_OPENCLIP_MODEL_NAMES = {"mobileclip2_s0": "MobileCLIP2-S0", "mobileclip2_s2": "MobileCLIP2-S2"}

# All MobileCLIP variants (s0/s1/s2/b) are trained at 256x256. Unlike
# OpenAI-CLIP (Model/clip/clip.py), MobileCLIP/MobileCLIP2 (except the S3,
# S4 and L-14 variants, none of which are used here) use IDENTITY image
# normalization -- confirmed against apple/ml-mobileclip's own model_kwargs
# (image_mean=(0, 0, 0), image_std=(1, 1, 1)) -- i.e. the model expects raw
# [0, 1] pixel tensors straight out of ToTensor(), not CLIP-style mean/std
# normalized ones. An earlier version of this file used the OpenAI-CLIP
# normalization constants here, which silently fed every image through the
# frozen pretrained encoder with the wrong pixel distribution.
MOBILECLIP_IMAGE_SIZE = 256
MOBILECLIP_MEAN = (0.0, 0.0, 0.0)
MOBILECLIP_STD = (1.0, 1.0, 1.0)


class MobileCLIPEncoder(nn.Module):
    """Vision-tower-only wrapper around apple/ml-mobileclip.

    Mirrors Model/dino/dino.py's role: loads a pretrained MobileCLIP image
    encoder and exposes pre-pooling spatial features as a patch-token
    sequence (B, N, C), analogous to DINOv2's `x_norm_patchtokens`.

    MobileCLIP's own `MCi.forward()` returns the pooled (B, embed_dim)
    embedding after its GlobalPool2D head. To keep spatial information we
    call the underlying FastViT backbone's stages directly and stop right
    before that pooling head (`backbone.head`).
    """

    def __init__(self, model_name, checkpoint_path=None):
        super().__init__()
        if model_name not in _MODEL_NAMES and model_name not in _OPENCLIP_MODEL_NAMES:
            raise ValueError(f"Unknown mobileclip model_name: {model_name!r} "
                             f"(expected one of {sorted(_MODEL_NAMES | set(_OPENCLIP_MODEL_NAMES))})")
        self.model_name = model_name

        pretrained = str(checkpoint_path) if checkpoint_path else None
        if pretrained and not Path(pretrained).is_file():
            raise FileNotFoundError(
                f"mobileclip checkpoint not found: {pretrained}. "
                f"Run tools/download_mobileclip.py --model {model_name} first."
            )

        if model_name in _OPENCLIP_MODEL_NAMES:
            self.backbone = self._load_openclip_trunk(_OPENCLIP_MODEL_NAMES[model_name], pretrained)
        else:
            # Imported lazily: the third-party "mobileclip" pip package
            # (apple/ml-mobileclip, installed separately with `pip install
            # --no-deps git+https://github.com/apple/ml-mobileclip.git`) is only
            # required when a MobileCLIP encoder is actually built. Importing it
            # at module level would make every DINOv2/CLIP config (which never
            # touch this class) fail to even import Model.TRCaptionNet in an
            # environment that hasn't installed this optional dependency.
            import mobileclip

            clip_model, _, _ = mobileclip.create_model_and_transforms(
                model_name, pretrained=pretrained, reparameterize=True, device="cpu"
            )
            # Only the vision tower is needed; drop the text tower / logit scale
            # so they are not carried around (and never saved) inside this model.
            self.backbone = clip_model.image_encoder.model
            del clip_model

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return

    @staticmethod
    def _load_openclip_trunk(openclip_name, pretrained):
        import open_clip

        if openclip_name not in open_clip.list_models():
            raise RuntimeError(
                f"open_clip {open_clip.__version__} has no '{openclip_name}' config. "
                f"Upgrade with: pip install -U open_clip_torch timm"
            )
        if not pretrained:
            raise ValueError(f"{openclip_name} requires an explicit pretrained checkpoint path.")
        # load_checkpoint inside create_model is strict: a key mismatch between
        # the Apple checkpoint and the open_clip/timm architecture fails loudly.
        clip_model = open_clip.create_model(openclip_name, pretrained=pretrained, device="cpu")
        trunk = clip_model.visual.trunk
        del clip_model
        # Same inference-time reparameterization the v1 path gets through
        # mobileclip.create_model_and_transforms(reparameterize=True).
        trunk.eval()
        for module in trunk.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return trunk

    def train(self, mode: bool = True):
        # The encoder is always frozen/eval: never let a parent module's
        # .train() call re-enable dropout/stochastic-depth on it.
        return super().train(False)

    def forward(self, x):
        if self.model_name in _OPENCLIP_MODEL_NAMES:
            # timm FastVit.forward_features = stem -> stages -> final_conv, i.e.
            # the same pre-pooling point as the v1 conv_exp output below.
            feat = self.backbone.forward_features(x)  # (B, C, H, W)
        else:
            feat = self.backbone.forward_embeddings(x)
            feat = self.backbone.forward_tokens(feat)
            feat = self.backbone.conv_exp(feat)  # (B, C, H, W)
        return feat.flatten(2).transpose(1, 2)  # (B, N, C)

    def get_output_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, MOBILECLIP_IMAGE_SIZE, MOBILECLIP_IMAGE_SIZE,
                                device=next(self.parameters()).device)
            return self.forward(dummy).shape[-1]
