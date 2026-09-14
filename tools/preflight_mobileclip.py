"""Preflight check for the MobileCLIP hybrid pipeline (S0/S1/S2 + BERTurk).

Unlike tools/preflight_colab.py (which pins exact hyperparameters/SHA256 for
the frozen DINOv2 baseline reproduction), this script validates the *new*
mobileclip_* configs: runtime, TasvirEt data integrity (same checks, dataset
is unchanged), the mobileclip package/checkpoint, and an actual forward +
backward + generate() smoke test so a real training run does not fail after
several minutes of setup.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import transformers
import yaml
from PIL import Image

# (image_count, caption_count) per split. Both TasvirEt and the English
# Karpathy Flickr8k split (tools/prepare_flickr8k_en.py) share the exact
# same 6000/1000/1000 image split -- only the caption counts differ
# (TasvirEt: ~2 captions/image; standard Flickr8k: 5 captions/image).
EXPECTED_SPLITS_BY_DATASET = {
    "tasvir-et": {
        "train": (6000, 12028),
        "val": (1000, 2006),
        "test": (1000, 2003),
    },
    "flickr8k-en": {
        "train": (6000, 30000),
        "val": (1000, 5000),
        "test": (1000, 5000),
    },
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def image_path(images_root, filename):
    candidates = [filename]
    if "_" in filename:
        candidates.append(f"{filename.split('_')[0]}.jpg")
    for candidate in candidates:
        path = images_root / candidate
        if path.exists():
            return path
    return images_root / filename


def check_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Select a GPU runtime before training.")
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / (1024 ** 3)
    print(f"[OK] Python {sys.version.split()[0]}")
    print(f"[OK] PyTorch {torch.__version__}, Transformers {transformers.__version__}")
    print(f"[OK] GPU {properties.name}, VRAM {vram_gib:.2f} GiB")
    # METEOR/SPICE are skipped in eval.py (Java subprocess dependency
    # removed after it crashed training on a malformed stats line), so Java
    # is no longer required here.


def check_data(data_dir, images_root, dataset="tasvir-et"):
    json_prefix = "tasvir" if dataset == "tasvir-et" else "flickr8k"
    expected_splits = EXPECTED_SPLITS_BY_DATASET[dataset]
    split_ids = {}
    all_image_paths = []
    for split, expected_counts in expected_splits.items():
        path = data_dir / f"{json_prefix}_{split}.json"
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        counts = (len(payload["images"]), len(payload["annotations"]))
        if counts != expected_counts:
            raise ValueError(f"{split} counts are {counts}, expected {expected_counts}")
        split_ids[split] = {int(item["id"]) for item in payload["images"]}
        for item in payload["images"]:
            resolved = image_path(images_root, item["filename"])
            if not resolved.exists() or resolved.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty image: {resolved}")
            all_image_paths.append(resolved)
        print(f"[OK] {split}: {counts[0]} images, {counts[1]} captions")

    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"]:
        raise ValueError("Train/validation/test image ids overlap")
    if len(set.union(*split_ids.values())) != 8000:
        raise ValueError("Train/validation/test union does not contain exactly 8000 image ids")
    print(f"[OK] All {len(all_image_paths)} referenced image files are present and disjoint by split")


def check_mobileclip_package(model_name, checkpoint_path):
    try:
        import mobileclip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "mobileclip package not importable. Install with:\n"
            "  pip install --no-deps git+https://github.com/apple/ml-mobileclip.git"
        ) from exc
    print(f"[OK] mobileclip package importable (module: {mobileclip.__file__})")

    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing/empty checkpoint: {checkpoint_path}. Run:\n"
            f"  python tools/download_mobileclip.py --model {model_name}"
        )
    print(f"[OK] Checkpoint present: {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")


def check_init_model_ckpt(init_model_ckpt):
    """Some configs (e.g. mobileclip_s0_proj_warmstart.yaml) warm-start
    proj+decoder from checkpoints/TRCaptionNetpp_Large.pth via
    init_model_ckpt. That file is in .gitignore (checkpoints/*.pth) and
    never part of a fresh `git clone`, so a run against such a config
    crashes deep inside trainer.py's __call__ (torch.load FileNotFoundError)
    if tools/download_checkpoint.py was never run first -- catch it here
    instead, before wasting time on the rest of preflight/training."""
    if not init_model_ckpt:
        return
    path = repo_path(init_model_ckpt)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing/empty init_model_ckpt: {path}. This is a large file not "
            f"tracked in git (see .gitignore). Run:\n"
            f"  python tools/download_checkpoint.py"
        )
    print(f"[OK] init_model_ckpt present: {path} ({path.stat().st_size} bytes)")


def check_decoder_pretrained_weights(bert_config_value):
    """Verify the language decoder's HF checkpoint actually loaded its
    pretrained weights, rather than silently falling back to random init.

    This is a regression check for a real bug hit on this project: config
    once pointed `model.bert` at an Electra checkpoint, whose state-dict
    keys are prefixed "electra." while BertLMHeadModel (Model/bert/med.py)
    expects "bert." -- none of the keys matched, so the *entire* decoder
    (embeddings + all self-attention/FFN layers) loaded randomly
    initialized instead of pretrained, and nothing in the pipeline raised
    an error about it (see devam.md). Only cross-attention sublayers and
    the cls.predictions LM head are *supposed* to be missing from any
    plain BERT checkpoint (they don't exist in one; they're added fresh
    for captioning, same as BLIP/ALBEF-style decoder init). If anything
    else is missing -- embeddings, self-attention, FFN -- that means the
    checkpoint id/path is wrong or incompatible, exactly like the Electra
    case, and training would silently proceed on a randomly initialized
    decoder again.
    """
    if os.path.isfile(bert_config_value):
        print(f"[OK] Decoder uses a local BertConfig ({bert_config_value}); "
             f"no pretrained checkpoint to verify, decoder trains from scratch as intended.")
        return

    from Model.bert import BertLMHeadModel

    model, loading_info = BertLMHeadModel.from_pretrained(
        bert_config_value, is_decoder=True, add_cross_attention=True, output_loading_info=True)
    total_keys = len(list(model.state_dict().keys()))
    missing_keys = loading_info["missing_keys"]

    def is_expected_missing(key):
        return "crossattention" in key or key.startswith("cls.")

    unexpected_missing = [k for k in missing_keys if not is_expected_missing(k)]
    if unexpected_missing:
        raise RuntimeError(
            f"Decoder checkpoint '{bert_config_value}' did not load as expected: "
            f"{len(unexpected_missing)} core weight(s) (embeddings/self-attention/FFN) came back "
            f"randomly initialized instead of pretrained, e.g. {unexpected_missing[:5]}. "
            f"This is the same failure mode as the Electra/BERT prefix mismatch bug -- "
            f"check that '{bert_config_value}' is a real BertModel-compatible checkpoint."
        )
    print(f"[OK] Decoder pretrained weights loaded correctly from '{bert_config_value}': "
         f"{total_keys - len(missing_keys)}/{total_keys} weights matched the checkpoint; "
         f"the only {len(missing_keys)} newly-initialized ones are cross-attention + LM head, as expected.")
    del model


def model_smoke_test(config, images_root, test_json):
    transformers.logging.set_verbosity_error()
    from Datasets.dataset_utils import getTestTransforms
    from Model import TRCaptionNetpp

    freeze_decoder = bool(config.get("freeze_decoder", False))
    model = TRCaptionNetpp(config["model"])

    # Mirror trainer.py's __call__ exactly (model construction, then
    # init_model_ckpt applied on top with strict_init) so this smoke test
    # actually exercises a warm-start config's real loading path (e.g.
    # mobileclip_s0_proj_warmstart.yaml's strict_init: false load from
    # checkpoints/TRCaptionNetpp_Large.pth), not just the cold-constructed
    # architecture. Without this, preflight would pass on a config whose
    # real training run fails or silently loads the wrong weights.
    init_model_ckpt = config.get("init_model_ckpt")
    if init_model_ckpt:
        strict_init = bool(config.get("strict_init", True))
        checkpoint = torch.load(repo_path(init_model_ckpt), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        result = model.load_state_dict(state_dict, strict=strict_init)
        if not strict_init:
            missing_by_module = {}
            for k in result.missing_keys:
                top = k.split(".")[0]
                missing_by_module[top] = missing_by_module.get(top, 0) + 1
            unexpected_by_module = {}
            for k in result.unexpected_keys:
                top = k.split(".")[0]
                unexpected_by_module[top] = unexpected_by_module.get(top, 0) + 1
            non_vision_missing = [k for k in result.missing_keys if not k.startswith("vision_encoder.")]
            non_vision_unexpected = [k for k in result.unexpected_keys if not k.startswith("vision_encoder.")]
            if non_vision_missing or non_vision_unexpected:
                raise RuntimeError(
                    f"init_model_ckpt '{init_model_ckpt}' left non-vision_encoder keys "
                    f"unmatched -- missing: {non_vision_missing[:5]}, unexpected: {non_vision_unexpected[:5]}. "
                    f"Expected ONLY vision_encoder.* to mismatch (different architecture, loaded "
                    f"separately via model.mobileclip_ckpt); anything else mismatching means proj "
                    f"or language_decoder did not actually warm-start as intended."
                )
            print(f"[OK] init_model_ckpt '{init_model_ckpt}' loaded (strict_init=false): "
                 f"missing_keys by module={missing_by_module}, unexpected_keys by module={unexpected_by_module} "
                 f"(only vision_encoder.* mismatching is expected/correct)")
        else:
            print(f"[OK] init_model_ckpt '{init_model_ckpt}' loaded (strict_init=true, full match required)")

    device = torch.device("cuda:0")
    model = model.to(device)
    if freeze_decoder:
        for p in model.language_decoder.parameters():
            p.requires_grad_(False)
    print(f"[OK] Model constructed (mobileclip={config['model']['mobileclip']}, "
         f"proj_type={config['model'].get('proj_type')}, freeze_decoder={freeze_decoder})")

    with open(test_json, "r", encoding="utf-8") as fp:
        sample = json.load(fp)["images"][0]
    sample_path = image_path(images_root, sample["filename"])
    transform = getTestTransforms(model_config=config["model"])
    image = transform(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)

    # forward + backward: confirms gradients reach the projection layer (and,
    # in stage 2, the decoder), and that the frozen encoder/decoder truly get
    # no gradient.
    model.train()
    if freeze_decoder:
        model.language_decoder.eval()
    # Placeholder text: the smoke test only checks that gradients flow
    # through the right modules, not caption quality, so the exact wording
    # doesn't matter (and tasvir_test.json's captions live in "annotations",
    # keyed by image_id, not inline on the image entry).
    caption = ["bu bir fotograf ."]
    loss = model(image, caption)
    loss.backward()

    proj_grad_norm = sum(p.grad.norm().item() for p in model.proj.parameters() if p.grad is not None)
    if proj_grad_norm == 0.0:
        raise RuntimeError("Projection layer received zero gradient; check proj wiring.")
    print(f"[OK] Forward+backward smoke test: loss={loss.item():.4f}, proj_grad_norm={proj_grad_norm:.6f}")

    encoder_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                           for p in model.vision_encoder.parameters())
    if encoder_has_grad:
        raise RuntimeError("Vision encoder received a gradient; it must stay frozen.")
    print("[OK] Vision encoder confirmed frozen (no gradient)")

    if freeze_decoder:
        decoder_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                               for p in model.language_decoder.parameters())
        if decoder_has_grad:
            raise RuntimeError("Language decoder received a gradient while freeze_decoder=true.")
        print("[OK] Language decoder confirmed frozen (no gradient) for stage 1")
    else:
        decoder_grad_norm = sum(p.grad.norm().item() for p in model.language_decoder.parameters() if p.grad is not None)
        if decoder_grad_norm == 0.0:
            raise RuntimeError("Language decoder received zero gradient with freeze_decoder=false; check wiring.")
        print(f"[OK] Language decoder receiving gradient as expected (norm={decoder_grad_norm:.6f})")

    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        generated = model.generate(image)[0]
    print(f"[OK] End-to-end GPU generation: {generated!r}")

    # Real-batch-size smoke test: batch=1 above only proves the wiring is
    # correct, not that a full training/eval step fits in VRAM. A training
    # step (forward+backward at config["batch_size"]) and, separately, an
    # eval step (model.generate() with num_beams=3 at that same batch_size,
    # since trainer.py's eval() reuses the training batch_size for
    # test_loader) have very different memory profiles -- generate()
    # internally expands the batch by num_beams for beam search. A config
    # that only accounted for the vision encoder's memory footprint when
    # picking batch_size (while the language decoder, unfrozen in Stage 2,
    # costs the same regardless of encoder size) can pass the batch=1 test
    # above and still OOM hours into a real run. Catch that here instead.
    batch_size = int(config.get("batch_size", 1))
    if batch_size > 1:
        torch.cuda.reset_peak_memory_stats(device)
        images_batch = image.repeat(batch_size, 1, 1, 1)
        captions_batch = [caption[0]] * batch_size

        model.train()
        if freeze_decoder:
            model.language_decoder.eval()
        model.zero_grad(set_to_none=True)
        loss = model(images_batch, captions_batch)
        loss.backward()
        model.zero_grad(set_to_none=True)
        train_peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[OK] Full-batch (batch_size={batch_size}) train step: "
             f"loss={loss.item():.4f}, peak VRAM={train_peak_gib:.2f} GiB")

        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        with torch.no_grad():
            eval_preds = model.generate(images_batch)
        eval_peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[OK] Full-batch (batch_size={batch_size}) eval generate() "
             f"(num_beams=3, matches trainer.py's periodic eval): "
             f"peak VRAM={eval_peak_gib:.2f} GiB, sample={eval_preds[0]!r}")

        total_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        worst_peak_gib = max(train_peak_gib, eval_peak_gib)
        if worst_peak_gib > 0.9 * total_gib:
            print(f"[WARNING] Peak VRAM ({worst_peak_gib:.2f} GiB) is within 10% of "
                 f"total device memory ({total_gib:.2f} GiB); a real run risks OOM "
                 f"once memory fragments. Consider lowering batch_size.")
        model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description="Validate a MobileCLIP hybrid config before starting a real training run.")
    parser.add_argument("--config", required=True,
                        help="e.g. configs/tasviret/mobileclip_s0_stage1.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="Defaults to Data/tasvir-et, or Data/flickr8k-en if --dataset flickr8k-en.")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--dataset", choices=list(EXPECTED_SPLITS_BY_DATASET), default="tasvir-et",
                        help="tasvir-et (Turkish, default) or flickr8k-en (English Karpathy split, "
                             "see tools/prepare_flickr8k_en.py) -- selects expected split counts and "
                             "the tasvir_*.json vs flickr8k_*.json filename prefix.")
    parser.add_argument("--skip-model-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = "Data/tasvir-et" if args.dataset == "tasvir-et" else "Data/flickr8k-en"

    config_path = repo_path(args.config)
    data_dir = repo_path(args.data_dir)
    images_root = repo_path(args.images_root)
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    model_name = config["model"]["mobileclip"]
    checkpoint_path = repo_path(config["model"]["mobileclip_ckpt"])

    json_prefix = "tasvir" if args.dataset == "tasvir-et" else "flickr8k"

    check_runtime()
    check_data(data_dir, images_root, dataset=args.dataset)
    check_mobileclip_package(model_name, checkpoint_path)
    check_init_model_ckpt(config.get("init_model_ckpt"))
    check_decoder_pretrained_weights(config["model"]["bert"])
    if not args.skip_model_smoke_test:
        model_smoke_test(config, images_root, data_dir / f"{json_prefix}_test.json")
    print(f"PREFLIGHT PASSED: {args.config} is ready for training.")


if __name__ == "__main__":
    main()
