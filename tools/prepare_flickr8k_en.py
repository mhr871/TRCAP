"""Prepare an ENGLISH-caption sibling of Data/tasvir-et, reusing the exact
same images and the exact same train/val/test split, so a MobileCLIP hybrid
config can be trained on English captions with zero other changes.

Why: after extensive debugging, Bleu_4/CIDEr have stayed flat (~0.011-0.014)
across four very different MobileCLIP S0/S2 + proj-architecture + decoder-
init combinations, even ~40% into a 50k-iteration schedule. One
still-untested variable is the caption LANGUAGE itself (Turkish, agglutinative,
scored by an English-oriented PTBTokenizer -- see tools/check_tokenization.py).
This script produces a same-schema, same-split, same-images dataset with the
ORIGINAL ENGLISH Flickr8k captions (Karpathy split, the same standard split
TasvirEt itself is built on -- both give exactly 6000/1000/1000 train/val/test
images) so the MobileCLIP S0 + mlp2 experiment (configs/tasviret/
mobileclip_s0_stage1.yaml / _stage2.yaml) can be re-run essentially unchanged,
just pointed at English data, to isolate whether Turkish specifically is
capping the metrics.

No image download needed: Data/flickr8k/images/ (already populated by
tools/download_tasviret_images.py) uses the same standard Flickr8k filenames.
Output schema is byte-for-byte compatible with Datasets/tasviret.py's
TasvirEtTrain/TasvirEtTest (same field names as tools/prepare_tasviret.py's
output), so no dataset-class code changes are needed -- only new configs
pointing train_json_path/val_json_path/test_json_path here.
"""

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

KARPATHY_SPLIT_URL = "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_flickr8k.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_IMAGE_COUNTS = {"train": 6000, "val": 1000, "test": 1000}
EXPECTED_CAPTION_COUNTS = {"train": 30000, "val": 5000, "test": 5000}  # 5 captions/image, standard Flickr8k


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def download_split_json(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json = output_dir / "dataset_flickr8k.json"
    if raw_json.exists() and raw_json.stat().st_size > 0:
        print(f"Using existing Karpathy split file: {raw_json}")
        return raw_json
    print(f"Downloading Karpathy Flickr8k split: {KARPATHY_SPLIT_URL}")
    try:
        urllib.request.urlretrieve(KARPATHY_SPLIT_URL, raw_json)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {KARPATHY_SPLIT_URL}. Usually a temporary "
            f"DNS/network issue -- re-run, or place dataset_flickr8k.json "
            f"under {output_dir} yourself and re-run."
        ) from exc
    return raw_json


def resolve_image_path(images_root, filename):
    candidates = [filename]
    if "_" in filename:
        candidates.append(f'{filename.split("_")[0]}.jpg')
    for candidate in candidates:
        path = images_root / candidate
        if path.exists():
            return path
    return images_root / candidates[0]


def convert_split(raw_images, split):
    images = []
    annotations = []
    ann_id = 0

    for sample in raw_images:
        if sample.get("split") != split:
            continue

        image_id = int(sample["imgid"])
        filename = sample["filename"]
        images.append(
            {
                "id": image_id,
                "imgid": image_id,
                "filename": filename,
                "file_name": filename,
            }
        )

        for sentence in sample.get("sentences", []):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "imgid": image_id,
                    "filename": filename,
                    "caption": sentence["raw"],
                }
            )
            ann_id += 1

    return {
        "info": {"source": "Karpathy Flickr8k split -- original English captions"},
        "images": images,
        "annotations": annotations,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)


def check_images(images_root, datasets):
    if not images_root.exists():
        print(f"WARNING: image root does not exist yet: {images_root}")
        return []
    missing = []
    seen = {}
    for payload in datasets.values():
        for image in payload["images"]:
            seen[image["filename"]] = True
    for filename in seen:
        if not resolve_image_path(images_root, filename).exists():
            missing.append(filename)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Prepare English Flickr8k (Karpathy split) splits, reusing TasvirEt's images.")
    parser.add_argument("--output-dir", default="Data/flickr8k-en")
    parser.add_argument("--raw-json", default=None)
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    images_root = repo_path(args.images_root)

    raw_json = repo_path(args.raw_json) if args.raw_json else None
    if raw_json is None or not raw_json.exists():
        raw_json = download_split_json(output_dir)

    with open(raw_json, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    raw_images = raw["images"]
    split_counts = Counter(sample["split"] for sample in raw_images)
    print(f"Raw split counts: {dict(split_counts)}")

    datasets = {
        "train": convert_split(raw_images, "train"),
        "val": convert_split(raw_images, "val"),
        "test": convert_split(raw_images, "test"),
    }

    write_json(output_dir / "flickr8k_train.json", datasets["train"])
    write_json(output_dir / "flickr8k_val.json", datasets["val"])
    write_json(output_dir / "flickr8k_test.json", datasets["test"])

    for split, payload in datasets.items():
        n_images, n_captions = len(payload["images"]), len(payload["annotations"])
        print(f"{split}: {n_images} images, {n_captions} captions")
        expected_i, expected_c = EXPECTED_IMAGE_COUNTS[split], EXPECTED_CAPTION_COUNTS[split]
        if (n_images, n_captions) != (expected_i, expected_c):
            print(f"  [WARNING] expected {expected_i} images / {expected_c} captions for '{split}'")

    missing = check_images(images_root, datasets)
    if missing:
        print(f"Missing images under {images_root}: {len(missing)}")
        for filename in missing[:10]:
            print(f"  missing: {filename}")
        if not args.allow_missing_images:
            raise FileNotFoundError(
                f"{len(missing)} images referenced by the English split are missing under {images_root}. "
                f"Run tools/download_tasviret_images.py first (same images, already needed for TasvirEt), "
                f"or pass --allow-missing-images to skip this check."
            )
    else:
        print(f"[OK] All images referenced by the English split are present under {images_root}.")


if __name__ == "__main__":
    main()
