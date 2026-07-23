import argparse
import json
import os
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


OFFICIAL_CAPTION_URL = "https://vision.cs.hacettepe.edu.tr/files/ff1082bf8f613d4a67e4c89a697288e6.zip"
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def download_and_extract_captions(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "tasviret8k_captions.zip"
    if not zip_path.exists():
        print(f"Downloading TasvirEt captions: {OFFICIAL_CAPTION_URL}")
        try:
            urllib.request.urlretrieve(OFFICIAL_CAPTION_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download TasvirEt captions from {OFFICIAL_CAPTION_URL}. "
                "This is usually a temporary DNS/network issue. Re-run this command, "
                f"or place tasviret8k_captions.json/zip under {output_dir} and run again."
            ) from exc
    else:
        print(f"Using existing TasvirEt caption archive: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    raw_json = output_dir / "tasviret8k_captions.json"
    if not raw_json.exists():
        raise FileNotFoundError(f"Expected raw JSON was not found: {raw_json}")
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
        "info": {"source": "TasvirEt Flickr8K Turkish captions"},
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
    parser = argparse.ArgumentParser(description="Prepare TasvirEt splits for TRCaptionNet++ training/evaluation.")
    parser.add_argument("--output-dir", default="Data/tasvir-et")
    parser.add_argument("--raw-json", default=None)
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    images_root = repo_path(args.images_root)

    raw_json = repo_path(args.raw_json) if args.raw_json else output_dir / "tasviret8k_captions.json"
    if not raw_json.exists():
        raw_json = download_and_extract_captions(output_dir)

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

    write_json(output_dir / "tasvir_train.json", datasets["train"])
    write_json(output_dir / "tasvir_val.json", datasets["val"])
    write_json(output_dir / "tasvir_test.json", datasets["test"])

    for split, payload in datasets.items():
        print(f"{split}: {len(payload['images'])} images, {len(payload['annotations'])} captions")

    missing = check_images(images_root, datasets)
    if missing:
        print(f"Missing images under {images_root}: {len(missing)}")
        for filename in missing[:20]:
            print(f"  - {filename}")
        if not args.allow_missing_images:
            sys.exit("Image check failed. Put Flickr8K images under --images-root or pass --allow-missing-images.")

    print("TasvirEt split files are ready.")


if __name__ == "__main__":
    main()
