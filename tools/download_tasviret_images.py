import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download


DATASET_REPO = "atasoglu/flickr8k-turkish"
DATASET_REVISION = "12424a449271183bffac026dc2ed25b875c0efa4"
PARQUET_FILES = (
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_metadata(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    metadata = {}
    for sample in raw["images"]:
        image_id = int(sample["imgid"])
        if image_id in metadata:
            raise ValueError(f"Duplicate imgid in TasvirEt JSON: {image_id}")
        metadata[image_id] = {
            "filename": sample["filename"],
            "captions": [sentence["raw"] for sentence in sample.get("sentences", [])],
        }
    return metadata


def load_metadata_from_splits(data_dir):
    metadata = {}
    split_paths = [
        data_dir / "tasvir_train.json",
        data_dir / "tasvir_val.json",
        data_dir / "tasvir_test.json",
    ]
    missing = [path for path in split_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"TasvirEt JSON is missing: {data_dir / 'tasviret8k_captions.json'}. "
            "Run tools/prepare_tasviret.py first."
        )

    for path in split_paths:
        with open(path, "r", encoding="utf-8") as fp:
            split = json.load(fp)

        for image in split["images"]:
            image_id = int(image.get("imgid", image.get("id")))
            if image_id in metadata:
                raise ValueError(f"Duplicate imgid in TasvirEt splits: {image_id}")
            metadata[image_id] = {
                "filename": image.get("filename", image.get("file_name")),
                "captions": [],
            }

        for annotation in split["annotations"]:
            image_id = int(annotation.get("imgid", annotation.get("image_id")))
            metadata[image_id]["captions"].append(annotation["caption"])

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract Flickr8K images that correspond to the public TasvirEt JSON."
    )
    parser.add_argument("--raw-json", default="Data/tasvir-et/tasviret8k_captions.json")
    parser.add_argument("--output-dir", default="Data/flickr8k/images")
    args = parser.parse_args()

    raw_json_path = repo_path(args.raw_json)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_json_path.exists():
        metadata = load_metadata(raw_json_path)
    else:
        data_dir = raw_json_path.parent
        print(f"Raw TasvirEt JSON is missing: {raw_json_path}")
        print(f"Using prepared split JSON files under {data_dir}")
        metadata = load_metadata_from_splits(data_dir)
    extracted_ids = set()
    caption_mismatches = []

    for filename in PARQUET_FILES:
        print(f"Downloading/reading {filename}")
        parquet_path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=filename,
        )
        parquet_file = parquet.ParquetFile(parquet_path)
        columns = ["image", "imgid", "caption0", "caption1"]

        for batch in parquet_file.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                image_id = int(row["imgid"])
                if image_id in extracted_ids:
                    raise ValueError(f"Duplicate imgid in image mirror: {image_id}")
                if image_id not in metadata:
                    raise ValueError(f"Image mirror contains an unknown TasvirEt imgid: {image_id}")

                expected = metadata[image_id]
                mirror_captions = [row["caption0"], row["caption1"]]
                if mirror_captions != expected["captions"][:2]:
                    caption_mismatches.append(image_id)

                image_record = row["image"]
                image_bytes = image_record.get("bytes") if isinstance(image_record, dict) else None
                if not image_bytes:
                    raise ValueError(f"No embedded image bytes for imgid {image_id}")

                output_path = output_dir / expected["filename"]
                if not output_path.exists() or output_path.stat().st_size == 0:
                    output_path.write_bytes(image_bytes)
                extracted_ids.add(image_id)

                if len(extracted_ids) % 500 == 0:
                    print(f"Prepared {len(extracted_ids)}/{len(metadata)} images")

    missing_ids = sorted(set(metadata) - extracted_ids)
    if caption_mismatches:
        raise ValueError(
            f"Caption identity check failed for {len(caption_mismatches)} rows; "
            f"first ids: {caption_mismatches[:10]}"
        )
    if missing_ids:
        raise ValueError(f"Image mirror is missing {len(missing_ids)} ids; first ids: {missing_ids[:10]}")
    if len(extracted_ids) != 8000:
        raise ValueError(f"Expected 8000 TasvirEt images, got {len(extracted_ids)}")

    print(f"TasvirEt image extraction is complete: {len(extracted_ids)} images in {output_dir}")
    print(f"Source revision: {DATASET_REPO}@{DATASET_REVISION}")


if __name__ == "__main__":
    main()
