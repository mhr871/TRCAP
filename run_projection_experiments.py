"""Run the P1-P7 projection-adapter experiments sequentially in one process
/ Colab session. Adapted from run_encoder_experiments.py's pattern, but
single-stage: encoder AND decoder are frozen for the whole run
(freeze_decoder: true), so only the projection adapter trains and it is
the sole independent variable across P1-P7.

Every experiment is independent: it builds a fresh model from the SAME
pretrained MobileCLIP-S2 encoder + dbmdz BERTurk decoder, a fresh
optimizer/scheduler and iteration counter, trains for max_iter, scores the
best-val and final checkpoints on the test split, and writes its results
before the next experiment starts. Nothing is carried from one adapter to
another.

Usage (from the repo root):
  python run_projection_experiments.py
  python run_projection_experiments.py --save-dir /content/drive/MyDrive/TRCAP_projection_exp
  python run_projection_experiments.py --adapters linear mlp residual
"""
import argparse
import copy
import csv
import gc
import json
import random
import sys
import time
import traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy
import torch
import yaml
from torch.backends import cudnn

from Model.projection_adapters import ADAPTERS, ADAPTER_CODES

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "projection_exp"
METRICS = ["CIDEr", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"]
METRICS_NOTE = "METEOR and SPICE are not computed (both disabled for training speed); their fields are empty."

CSV_FIELDS = (
    ["adapter_id", "adapter_name", "status", "trainable_params", "total_params",
     "batch_size", "lr_proj", "max_iter", "final_iter", "best_iter", "best_val_CIDEr"]
    + [f"test_{m}" for m in METRICS]
    + ["training_time", "training_time_sec", "peak_vram_gib", "seed", "gpu",
       "checkpoint_dir", "started_at", "finished_at", "error"]
)


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def experiment_paths(save_root, adapter_name, code):
    root = save_root / f"{code}_{adapter_name}"
    return {"root": root}


def build_args(config_path, adapter_name, paths, save_dir_override):
    with open(config_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if save_dir_override:
        cfg["save_dir"] = str(save_dir_override)
    cfg["save_path"] = str(paths["root"])
    cfg["ckpt_dir"] = str(paths["root"])
    cfg["metrics_dir"] = str(paths["root"])
    cfg["ckpt_prefix"] = f"{ADAPTER_CODES[adapter_name]}_{adapter_name}"
    cfg["save_best"] = True
    cfg["checkpoint_iters"] = [cfg["max_iter"]]
    cfg["ckpt_meta"] = {"adapter": adapter_name}
    return Namespace(**cfg)


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def count_params(model):
    total = sum(p.numel() for p in model.proj.parameters())
    trainable = sum(p.numel() for p in model.proj.parameters() if p.requires_grad)
    return {"trainable_params": trainable, "total_params": total}


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    tmp.replace(path)


def update_common_results(save_root, record):
    json_path = save_root / "projection_results.json"
    results = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fp:
            results = json.load(fp)
    results[record["adapter_id"]] = record
    results = dict(sorted(results.items()))
    write_json(json_path, results)

    csv_path = save_root / "projection_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results.values():
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def base_record(adapter_name, code, args, paths):
    return {
        "adapter_id": code,
        "adapter_name": adapter_name,
        "status": "running",
        "batch_size": args.batch_size,
        "lr_proj": float(args.lr_proj),
        "max_iter": args.max_iter,
        "target_metric": args.target_metric,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None,
        "checkpoint_dir": str(paths["root"]),
        "metrics_note": METRICS_NOTE,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error": None,
    }


def run_experiment(config_path, adapter_name, save_root, overwrite):
    from trainer import Trainer
    from utils import TBLog, get_logger

    code = ADAPTER_CODES[adapter_name]
    paths = experiment_paths(save_root, adapter_name, code)
    if paths["root"].exists() and overwrite:
        import shutil
        shutil.rmtree(paths["root"])
    paths["root"].mkdir(parents=True, exist_ok=True)

    args = build_args(config_path, adapter_name, paths, save_root)
    record = base_record(adapter_name, code, args, paths)
    summary_path = paths["root"] / "summary.json"
    write_json(summary_path, record)
    update_common_results(save_root, record)
    with open(paths["root"] / "config.yaml", "w", encoding="utf-8") as fp:
        yaml.safe_dump(vars(args), fp, allow_unicode=True, sort_keys=False)

    wall_start = time.time()
    try:
        set_seed(args.seed)
        cleanup_gpu()
        logger = get_logger(f"{code}_{adapter_name}", str(paths["root"]), "INFO")
        tb_logger = TBLog(str(paths["root"]), "tensorboard", True)
        logger.info(f"===== {code} {adapter_name} =====")
        logger.info("\n" + yaml.safe_dump(vars(args), allow_unicode=True, sort_keys=False))

        trainer = Trainer(args=args, tb_logger=tb_logger, logger=logger)
        trainer()

        record.update(count_params(trainer.model))
        record.update({
            "status": "completed",
            "final_iter": trainer.it,
            "best_iter": trainer.best_it,
            "best_val_CIDEr": trainer.best_eval_val,
            "training_time_sec": trainer.train_time_sec,
            "training_time": hms(trainer.train_time_sec),
            "peak_vram_gib": trainer.peak_vram_gib,
        })

        test_result = trainer.evaluate_test(f"{code}_{adapter_name}_final")
        for metric in METRICS:
            record[f"test_{metric}"] = test_result.get(metric)

        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        if tb_logger.writer is not None:
            tb_logger.writer.close()
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(summary_path, record)
        update_common_results(save_root, record)
        cleanup_gpu()
    return record


def main():
    parser = argparse.ArgumentParser(description="Run the P1-P7 projection adapter experiments.")
    parser.add_argument("--configs-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--save-dir", default=None, help="Output root, e.g. a mounted Google Drive directory.")
    parser.add_argument("--adapters", nargs="+", default=None,
                        help="Subset of adapter names to run, e.g. linear mlp. Default: all P1-P7.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and re-run an adapter whose output directory already exists.")
    args = parser.parse_args()

    configs_dir = repo_path(args.configs_dir)
    save_root = repo_path(args.save_dir) if args.save_dir else (REPO_ROOT / "experiments" / "projection_exp")
    save_root.mkdir(parents=True, exist_ok=True)

    selected = args.adapters or list(ADAPTERS)
    unknown = [a for a in selected if a not in ADAPTERS]
    if unknown:
        parser.error(f"unknown adapters {unknown}; available: {list(ADAPTERS)}")

    records = []
    for adapter_name in selected:
        code = ADAPTER_CODES[adapter_name]
        config_path = configs_dir / f"{code}_{adapter_name}.yaml"
        if not config_path.is_file():
            raise SystemExit(f"missing config for {code}_{adapter_name}: expected {config_path}")
        print(f"\n===== {code} {adapter_name} =====")
        records.append(run_experiment(config_path, adapter_name, save_root, args.overwrite))

    print("\n===== projection adapter experiments =====")
    for r in records:
        print(f"{r['adapter_id']} {r['adapter_name']:<16} {r['status']:<10} "
              f"best_iter={r.get('best_iter')} val_CIDEr={r.get('best_val_CIDEr')} "
              f"test_CIDEr={r.get('test_CIDEr')} time={r.get('training_time')}")
    print(f"results: {save_root / 'projection_results.csv'}")
    if any(r["status"] != "completed" for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
