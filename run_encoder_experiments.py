"""Run the E1-E4 encoder experiments sequentially in one process / Colab session.

Every experiment is independent: it builds a fresh model from its own
pretrained encoder + dbmdz BERTurk, fresh optimizer/scheduler and iteration
counter, runs stage 1 (MLP2 warmup) and stage 2 (main training), scores the
best-val and final checkpoints on the test split, and writes its results
before the next experiment starts. Nothing (weights, optimizer, scheduler,
iteration state, checkpoints) is carried from one experiment to another.

Usage (from the repo root):
  python run_encoder_experiments.py --preflight
  python run_encoder_experiments.py --save-dir /content/drive/MyDrive/TRCAP_encoder_exp
  python run_encoder_experiments.py --experiments E3 E4 --save-dir ...
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import random
import shutil
import subprocess
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

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = "configs/encoder_exp/encoder_experiments.yaml"
REQUIRED_EXPERIMENT_KEYS = {"id", "name", "display_name", "encoder", "pretrained_ckpt", "pretrained_source"}
METRICS = ["CIDEr", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"]
METRICS_NOTE = "METEOR and SPICE are not computed (skipped by design); their fields are empty."

CSV_FIELDS = (
    ["experiment_id", "encoder", "encoder_key", "status",
     "pretrained_weight", "pretrained_source", "pretrained_sha256",
     "encoder_params", "projection_params", "decoder_params", "total_params", "trainable_params",
     "frozen_components", "trainable_components", "input_size",
     "batch_size", "lr_decoder", "lr_proj", "optimizer", "scheduler",
     "stage1_batch_size", "stage1_lr_proj", "stage1_max_iter",
     "max_iter", "final_iter", "best_iter", "best_val_CIDEr"]
    + [f"test_best_{m}" for m in METRICS]
    + [f"test_final_{m}" for m in METRICS]
    + ["training_time", "training_time_sec", "stage1_time_sec", "stage2_time_sec", "experiment_wall_time_sec",
       "peak_vram_gib", "seed", "gpu", "git_commit",
       "checkpoint_dir", "best_checkpoint", "final_checkpoint", "stage1_checkpoint",
       "log_dir", "metrics_dir", "started_at", "finished_at", "error"]
)


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_plan(path):
    with open(repo_path(path), "r", encoding="utf-8") as fp:
        plan = yaml.safe_load(fp)

    ids = [exp["id"] for exp in plan["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate experiment ids: {ids}")
    for exp in plan["experiments"]:
        keys = set(exp)
        if keys != REQUIRED_EXPERIMENT_KEYS:
            # Only the encoder may differ between experiments; any extra key
            # would be a second, uncontrolled experimental variable.
            raise ValueError(f"experiment {exp.get('id')} must define exactly {sorted(REQUIRED_EXPERIMENT_KEYS)}, "
                             f"got {sorted(keys)}")

    for stage in ("stage1", "stage2"):
        st = plan[stage]
        iters = st["checkpoint_iters"]
        if st["max_iter"] not in iters or any(i > st["max_iter"] for i in iters):
            raise ValueError(f"{stage}: checkpoint_iters {iters} must include and not exceed max_iter {st['max_iter']}")
    return plan


def experiment_prefix(exp):
    return f"{exp['id']}_{exp['name']}"


def experiment_paths(save_root, exp):
    root = save_root / experiment_prefix(exp)
    return {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "metrics": root / "metrics",
        "config": root / "config",
    }


def build_stage_args(plan, exp, stage, paths, init_ckpt=None):
    prefix = experiment_prefix(exp)
    data = {k: str(repo_path(v)) if k.endswith(("_root", "_path")) else v for k, v in plan["data"].items()}
    model = copy.deepcopy(plan["model"])
    model["mobileclip"] = exp["encoder"]
    model["mobileclip_ckpt"] = str(repo_path(exp["pretrained_ckpt"]))
    model["init_seed"] = plan["seed"]

    cfg = {
        "seed": plan["seed"],
        "gpu": plan["gpu"],
        "num_workers": plan["num_workers"],
        **data,
        **copy.deepcopy(plan[stage]),
        "model": model,
        "save_dir": str(paths["root"]),
        "save_name": f"{prefix}_{stage}",
        "save_path": str(paths["root"]),
        "ckpt_dir": str(paths["checkpoints"]),
        "metrics_dir": str(paths["metrics"] if stage == "stage2" else paths["metrics"] / "stage1"),
        "ckpt_prefix": prefix if stage == "stage2" else f"{prefix}_stage1",
        "save_best": stage == "stage2",
        "ckpt_meta": {"experiment_id": exp["id"], "encoder": exp["encoder"], "stage": stage},
        "init_model_ckpt": init_ckpt,
        "init_ckpt_expected_meta": ({"experiment_id": exp["id"], "encoder": exp["encoder"], "stage": "stage1"}
                                    if init_ckpt else None),
    }
    return Namespace(**cfg)


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_params(model):
    def total(module):
        return sum(p.numel() for p in module.parameters())
    return {
        "encoder_params": total(model.vision_encoder),
        "projection_params": total(model.proj),
        "decoder_params": total(model.language_decoder),
        "total_params": total(model),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    tmp.replace(path)


def update_common_results(save_root, record):
    json_path = save_root / "encoder_results.json"
    results = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fp:
            results = json.load(fp)
    results[record["experiment_id"]] = record
    results = dict(sorted(results.items()))
    write_json(json_path, results)

    csv_path = save_root / "encoder_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results.values():
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def run_training_stage(stage, args, paths, after_train=None):
    """Build a fresh Trainer (model/optimizer/scheduler/iteration counter),
    train it, and return plain results. The Trainer only lives inside this
    function, so its GPU memory is released once it returns."""
    from trainer import Trainer
    from utils import TBLog, get_logger

    set_seed(args.seed)
    logger = get_logger(args.save_name, str(paths["logs"]), "INFO", filename=f"{stage}_log.txt")
    tb_logger = TBLog(str(paths["logs"]), f"tensorboard_{stage}", True)
    try:
        logger.info(f"===== {args.save_name} =====")
        logger.info("\n" + yaml.safe_dump(vars(args), allow_unicode=True, sort_keys=False))
        trainer = Trainer(args=args, tb_logger=tb_logger, logger=logger)
        trainer()
        result = {
            "final_iter": trainer.it,
            "best_iter": trainer.best_it,
            "best_val": trainer.best_eval_val,
            "best_ckpt": trainer.best_ckpt_path,
            "checkpoints": dict(trainer.saved_checkpoints),
            "val_history": trainer.val_history,
            "train_time_sec": trainer.train_time_sec,
            "peak_vram_gib": trainer.peak_vram_gib,
            "params": count_params(trainer.model),
        }
        if after_train is not None:
            result.update(after_train(trainer))
        return result
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        tb_logger.writer.close()


def evaluate_best_and_final_on_test(trainer):
    final_it = trainer.it
    test_final = trainer.evaluate_test(f"final_iter_{final_it}")
    if trainer.best_it == final_it:
        test_best = test_final
        trainer.save_result(test_best, f"test_result_best_iter_{final_it}.json")
    else:
        trainer.load_weights(trainer.best_ckpt_path)
        test_best = trainer.evaluate_test(f"best_iter_{trainer.best_it}")
    return {"test_final": test_final, "test_best": test_best}


def base_record(plan, exp, paths, commit):
    s1, s2 = plan["stage1"], plan["stage2"]
    return {
        "experiment_id": exp["id"],
        "encoder": exp["display_name"],
        "encoder_key": exp["encoder"],
        "status": "running",
        "pretrained_weight": str(repo_path(exp["pretrained_ckpt"])),
        "pretrained_source": exp["pretrained_source"],
        "pretrained_sha256": sha256_of(repo_path(exp["pretrained_ckpt"])),
        "frozen_components": "vision encoder (both stages); BERTurk decoder (stage 1 only)",
        "trainable_components": "stage 1: MLP2 projection; stage 2: MLP2 projection + BERTurk decoder",
        "input_size": plan["model"]["image_size"],
        "batch_size": s2["batch_size"],
        "lr_decoder": float(s2["lr"]),
        "lr_proj": float(s2["lr_proj"]),
        "optimizer": f"AdamW(betas={s2['betas']}, weight_decay={float(s2['weight_decay'])}), grad clip 1.0",
        "scheduler": (f"linear warmup {s2['warm_up_iter']} it, linear decay to 0 at "
                      f"{s2['scheduler_total_iter']} it"),
        "stage1_batch_size": s1["batch_size"],
        "stage1_lr_proj": float(s1["lr_proj"]),
        "stage1_max_iter": s1["max_iter"],
        "max_iter": s2["max_iter"],
        "target_metric": s2["target_metric"],
        "seed": plan["seed"],
        "gpu": torch.cuda.get_device_name(plan["gpu"]) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "checkpoint_dir": str(paths["checkpoints"]),
        "log_dir": str(paths["logs"]),
        "metrics_dir": str(paths["metrics"]),
        "metrics_note": METRICS_NOTE,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error": None,
    }


def run_experiment(plan, exp, save_root, commit, overwrite):
    paths = experiment_paths(save_root, exp)
    if paths["root"].exists() and overwrite:
        print(f"[{exp['id']}] --overwrite: removing previous run at {paths['root']}")
        shutil.rmtree(paths["root"])
    for key in ("checkpoints", "logs", "metrics", "config"):
        paths[key].mkdir(parents=True, exist_ok=True)

    record = base_record(plan, exp, paths, commit)
    summary_path = paths["metrics"] / "summary.json"
    write_json(summary_path, record)
    update_common_results(save_root, record)
    write_yaml(paths["config"] / "experiment.yaml",
               {"experiment": exp, **{k: v for k, v in plan.items() if k != "experiments"}})

    wall_start = time.time()
    try:
        cleanup_gpu()
        args1 = build_stage_args(plan, exp, "stage1", paths)
        write_yaml(paths["config"] / f"{args1.save_name}.yaml", vars(args1))
        print(f"\n[{exp['id']}] stage 1 ({exp['display_name']}, from {args1.model['mobileclip_ckpt']})")
        stage1 = run_training_stage("stage1", args1, paths)
        cleanup_gpu()
        stage1_ckpt = stage1["checkpoints"][args1.max_iter]
        record["stage1_checkpoint"] = stage1_ckpt
        record["stage1_time_sec"] = stage1["train_time_sec"]
        record["stage1_val_history"] = stage1["val_history"]

        args2 = build_stage_args(plan, exp, "stage2", paths, init_ckpt=stage1_ckpt)
        write_yaml(paths["config"] / f"{args2.save_name}.yaml", vars(args2))
        print(f"\n[{exp['id']}] stage 2 ({exp['display_name']}, init from {stage1_ckpt})")
        stage2 = run_training_stage("stage2", args2, paths, after_train=evaluate_best_and_final_on_test)
        cleanup_gpu()

        record.update(stage2["params"])
        training_time = record["stage1_time_sec"] + stage2["train_time_sec"]
        record.update({
            "status": "completed",
            "final_iter": stage2["final_iter"],
            "best_iter": stage2["best_iter"],
            "best_val_CIDEr": stage2["best_val"],
            "best_checkpoint": stage2["best_ckpt"],
            "final_checkpoint": stage2["checkpoints"][stage2["final_iter"]],
            "milestone_checkpoints": {str(k): v for k, v in sorted(stage2["checkpoints"].items())},
            "val_history": stage2["val_history"],
            "stage2_time_sec": stage2["train_time_sec"],
            "training_time_sec": training_time,
            "training_time": hms(training_time),
            "peak_vram_gib": max(stage1["peak_vram_gib"], stage2["peak_vram_gib"]),
        })
        for tag in ("best", "final"):
            for metric in METRICS:
                record[f"test_{tag}_{metric}"] = stage2[f"test_{tag}"].get(metric)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        record["experiment_wall_time_sec"] = time.time() - wall_start
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(summary_path, record)
        update_common_results(save_root, record)
        cleanup_gpu()
    return record


def preflight(plan, experiments):
    from Model.mobileclip import MobileCLIPEncoder
    from tools import preflight_mobileclip as pf

    data_dir = repo_path(plan["data"]["val_json_path"]).parent
    images_root = repo_path(plan["data"]["train_dataset_root"])
    test_json = repo_path(plan["data"]["test_json_path"])
    pf.check_runtime()
    pf.check_data(data_dir, images_root)
    pf.check_decoder_pretrained_weights(plan["model"]["bert"])

    for exp in experiments:
        print(f"\n===== preflight {exp['id']} {exp['display_name']} =====")
        ckpt = repo_path(exp["pretrained_ckpt"])
        pf.check_mobileclip_package(exp["encoder"], ckpt)
        encoder = MobileCLIPEncoder(exp["encoder"], checkpoint_path=ckpt)
        print(f"[OK] {exp['encoder']}: output dim {encoder.get_output_dim()}, "
              f"{sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M params, sha256 {sha256_of(ckpt)[:12]}")
        del encoder
        model_cfg = {**plan["model"], "mobileclip": exp["encoder"], "mobileclip_ckpt": str(ckpt)}
        for stage in ("stage1", "stage2"):
            print(f"--- {stage} smoke test ---")
            pf.model_smoke_test({"model": model_cfg,
                                 "freeze_decoder": plan[stage]["freeze_decoder"],
                                 "batch_size": plan[stage]["batch_size"]},
                                images_root, test_json)
            cleanup_gpu()
    print("\nPREFLIGHT PASSED for " + ", ".join(exp["id"] for exp in experiments))


def main():
    parser = argparse.ArgumentParser(description="Independent E1-E4 encoder experiments.")
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--save-dir", default=None, help="Output root, e.g. a mounted Google Drive directory.")
    parser.add_argument("--experiments", nargs="+", default=None, help="Subset of experiment ids, e.g. E1 E3.")
    parser.add_argument("--preflight", action="store_true", help="Only run environment/model checks, no training.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and re-run an experiment whose output directory already has checkpoints.")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    by_id = {exp["id"]: exp for exp in plan["experiments"]}
    selected_ids = args.experiments or list(by_id)
    unknown = [i for i in selected_ids if i not in by_id]
    if unknown:
        parser.error(f"unknown experiment ids {unknown}; available: {list(by_id)}")
    experiments = [by_id[i] for i in selected_ids]

    if args.preflight:
        preflight(plan, experiments)
        return

    save_root = Path(args.save_dir) if args.save_dir else repo_path(plan["save_dir"])
    save_root.mkdir(parents=True, exist_ok=True)

    # Refuse up front, before any training, rather than failing after hours.
    missing = [str(repo_path(exp["pretrained_ckpt"])) for exp in experiments
               if not repo_path(exp["pretrained_ckpt"]).is_file()]
    if missing:
        raise SystemExit("pretrained encoder weights missing (run tools/download_mobileclip.py):\n  "
                         + "\n  ".join(missing))
    conflicts =[str(experiment_paths(save_root, exp)["checkpoints"]) for exp in experiments
                 if any(experiment_paths(save_root, exp)["checkpoints"].glob("*.pth"))]
    if conflicts and not args.overwrite:
        raise SystemExit("checkpoints already exist (pass --overwrite to delete and re-run):\n  "
                         + "\n  ".join(conflicts))

    commit = git_commit()
    records = []
    for exp in experiments:
        records.append(run_experiment(plan, exp, save_root, commit, args.overwrite))

    print("\n===== encoder experiments =====")
    for r in records:
        print(f"{r['experiment_id']} {r['encoder']:<15} {r['status']:<10} "
              f"best_iter={r.get('best_iter')} val_CIDEr={r.get('best_val_CIDEr')} "
              f"test_best_CIDEr={r.get('test_best_CIDEr')} test_final_CIDEr={r.get('test_final_CIDEr')} "
              f"time={r.get('training_time')}")
    print(f"results: {save_root / 'encoder_results.csv'}")
    if any(r["status"] != "completed" for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
