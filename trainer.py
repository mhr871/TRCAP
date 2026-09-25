import json
import os
import time
from logging import Logger

import torch
import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from Datasets.dataset_utils import getTrainDataset, getTestDataset
from Model import TRCaptionNetpp
from eval import evaluate_on_coco_caption, predict
from utils import TBLog
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, args, tb_logger: TBLog = None, logger: Logger = None):

        # initialize parameters
        self.args = args
        self.experiment_root = args.save_path
        self.ckpt_dir = getattr(args, "ckpt_dir", None) or args.save_path
        self.metrics_dir = getattr(args, "metrics_dir", None) or args.save_path
        self.ckpt_prefix = getattr(args, "ckpt_prefix", None) or args.save_name
        self.checkpoint_iters = set(getattr(args, "checkpoint_iters", None) or [args.max_iter])
        self.save_best = bool(getattr(args, "save_best", True))
        self.ckpt_meta = dict(getattr(args, "ckpt_meta", None) or {})
        self.init_ckpt_expected_meta = getattr(args, "init_ckpt_expected_meta", None)
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{args.gpu}")
        self.lr = float(args.lr)
        self.lr_proj = float(args.lr_proj)
        self.betas = args.betas
        self.weight_decay = float(args.weight_decay)
        self.max_iter = int(args.max_iter)
        self.warm_up_iter = int(args.warm_up_iter)
        self.scheduler_total_iter = int(getattr(args, "scheduler_total_iter", None) or self.max_iter)
        self.target_metric = args.target_metric
        self.init_model_ckpt = getattr(args, "init_model_ckpt", None)
        self.strict_init = getattr(args, "strict_init", True)
        self.freeze_decoder = bool(getattr(args, "freeze_decoder", False))
        self.last_grad_norm = None
        self.it = 0
        self.best_eval_val = -1
        self.best_it = -1
        self.best_ckpt_path = None
        self.saved_checkpoints = {}
        self.val_history = []
        self.train_time_sec = None
        self.peak_vram_gib = None

        # dataset parameters
        self.train_dataset_name = args.train_dataset_name
        self.test_dataset_name = args.test_dataset_name
        self.train_dataset_root = args.train_dataset_root
        self.test_dataset_root = args.test_dataset_root
        self.train_json_path = args.train_json_path
        self.val_json_path = args.val_json_path
        self.test_json_path = getattr(args, "test_json_path", None)

        # set tensorboard logger
        self.tb_logger = tb_logger

        # set logger function
        self.logger_fn = logger.info if logger is not None else print
        self.logger_fn(f"USE: {self.device} for training")
        return

    def __call__(self):

        # set dataloaders
        self.train_loader, self.test_loader = self.getDataloaders()

        # initialize model
        self.model = TRCaptionNetpp(self.args.model)
        if self.init_model_ckpt:
            checkpoint = torch.load(self.init_model_ckpt, map_location="cpu")
            if self.init_ckpt_expected_meta:
                # Guards against initializing from another experiment's
                # checkpoint (e.g. E2 stage 2 picking up E1's stage 1 weights).
                meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
                mismatched = {k: (meta.get(k), v) for k, v in self.init_ckpt_expected_meta.items()
                              if meta.get(k) != v}
                if mismatched:
                    raise RuntimeError(f"init_model_ckpt {self.init_model_ckpt} belongs to a different run: "
                                       f"(found, expected) = {mismatched}")
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=self.strict_init)
            self.logger_fn(f"initialized model weights from {self.init_model_ckpt}")
            del state_dict, checkpoint
        self.model = self.model.to(self.device)
        self.log_special_token_ids()

        if self.freeze_decoder:
            for p in self.model.language_decoder.parameters():
                p.requires_grad_(False)
            self.model.language_decoder.eval()
            self.logger_fn("Stage 1 warmup: language_decoder frozen, only the projection layer is trained.")

        # initialize optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'name': 'proj_decay',
             'params': [p for n, p in self.model.proj.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay, "lr": self.lr_proj},
            {'name': 'proj_no_decay',
             'params': [p for n, p in self.model.proj.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.lr_proj},

        ]
        if not self.freeze_decoder:
            optimizer_grouped_parameters = [
                {'name': 'decoder_decay',
                 'params': [p for n, p in self.model.language_decoder.named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': self.weight_decay, "lr": self.lr},
                {'name': 'decoder_no_decay',
                 'params': [p for n, p in self.model.language_decoder.named_parameters() if
                            any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.lr},
            ] + optimizer_grouped_parameters
        # self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=self.betas)
        self.validate_optimizer_param_groups()

        # initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_up_iter, self.scheduler_total_iter)
        self.logger_fn(f"scheduler: linear warmup for {self.warm_up_iter} iterations, then linear decay "
                       f"to 0 at iteration {self.scheduler_total_iter} (training stops at {self.max_iter})")

        self.logger_fn("Train is starting...")
        self.train()
        return

    def train(self):
        # train
        self.model.train()
        if self.freeze_decoder:
            self.model.language_decoder.eval()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats(self.device)
        train_start = time.time()
        start_batch.record()

        remaining_iters = max(0, self.max_iter - self.it)
        tbar = tqdm.tqdm(total=remaining_iters, colour='BLUE')
        # Training loss was only ever written to TensorBoard, never printed
        # to the console log -- meaning every debugging session so far had
        # no visibility into whether the model was actually fitting the
        # training data better over time, only into periodic validation
        # metrics. Track a running window between eval points and print its
        # mean alongside eval results so this is visible in plain console
        # logs (e.g. a Colab cell's output), not just TensorBoard.
        loss_window = []
        acc_window = []
        for image, caption, ids in self.train_loader:
            if self.it >= self.max_iter:
                break
            tbar.update(1)
            self.it += 1

            end_batch.record()
            start_run.record()

            image = image.to(self.device)
            loss, acc = self.model(image, caption, return_acc=True)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.last_grad_norm = float(grad_norm.detach().cpu().item())
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            loss_value = loss.detach().cpu().item()
            acc_value = acc.detach().cpu().item()
            tb_dict['train/loss'] = loss_value
            tb_dict['train/acc'] = acc_value
            loss_window.append(loss_value)
            acc_window.append(acc_value)
            current_lrs = self.get_current_lrs()
            tb_dict['lr'] = current_lrs['decoder_lr']
            tb_dict['lr/decoder'] = current_lrs['decoder_lr']
            tb_dict['lr/proj'] = current_lrs['proj_lr']
            tb_dict['train/grad_norm'] = self.last_grad_norm
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.args.num_eval_iter == 0:
                mean_train_loss = sum(loss_window) / len(loss_window) if loss_window else float("nan")
                mean_train_acc = sum(acc_window) / len(acc_window) if acc_window else float("nan")
                loss_window = []
                acc_window = []

                eval_dict = self.eval(self.it)
                tb_dict.update(eval_dict)
                self.val_history.append({"iter": self.it, **eval_dict})

                if self.it in self.checkpoint_iters:
                    self.saved_checkpoints[self.it] = self.save_model(f"{self.ckpt_prefix}_iter_{self.it}.pth")

                if eval_dict[self.target_metric] > self.best_eval_val:
                    self.best_eval_val = eval_dict[self.target_metric]
                    self.best_it = self.it
                    if self.save_best:
                        # Eval points coincide with milestone checkpoints, so the
                        # best one is normally already on disk; reuse it instead
                        # of writing a duplicate ~0.5 GB file.
                        self.best_ckpt_path = (self.saved_checkpoints.get(self.it)
                                               or self.save_model(f"{self.ckpt_prefix}_best.pth"))

                self.logger_fn(f"mean train/loss over last {self.args.num_eval_iter} iterations: "
                               f"{mean_train_loss:.4f}, mean train/acc (teacher-forced next-token): "
                               f"{mean_train_acc:.4f}")
                self.logger_fn(f"\n {self.it} iteration, {eval_dict},"
                               f" \n BEST {self.target_metric}: {self.best_eval_val}, at {self.best_it} iters")
                self.logger_fn(f" {self.it} iteration, {self.target_metric}:"
                               f" {eval_dict[self.target_metric]}\n")

            elif self.it in self.checkpoint_iters:
                self.saved_checkpoints[self.it] = self.save_model(f"{self.ckpt_prefix}_iter_{self.it}.pth")

            if self.tb_logger is not None:
                self.tb_logger.update(tb_dict, self.it)
            del tb_dict
            start_batch.record()

        tbar.close()
        self.train_time_sec = time.time() - train_start
        self.peak_vram_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        if self.it != self.max_iter:
            raise RuntimeError(f"training stopped at iteration {self.it}, expected {self.max_iter}")
        self.logger_fn(f"training finished at iteration {self.it} in {self.train_time_sec:.1f}s, "
                       f"peak VRAM {self.peak_vram_gib:.2f} GiB, best {self.target_metric}="
                       f"{self.best_eval_val} at {self.best_it}")
        return

    def log_special_token_ids(self):
        tokenizer = self.model.tokenizer
        decoder_config = self.model.language_decoder.config
        self.logger_fn(
            "tokenizer ids: "
            f"cls_token_id={tokenizer.cls_token_id}, "
            f"sep_token_id={tokenizer.sep_token_id}, "
            f"pad_token_id={tokenizer.pad_token_id}"
        )
        self.logger_fn(
            "decoder config ids: "
            f"bos_token_id={decoder_config.bos_token_id}, "
            f"eos_token_id={decoder_config.eos_token_id}, "
            f"pad_token_id={decoder_config.pad_token_id}"
        )
        return

    def validate_optimizer_param_groups(self):
        decoder_lrs = sorted({group['lr'] for group in self.optimizer.param_groups
                              if group.get('name', '').startswith('decoder')})
        proj_lrs = sorted({group['lr'] for group in self.optimizer.param_groups
                           if group.get('name', '').startswith('proj')})

        if not self.freeze_decoder and decoder_lrs != [self.lr]:
            raise ValueError(f"Decoder LR mismatch: expected {self.lr}, got {decoder_lrs}")
        if self.freeze_decoder and decoder_lrs:
            raise ValueError(f"Decoder should have no optimizer param groups while frozen, got {decoder_lrs}")
        if proj_lrs != [self.lr_proj]:
            raise ValueError(f"Projection LR mismatch: expected {self.lr_proj}, got {proj_lrs}")

        self.logger_fn(f"optimizer LR groups verified: "
                       f"decoder_lr={'frozen' if self.freeze_decoder else self.lr}, proj_lr={self.lr_proj}")
        return

    def get_current_lrs(self):
        decoder_lrs = [group['lr'] for group in self.optimizer.param_groups
                       if group.get('name', '').startswith('decoder')]
        proj_lrs = [group['lr'] for group in self.optimizer.param_groups
                    if group.get('name', '').startswith('proj')]
        return {
            'decoder_lr': decoder_lrs[0] if decoder_lrs else 0.0,
            'proj_lr': proj_lrs[0] if proj_lrs else 0.0,
        }

    def getDataloaders(self):

        # load train dataset
        if type(self.train_dataset_name) == str:
            assert type(self.train_dataset_root) == str
            assert type(self.train_json_path) == str

            train_dataset = getTrainDataset(self.train_dataset_name, self.train_dataset_root, self.train_json_path,
                                            model_config=self.args.model)

        elif type(self.train_dataset_name) == list:
            assert type(self.train_dataset_root) == list
            assert type(self.train_json_path) == list
            train_datasets = []

            for i in range(len(self.train_dataset_name)):
                train_dataset = getTrainDataset(self.train_dataset_name[i],
                                                self.train_dataset_root[i],
                                                self.train_json_path[i],
                                                model_config=self.args.model)
                train_datasets.append(train_dataset)

            train_dataset = ConcatDataset(train_datasets)
        else:
            raise Exception("What do u want to do!! ")

        # Dedicated seeded generators: the sample order must not depend on how
        # much of the global RNG the (experiment-specific) encoder consumed
        # while being built, so every encoder sees the same batch sequence.
        seed = getattr(self.args, "seed", None)
        sampler_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        loader_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  sampler=RandomSampler(data_source=train_dataset,
                                                        replacement=True,
                                                        num_samples=self.args.max_iter * self.args.batch_size,
                                                        generator=sampler_generator),
                                  generator=loader_generator,
                                  pin_memory=True, drop_last=True)

        # load test dataset
        test_dataset = getTestDataset(self.test_dataset_name, self.test_dataset_root,
                                      self.val_json_path, model_config=self.args.model)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        return train_loader, test_loader

    def eval(self, iter=-1):
        self.model.eval()
        self.logger_fn("Start evaluating")
        val_result, eval_diagnostics = predict(self.model, self.test_loader, self.device, return_diagnostics=True)
        self.save_result(val_result, f"val_prediction_iter_{iter}.json")
        result = evaluate_on_coco_caption(os.path.join(self.metrics_dir, f"val_prediction_iter_{iter}.json"),
                                          self.val_json_path,
                                          os.path.join(self.metrics_dir, f"val_result_iter_{iter}.json"))
        result['avg_caption_len'] = eval_diagnostics['avg_caption_len']
        result['eos_rate'] = eval_diagnostics['eos_rate']
        self.save_result(result, f"val_result_iter_{iter}.json")
        current_lrs = self.get_current_lrs()
        self.logger_fn(
            f"eval diagnostics at {iter}: "
            f"Bleu_4={result.get('Bleu_4')}, "
            f"CIDEr={result.get('CIDEr')}, "
            f"avg_caption_len={result['avg_caption_len']:.3f}, "
            f"eos_rate={result['eos_rate']:.3f}, "
            f"decoder_lr={current_lrs['decoder_lr']:.8f}, "
            f"proj_lr={current_lrs['proj_lr']:.8f}, "
            f"grad_norm={self.last_grad_norm}"
        )
        for index, sample_caption in enumerate(eval_diagnostics['sample_captions'], start=1):
            self.logger_fn(f"sample_caption_{index}: {sample_caption}")
        self.logger_fn(result)
        self.model.train()
        if self.freeze_decoder:
            self.model.language_decoder.eval()
        return result

    def evaluate_test(self, tag):
        """Score the model's current weights on the held-out test split."""
        if not self.test_json_path:
            raise ValueError("test_json_path is not set")
        test_dataset = getTestDataset(self.test_dataset_name, self.test_dataset_root,
                                      self.test_json_path, model_config=self.args.model)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        self.model.eval()
        self.logger_fn(f"Start test evaluation ({tag})")
        predictions, diagnostics = predict(self.model, test_loader, self.device, return_diagnostics=True)
        self.save_result(predictions, f"test_prediction_{tag}.json")
        result = evaluate_on_coco_caption(os.path.join(self.metrics_dir, f"test_prediction_{tag}.json"),
                                          self.test_json_path,
                                          os.path.join(self.metrics_dir, f"test_result_{tag}.json"))
        result['avg_caption_len'] = diagnostics['avg_caption_len']
        result['eos_rate'] = diagnostics['eos_rate']
        self.save_result(result, f"test_result_{tag}.json")
        self.logger_fn(f"test result ({tag}): {result}")
        return result

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        self.logger_fn(f"loaded model weights from {path}")
        return

    def save_model(self, model_name: str):
        # Model weights only (no optimizer/scheduler state): runs are never
        # resumed, and this keeps each checkpoint at ~0.5 GB on Drive.
        # makedirs is defensive: Google Drive's FUSE mount can drop/desync a
        # directory mid-run.
        os.makedirs(self.ckpt_dir, exist_ok=True)
        save_filename = os.path.join(self.ckpt_dir, model_name)
        temp_filename = save_filename + '.tmp'
        self.model.eval()
        save_obj = {
            'model': self.model.state_dict(),
            'it': self.it,
            'best_eval_val': self.best_eval_val,
            'best_it': self.best_it,
            'target_metric': self.target_metric,
            'meta': self.ckpt_meta,
        }
        torch.save(save_obj, temp_filename)
        os.replace(temp_filename, save_filename)
        self.model.train()
        if self.freeze_decoder:
            self.model.language_decoder.eval()
        self.logger_fn(f"model saved: {save_filename}\n")
        return save_filename

    def save_result(self, result, filename):
        os.makedirs(self.metrics_dir, exist_ok=True)
        result_file = os.path.join(self.metrics_dir, '%s' % filename)
        with open(result_file, 'w') as fp:
            json.dump(result, fp)
        return
