import csv
import json
import os
from logging import Logger

import torch
import tqdm
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup

from Datasets.dataset_utils import getTestTransforms, getTrainDataset, getTestDataset
from Datasets.tasviret import TasvirEtTrain
from Model import TRCaptionNetPP
from eval import evaluate_on_coco_caption, predict
from utils import TBLog

EPOCH_METRICS_FIELDS = ["epoch", "train_loss", "val_loss",
                        "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge", "cider"]


class Trainer:
    """Trains one TRCaptionNetPP run: frozen DINOv2 + a single projection
    adapter (config['model']['projection_adapter']) + ELECTRA decoder.

    One Trainer instance is one P1-P7 experiment. run_projection_experiments.py
    builds a fresh Trainer per adapter so nothing (weights, optimizer,
    scheduler, iteration count) carries over between adapters.
    """

    def __init__(self, args, tb_logger: TBLog = None, logger: Logger = None):

        # initialize parameters
        self.args = args
        self.experiment_root = args.save_path
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{args.gpu}")
        self.lr_decoder = float(args.lr_decoder)
        self.lr_projection = float(args.lr_projection)
        self.betas = args.betas
        self.weight_decay = args.weight_decay
        self.max_iter = args.max_iter
        self.warm_up_iter = args.warm_up_iter
        self.num_eval_iter = args.num_eval_iter
        self.target_metric = args.target_metric
        self.resume_ckpt = getattr(args, "resume_ckpt", None)
        self.train_decoder = bool(args.model.get("train_decoder", True))
        self.train_projection = bool(args.model.get("train_projection", True))
        self.last_grad_norm = None
        self.it = 0
        self.best_eval_val = -1
        self.best_it = -1

        # dataset parameters
        self.train_dataset_root = args.train_dataset_root
        self.test_dataset_root = args.test_dataset_root
        self.train_json_path = args.train_json_path
        self.val_json_path = args.val_json_path

        # set tensorboard logger
        self.tb_logger = tb_logger

        # set logger function
        self.logger_fn = logger.info if logger is not None else print
        self.logger_fn(f"USE: {self.device} for training")
        return

    def __call__(self):

        # set dataloaders
        self.train_loader, self.test_loader, self.val_loss_loader = self.getDataloaders()
        self.iters_per_epoch = max(1, len(self.train_loader.dataset) // self.batch_size)

        # initialize model
        self.model = TRCaptionNetPP(self.args.model)
        self.model = self.model.to(self.device)
        self.save_adapter_summary()
        self.log_special_token_ids()

        # initialize optimizer -- only ever sees trainable parameters: a
        # frozen component (encoder always, and decoder/projection when
        # train_decoder / train_projection is false) contributes no group
        # at all, rather than a group with lr=0.
        optimizer_grouped_parameters = self.build_param_groups()
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=self.betas)
        self.validate_optimizer_param_groups()

        # initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_up_iter, self.max_iter)
        self.logger_fn(f"scheduler: linear warmup for {self.warm_up_iter} iterations, then linear decay to 0")

        if self.resume_ckpt:
            self.load_training_checkpoint(self.resume_ckpt)

        self.init_epoch_metrics_csv()
        self.logger_fn("Train is starting...")
        self.train()
        return

    def build_param_groups(self):
        no_decay = ['bias', 'LayerNorm.weight']
        groups = []
        if self.train_projection:
            groups += [
                {'name': 'projection_decay',
                 'params': [p for n, p in self.model.proj.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.weight_decay, 'lr': self.lr_projection},
                {'name': 'projection_no_decay',
                 'params': [p for n, p in self.model.proj.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': self.lr_projection},
            ]
        if self.train_decoder:
            groups += [
                {'name': 'decoder_decay',
                 'params': [p for n, p in self.model.language_decoder.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.weight_decay, 'lr': self.lr_decoder},
                {'name': 'decoder_no_decay',
                 'params': [p for n, p in self.model.language_decoder.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': self.lr_decoder},
            ]
        return groups

    def train(self):
        # train
        self.model.train()
        if not self.train_decoder:
            self.model.language_decoder.eval()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        remaining_iters = max(0, self.max_iter - self.it)
        tbar = tqdm.tqdm(total=remaining_iters, colour='BLUE')
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
            tb_dict['lr/decoder'] = current_lrs['decoder_lr']
            tb_dict['lr/projection'] = current_lrs['projection_lr']
            tb_dict['train/grad_norm'] = self.last_grad_norm
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.num_eval_iter == 0:
                mean_train_loss = sum(loss_window) / len(loss_window) if loss_window else float("nan")
                mean_train_acc = sum(acc_window) / len(acc_window) if acc_window else float("nan")
                loss_window = []
                acc_window = []

                eval_dict = self.eval(self.it)
                val_loss = self.compute_val_loss()
                eval_dict['val_loss'] = val_loss
                tb_dict.update({f'eval/{k}': v for k, v in eval_dict.items()})

                if eval_dict[self.target_metric] > self.best_eval_val:
                    self.best_eval_val = eval_dict[self.target_metric]
                    self.best_it = self.it
                    self.save_model('best.pth')

                # Keep a resumable checkpoint at every validation boundary.
                self.save_model('last.pth')

                self.append_epoch_metrics_row(mean_train_loss, val_loss, eval_dict)
                self.save_metrics_summary(eval_dict, mean_train_loss, val_loss)

                self.logger_fn(f"mean train/loss over last {self.num_eval_iter} iterations: "
                               f"{mean_train_loss:.4f}, mean train/acc (teacher-forced next-token): "
                               f"{mean_train_acc:.4f}, val/loss: {val_loss:.4f}")
                self.logger_fn(f"\n {self.it} iteration, {eval_dict},"
                               f" \n BEST {self.target_metric}: {self.best_eval_val}, at {self.best_it} iters")

            if self.tb_logger is not None:
                self.tb_logger.update(tb_dict, self.it)
            del tb_dict
            start_batch.record()

        self.save_model('last.pth')
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

    def save_adapter_summary(self):
        summary = self.model.proj.summary()
        os.makedirs(self.experiment_root, exist_ok=True)
        with open(os.path.join(self.experiment_root, 'adapter_summary.json'), 'w', encoding='utf-8') as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        self.logger_fn(f"adapter summary: {summary}")
        return

    def validate_optimizer_param_groups(self):
        decoder_lrs = sorted({group['lr'] for group in self.optimizer.param_groups
                              if group.get('name', '').startswith('decoder')})
        proj_lrs = sorted({group['lr'] for group in self.optimizer.param_groups
                           if group.get('name', '').startswith('projection')})

        if self.train_decoder and decoder_lrs != [self.lr_decoder]:
            raise ValueError(f"Decoder LR mismatch: expected {self.lr_decoder}, got {decoder_lrs}")
        if not self.train_decoder and decoder_lrs:
            raise ValueError(f"Decoder should have no optimizer param groups while frozen, got {decoder_lrs}")
        if self.train_projection and proj_lrs != [self.lr_projection]:
            raise ValueError(f"Projection LR mismatch: expected {self.lr_projection}, got {proj_lrs}")
        if not self.train_projection and proj_lrs:
            raise ValueError(f"Projection should have no optimizer param groups while frozen, got {proj_lrs}")

        self.logger_fn(f"optimizer LR groups verified: "
                       f"decoder_lr={self.lr_decoder if self.train_decoder else 'frozen'}, "
                       f"projection_lr={self.lr_projection if self.train_projection else 'frozen'}")
        return

    def get_current_lrs(self):
        decoder_lrs = [group['lr'] for group in self.optimizer.param_groups
                       if group.get('name', '').startswith('decoder')]
        proj_lrs = [group['lr'] for group in self.optimizer.param_groups
                    if group.get('name', '').startswith('projection')]
        return {
            'decoder_lr': decoder_lrs[0] if decoder_lrs else 0.0,
            'projection_lr': proj_lrs[0] if proj_lrs else 0.0,
        }

    def getDataloaders(self):
        train_dataset = getTrainDataset(self.train_dataset_root, self.train_json_path, model_config=self.args.model)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  sampler=RandomSampler(data_source=train_dataset,
                                                        replacement=True,
                                                        num_samples=self.max_iter * self.batch_size),
                                  pin_memory=True, drop_last=True)

        test_dataset = getTestDataset(self.test_dataset_root, self.val_json_path, model_config=self.args.model)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 shuffle=False)

        # Same TasvirEt val split as `test_loader`, but loaded through
        # TasvirEtTrain (image, caption, id) instead of TasvirEtTest
        # (image, id) so teacher-forced val_loss can be computed alongside
        # the generation-based BLEU/METEOR/ROUGE/CIDEr metrics `test_loader`
        # is used for.
        val_loss_dataset = TasvirEtTrain(dataset_root=self.test_dataset_root, json_path=self.val_json_path,
                                         transforms=getTestTransforms(model_config=self.args.model))
        val_loss_loader = DataLoader(val_loss_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False)
        return train_loader, test_loader, val_loss_loader

    @torch.no_grad()
    def compute_val_loss(self):
        self.model.eval()
        losses = []
        for image, caption, _ in self.val_loss_loader:
            image = image.to(self.device)
            loss = self.model(image, caption, return_acc=False)
            losses.append(loss.item())
        self.model.train()
        if not self.train_decoder:
            self.model.language_decoder.eval()
        return sum(losses) / len(losses) if losses else float("nan")

    def eval(self, iter=-1):
        os.makedirs(self.experiment_root, exist_ok=True)
        self.model.eval()
        self.logger_fn("Start evaluating")
        val_result, eval_diagnostics = predict(self.model, self.test_loader, self.device, return_diagnostics=True)
        self.save_result(val_result, f"prediction_{iter}.json")
        result = evaluate_on_coco_caption(os.path.join(self.experiment_root, f"prediction_{iter}.json"),
                                          self.val_json_path,
                                          os.path.join(self.experiment_root, f"result_{iter}.json"))
        result['avg_caption_len'] = eval_diagnostics['avg_caption_len']
        result['eos_rate'] = eval_diagnostics['eos_rate']
        self.save_result(result, f"result_{iter}.json")
        current_lrs = self.get_current_lrs()
        self.logger_fn(
            f"eval diagnostics at {iter}: "
            f"Bleu_4={result.get('Bleu_4')}, "
            f"CIDEr={result.get('CIDEr')}, "
            f"avg_caption_len={result['avg_caption_len']:.3f}, "
            f"eos_rate={result['eos_rate']:.3f}, "
            f"decoder_lr={current_lrs['decoder_lr']}, "
            f"projection_lr={current_lrs['projection_lr']}, "
            f"grad_norm={self.last_grad_norm}"
        )
        for index, sample_caption in enumerate(eval_diagnostics['sample_captions'], start=1):
            self.logger_fn(f"sample_caption_{index}: {sample_caption}")
        self.logger_fn(result)
        self.model.train()
        if not self.train_decoder:
            self.model.language_decoder.eval()
        return result

    def init_epoch_metrics_csv(self):
        self.epoch_metrics_path = os.path.join(self.experiment_root, 'epoch_metrics.csv')
        if not os.path.exists(self.epoch_metrics_path):
            with open(self.epoch_metrics_path, 'w', encoding='utf-8', newline='') as fp:
                csv.DictWriter(fp, fieldnames=EPOCH_METRICS_FIELDS).writeheader()
        return

    def append_epoch_metrics_row(self, train_loss, val_loss, eval_dict):
        row = {
            "epoch": round(self.it / self.iters_per_epoch, 3),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu1": eval_dict.get("Bleu_1"),
            "bleu2": eval_dict.get("Bleu_2"),
            "bleu3": eval_dict.get("Bleu_3"),
            "bleu4": eval_dict.get("Bleu_4"),
            "meteor": eval_dict.get("METEOR"),
            "rouge": eval_dict.get("ROUGE_L"),
            "cider": eval_dict.get("CIDEr"),
        }
        with open(self.epoch_metrics_path, 'a', encoding='utf-8', newline='') as fp:
            csv.DictWriter(fp, fieldnames=EPOCH_METRICS_FIELDS).writerow(row)
        return

    def save_metrics_summary(self, eval_dict, train_loss, val_loss):
        summary = {
            "iteration": self.it,
            "epoch": round(self.it / self.iters_per_epoch, 3),
            "train_loss": train_loss,
            "target_metric": self.target_metric,
            "latest": eval_dict,
            "best": {
                "iteration": self.best_it,
                self.target_metric: self.best_eval_val,
            },
        }
        with open(os.path.join(self.experiment_root, 'metrics.json'), 'w', encoding='utf-8') as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        return

    def save_model(self, model_name: str):
        # Defensive: Google Drive's FUSE mount can drop/desync a directory
        # mid-run, so recreate it here (no-op if it already exists) rather
        # than losing a whole Colab run to a transient FileNotFoundError.
        os.makedirs(self.experiment_root, exist_ok=True)
        save_filename = os.path.join(self.experiment_root, model_name)
        temp_filename = save_filename + '.tmp'
        self.model.eval()
        save_obj = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'it': self.it,
            'best_eval_val': self.best_eval_val,
            'best_it': self.best_it,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        }
        torch.save(save_obj, temp_filename)
        os.replace(temp_filename, save_filename)
        self.model.train()
        if not self.train_decoder:
            self.model.language_decoder.eval()
        self.logger_fn(f"model saved: {save_filename}\n")
        return

    def load_training_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.best_eval_val = checkpoint.get('best_eval_val', -1)
        self.best_it = checkpoint.get('best_it', -1)
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state_all' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        del checkpoint
        self.logger_fn(f'training resumed from {load_path} at iteration {self.it}')
        return

    def save_result(self, result, filename):
        os.makedirs(self.experiment_root, exist_ok=True)
        result_file = os.path.join(self.experiment_root, '%s' % filename)
        json.dump(result, open(result_file, 'w'))
        return
