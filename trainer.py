import json
import os
from logging import Logger

import torch
import tqdm
from torch.utils.data import DataLoader, ConcatDataset

from Datasets.dataset_utils import getTrainDataset, getTestDataset
from Model import TRCaptionNetpp
from eval import evaluate_on_coco_caption, predict
from utils import TBLog


class Trainer:
    """Epoch-based trainer (MC1-epoch-based branch).

    Replaces MC0's iteration-count-driven training (fixed
    RandomSampler(num_samples=max_iter*batch_size) + linear
    warmup-then-decay-to-zero schedule that never looks at validation
    results) with a standard epoch loop: one pass over the real dataset
    per epoch, one validation at the end of every epoch, an LR schedule
    that only reduces when validation stops improving
    (ReduceLROnPlateau), and early stopping. This mirrors the
    mechanism used by the TIC/AC-Lite (ShuffleNetV2+GRU) experiments
    (https://github.com/mhr871/TIC.git, `ShuffleNet` branch), which
    trained successfully (healthy, non-degrading BLEU curves) with
    exactly this kind of adaptive, self-stopping loop -- unlike MC0's
    fixed 50000-iteration schedule that kept training long after
    validation BLEU-4 had already started declining.

    Deliberately NOT copied from that reference: the numeric
    hyperparameters (decoder_lr=5e-4, grad_clip=5.0, batch_size=256,
    ...). Those were tuned for a GRU decoder trained from scratch; our
    decoder is a pretrained BERT fine-tune, which needs much smaller
    learning rates and tighter gradient clipping. Only the *mechanism*
    (epoch loop + ReduceLROnPlateau + early stopping) is adopted here;
    the actual LR/clip values keep close to what MC0 already used
    before its ad hoc lr_exp1 reduction. See the new
    mobileclip_s0_stage2_epoch.yaml config for the reasoning behind
    each chosen number.

    SCST (self-critical sequence training / RL fine-tuning) is
    intentionally NOT implemented here yet -- out of scope for this
    pass, per instruction.
    """

    def __init__(self, args, tb_logger: TBLog = None, logger: Logger = None):

        # initialize parameters
        self.args = args
        self.experiment_root = args.save_path
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{args.gpu}")
        self.lr = float(args.lr)
        self.lr_proj = float(args.lr_proj)
        self.betas = args.betas
        self.weight_decay = args.weight_decay

        # epoch-based schedule (replaces max_iter/warm_up_iter/num_eval_iter)
        self.max_epochs = int(args.epochs)
        self.warmup_epochs = float(args.warmup_epochs)
        self.early_stop_patience = int(args.early_stop_patience)
        self.scheduler_factor = float(args.scheduler_factor)
        self.scheduler_patience = int(args.scheduler_patience)
        self.scheduler_min_lr = float(args.scheduler_min_lr)

        self.target_metric = args.target_metric
        self.init_model_ckpt = getattr(args, "init_model_ckpt", None)
        self.strict_init = getattr(args, "strict_init", True)
        self.resume_ckpt = getattr(args, "resume_ckpt", None)
        self.freeze_decoder = bool(getattr(args, "freeze_decoder", False))
        self.grad_clip_norm = float(getattr(args, "grad_clip_norm", 1.0))

        self.last_grad_norm = None
        self.global_step = 0
        self.start_epoch = 1
        self.best_eval_val = -1
        self.best_epoch = -1
        self.epochs_since_improvement = 0

        # dataset parameters
        self.train_dataset_name = args.train_dataset_name
        self.test_dataset_name = args.test_dataset_name
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
        self.train_loader, self.test_loader = self.getDataloaders()
        self.steps_per_epoch = len(self.train_loader)
        self.warmup_steps = int(round(self.warmup_epochs * self.steps_per_epoch))
        self.logger_fn(f"{self.steps_per_epoch} steps/epoch, {self.max_epochs} epochs max, "
                       f"warmup over {self.warmup_steps} steps ({self.warmup_epochs} epoch)")

        # initialize model
        self.model = TRCaptionNetpp(self.args.model)
        if self.init_model_ckpt and not self.resume_ckpt:
            checkpoint = torch.load(self.init_model_ckpt, map_location="cpu")
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

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=self.betas)
        self.validate_optimizer_param_groups()
        # captured for the manual warmup ramp; ReduceLROnPlateau reads/writes
        # optimizer.param_groups[i]['lr'] directly, so no separate "base" state
        # is needed for it beyond this.
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # initialize scheduler: ONLY reduces LR when validation target_metric
        # plateaus -- unlike MC0's fixed decay-to-zero schedule, this
        # never touches LR unless the model has actually stopped improving.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=self.scheduler_factor,
            patience=self.scheduler_patience, min_lr=self.scheduler_min_lr,
        )
        self.logger_fn(f"scheduler: ReduceLROnPlateau(mode=max, factor={self.scheduler_factor}, "
                       f"patience={self.scheduler_patience} epochs, min_lr={self.scheduler_min_lr}), "
                       f"tracking val/{self.target_metric} once per epoch")

        if self.resume_ckpt:
            self.load_training_checkpoint(self.resume_ckpt)

        self.logger_fn("Train is starting...")
        self.train()
        return

    def set_warmup_lr(self):
        scale = min(1.0, (self.global_step + 1) / self.warmup_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base_lr * scale
        return

    def train(self):
        self.model.train()
        if self.freeze_decoder:
            self.model.language_decoder.eval()

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.epoch = epoch
            loss_window = []
            acc_window = []

            tbar = tqdm.tqdm(self.train_loader, colour='BLUE', desc=f"epoch {epoch}/{self.max_epochs}")
            for image, caption, ids in tbar:
                if self.global_step < self.warmup_steps:
                    self.set_warmup_lr()

                image = image.to(self.device)
                loss, acc = self.model(image, caption, return_acc=True)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.last_grad_norm = float(grad_norm.detach().cpu().item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                loss_value = loss.detach().cpu().item()
                acc_value = acc.detach().cpu().item()
                loss_window.append(loss_value)
                acc_window.append(acc_value)
                tbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'acc': f'{acc_value:.4f}',
                    'decoder_lr': f'{self.get_current_lrs()["decoder_lr"]:.2e}',
                })

                if self.tb_logger is not None:
                    current_lrs = self.get_current_lrs()
                    self.tb_logger.update({
                        'train/loss': loss_value,
                        'train/acc': acc_value,
                        'lr/decoder': current_lrs['decoder_lr'],
                        'lr/proj': current_lrs['proj_lr'],
                        'train/grad_norm': self.last_grad_norm,
                    }, self.global_step)

            mean_train_loss = sum(loss_window) / len(loss_window) if loss_window else float("nan")
            mean_train_acc = sum(acc_window) / len(acc_window) if acc_window else float("nan")
            self.logger_fn(f"epoch {epoch}: mean train/loss={mean_train_loss:.4f}, "
                           f"mean train/acc (teacher-forced next-token)={mean_train_acc:.4f}")

            # ---- end-of-epoch validation ----
            eval_dict = self.eval(epoch)
            val_metric = eval_dict[self.target_metric]

            # scheduler only sees the post-warmup epochs meaningfully, but
            # calling it every epoch (including during warmup) is harmless:
            # it only ever *reduces* LR relative to whatever is currently in
            # the param groups, and warmup already pushed those up to
            # base_lr by the time warmup_steps is reached.
            self.scheduler.step(val_metric)

            is_best = val_metric > self.best_eval_val
            if is_best:
                self.best_eval_val = val_metric
                self.best_epoch = epoch
                self.epochs_since_improvement = 0
                self.save_model('model_best.pth')
            else:
                self.epochs_since_improvement += 1

            self.save_model('model_last.pth')

            self.logger_fn(f"epoch {epoch}/{self.max_epochs}: {eval_dict}, "
                           f"BEST {self.target_metric}={self.best_eval_val} at epoch {self.best_epoch}, "
                           f"epochs_since_improvement={self.epochs_since_improvement}/{self.early_stop_patience}")

            if self.tb_logger is not None:
                tb_eval = {f'eval/{k}': v for k, v in eval_dict.items() if isinstance(v, (int, float))}
                self.tb_logger.update(tb_eval, self.global_step)

            if self.epochs_since_improvement >= self.early_stop_patience:
                self.logger_fn(f"EARLY STOPPING: {self.target_metric} did not improve for "
                               f"{self.early_stop_patience} epochs. Stopped at epoch {epoch} "
                               f"(best was epoch {self.best_epoch}, {self.target_metric}={self.best_eval_val}).")
                break

        self.save_model('model_last.pth')
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

        # Epoch-based: a plain shuffled pass over the real dataset once per
        # epoch (drop_last so every epoch has the same step count for the
        # warmup/scheduler bookkeeping above). MC0 used
        # RandomSampler(replacement=True, num_samples=max_iter*batch_size)
        # instead, which has no notion of "one epoch" at all -- it just
        # draws max_iter*batch_size random samples with replacement from
        # the whole run's start, which is what made the training horizon a
        # fixed iteration count instead of a real, countable number of
        # dataset passes.
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True,
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

    def eval(self, epoch=-1):
        os.makedirs(self.experiment_root, exist_ok=True)
        self.model.eval()
        self.logger_fn("Start evaluating")
        val_result, eval_diagnostics = predict(self.model, self.test_loader, self.device, return_diagnostics=True)
        self.save_result(val_result, f"prediction_epoch{epoch}.json")
        result = evaluate_on_coco_caption(os.path.join(self.experiment_root, f"prediction_epoch{epoch}.json"),
                                          self.val_json_path,
                                          os.path.join(self.experiment_root, f"result_epoch{epoch}.json"))
        result['avg_caption_len'] = eval_diagnostics['avg_caption_len']
        result['eos_rate'] = eval_diagnostics['eos_rate']
        self.save_result(result, f"result_epoch{epoch}.json")
        current_lrs = self.get_current_lrs()
        self.logger_fn(
            f"eval diagnostics at epoch {epoch}: "
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

    def save_model(self, model_name: str):
        # Defensive: Google Drive's FUSE mount can drop/desync a directory
        # mid-run. Recreating it here is a no-op when it already exists but
        # prevents a FileNotFoundError from losing a whole run.
        os.makedirs(self.experiment_root, exist_ok=True)
        save_filename = os.path.join(self.experiment_root, model_name)
        temp_filename = save_filename + '.tmp'
        self.model.eval()
        save_obj = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_eval_val': self.best_eval_val,
            'best_epoch': self.best_epoch,
            'epochs_since_improvement': self.epochs_since_improvement,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        }
        torch.save(save_obj, temp_filename)
        os.replace(temp_filename, save_filename)
        self.model.train()
        if self.freeze_decoder:
            self.model.language_decoder.eval()
        self.logger_fn(f"model saved: {save_filename}\n")
        return

    def load_training_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_eval_val = checkpoint.get('best_eval_val', -1)
        self.best_epoch = checkpoint.get('best_epoch', -1)
        self.epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state_all' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        del checkpoint
        self.logger_fn(f'training resumed from {load_path}, starting at epoch {self.start_epoch}')
        return

    def save_result(self, result, filename):
        os.makedirs(self.experiment_root, exist_ok=True)
        result_file = os.path.join(self.experiment_root, '%s' % filename)
        json.dump(result, open(result_file, 'w'))
        return
