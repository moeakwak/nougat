"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from collections import defaultdict
import json
import math
from multiprocessing import Pool
import random
from pathlib import Path

import numpy as np
import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from nougat import NougatConfig, NougatModel
from nougat.metrics import compute_metrics, get_metrics, get_step_results

from sconf import Config

class NougatModelPLModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []  # list of { metric: {...}, result: { gt, pred }}
        self.test_saved_outputs_count = 0
        self.config = config
        assert config.get("test_save_every_steps"), "test_save_every_steps must be set"
        if self.config.get("model_path", False):
            self.model = NougatModel.from_pretrained(
                self.config.model_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                window_size=self.config.window_size,
                encoder_layer=self.config.encoder_layer,
                decoder_layer=self.config.decoder_layer,
                patch_size=self.config.patch_size,
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                hidden_dimension=self.config.hidden_dimension,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = NougatModel(
                config=NougatConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    window_size=self.config.window_size,
                    encoder_layer=self.config.encoder_layer,
                    decoder_layer=self.config.decoder_layer,
                    tokenizer_file=self.config.tokenizer,
                    patch_size=self.config.patch_size,
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    hidden_dimension=self.config.hidden_dimension,
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, attention_masks = list(), list(), list()
        if batch is None:
            return
        # print("train batch size", len(batch))
        for batch_data in batch:
            if batch_data is None or batch_data[0] is None:
                continue
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1])
            attention_masks.append(batch_data[2])
        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        attention_masks = torch.cat(attention_masks)
        loss = self.model(image_tensors, decoder_input_ids, attention_masks)[0]
        if loss is not None:
            self.log_dict({"train/loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if batch is None:
            return
        image_tensors, decoder_input_ids, _ = batch
        if image_tensors is None:
            return
        markdown = pad_sequence(
            decoder_input_ids,
            batch_first=True,
        )
        preds = self.model.inference(
            image_tensors=image_tensors,
            return_attentions=False,
            early_stopping=False,  # issue 174
        )["predictions"]
        gts = self.model.decoder.tokenizer.batch_decode(
            markdown, skip_special_tokens=True
        )
        metrics = get_metrics(gts, preds, pool=False)
        scores = {
            "val/" + key: sum(values) / len(values) for key, values in metrics.items()
        }
        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        if (
            self.validation_step_outputs is not None
            and len(self.validation_step_outputs) >= 1
        ):
            self.log_dict(self.validation_step_outputs[0], sync_dist=True)
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        if batch is None:
            return
        image_tensors, decoder_input_ids, _ = batch
        if image_tensors is None:
            return
        markdown = pad_sequence(
            decoder_input_ids,
            batch_first=True,
        )
        preds = self.model.inference(
            image_tensors=image_tensors,
            return_attentions=False,
            early_stopping=False,  # issue 174
        )["predictions"]
        gts = self.model.decoder.tokenizer.batch_decode(
            markdown, skip_special_tokens=True
        )

        metrics, output = get_step_results(gts, preds, pool=False)
        scores = {key: sum(values) / len(values) for key, values in metrics.items()}
        self.log_dict(
            {"test/" + key: values for key, values in scores.items()},
            prog_bar=True,
            on_step=True,
            reduce_fx="mean",
        )

        return output
    
    def _compute_metrics(self, outputs, save: bool, save_suffix: str = None):
        """
        outputs: [{ result: { pred: str, gt: str }, metrics: { ... } }]
        """
        if outputs is None:
            raise Exception("No test scores")

        metrics = {}
        for output in outputs:
            for key, value in output["metrics"].items():
                metrics.setdefault(key, []).append(value)

        final_scores = {}
        for metric, vals in metrics.items():
            final_scores[metric] = np.mean(vals)
        try:
            print(
                f"Total: {len(vals)}, Edit Distance accuracy score: {final_scores['edit_dist']:.2f}, BLEU score: {final_scores['bleu']:.2f}, precision: {final_scores['precision']:.2f}"
            )
        except:
            print(len(vals), final_scores)
            pass

        if save:
            assert save_suffix is not None, "save_suffix is None"
            save_dir = (
                Path(self.config.get("test_result_dir", "result_test"))
                / self.config.get("exp_name", "exp")
                / self.config.get("exp_version", "result")
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            # scores_path = str(save_dir / f"test_scores_{save_suffix}.json")
            # with open(scores_path, "w") as f:
            #     json.dump(final_scores, f, ensure_ascii=False, indent=2)

            outputs_path = str(save_dir / f"test_outputs_{save_suffix}.json")
            with open(outputs_path, "w") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
            print(f"Saved test results to {save_dir} ({save_suffix})")
        
        return final_scores

    def on_test_batch_end(self, output, batch, batch_idx):
        outputs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(outputs, output)
        torch.distributed.barrier()
        if not self.trainer.is_global_zero:
            return

        flattened_outputs = [item for sublist in outputs for item in sublist]
        print(
            f"batch {batch_idx}: Gathered {len(outputs)} device outputs, total {len(flattened_outputs)} steps"
        )

        self.test_step_outputs += flattened_outputs

        if len(self.test_step_outputs) > self.config.get("test_save_every_steps", 1000):
            current_steps = self.test_saved_outputs_count + len(self.test_step_outputs)
            print(f"Save test results at step {self.test_saved_outputs_count}-{current_steps}")
            self._compute_metrics(self.test_step_outputs, save=True, save_suffix=f"{self.test_saved_outputs_count}-{current_steps}")
            self.test_saved_outputs_count += len(self.test_step_outputs)
            self.test_step_outputs = []
        else:
            self._compute_metrics(flattened_outputs, save=False)

    def on_test_epoch_end(self):
        torch.distributed.barrier()
        if not self.trainer.is_global_zero:
            return
        current_steps = self.test_saved_outputs_count + len(self.test_step_outputs)
        print(f"Remain {len(self.test_step_outputs)}. Save test results at step {self.test_saved_outputs_count}-{current_steps}")
        self._compute_metrics(self.test_step_outputs, save=True, save_suffix=f"{self.test_saved_outputs_count}-{current_steps}")
        self.test_saved_outputs_count += len(self.test_step_outputs)
        self.test_step_outputs = []

        print(f"Done. Total saved: {self.test_saved_outputs_count}")

    def configure_optimizers(self):
        def _get_device_count():
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            elif torch.backends.mps.is_available():
                # Can MPS have more than one device?
                return 1
            return 1

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert (
                len(self.config.train_batch_sizes) == 1
            ), "Set max_epochs only if the number of datasets is 1"
            steps = self.config.num_training_samples_per_epoch
            max_iter = (self.config.max_epochs * steps) / max(
                1,
                (
                    self.config.train_batch_sizes[0]
                    * _get_device_count()
                    * self.config.get("num_nodes", 1)
                ),
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = (
                min(self.config.max_steps, max_iter)
                if max_iter is not None
                else self.config.max_steps
            )

        assert max_iter is not None
        print
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.exponential_scheduler(
                optimizer,
                self.config.warmup_steps,
                self.config.lr,
                self.config.get("min_lr", 5e-5),
                self.config.get("gamma", 0.9996),
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": self.config.get("lr_step", 1),
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=5e-5, gamma=0.9999):
        def lr_lambda(x):
            if x > warmup_steps or warmup_steps <= 0:
                if lr * gamma ** (x - warmup_steps) > min_lr:
                    return gamma ** (x - warmup_steps)
                else:
                    return min_lr / lr
            else:
                return x / warmup_steps

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = (
            Path(self.config.result_path)
            / self.config.exp_name
            / self.config.exp_version
        )
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class NougatDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.test_batch_sizes = self.config.test_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.train_datasets),
                batch_size=self.train_batch_sizes[0],
                num_workers=self.config.num_workers,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=True,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    def val_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.val_datasets),
                batch_size=self.val_batch_sizes[0],
                num_workers=self.config.num_workers,
                pin_memory=True,
                shuffle=False,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    def test_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.test_datasets),
                batch_size=self.test_batch_sizes[0],
                num_workers=self.config.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass
