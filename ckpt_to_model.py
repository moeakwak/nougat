"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
from os.path import basename
from pathlib import Path
from torch.utils.data import Subset
import lightning.pytorch as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.plugins import CheckpointIO
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from sconf import Config
import torch
from lightning_module import NougatDataPLModule, NougatModelPLModule

import warnings
warnings.filterwarnings("ignore")


def save_model(config, ckpt_path: str, output_path: str = None):
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module = NougatModelPLModule.load_from_checkpoint(ckpt_path, config=config)

    output_path = output_path or Path(ckpt_path).stem + ".bin"

    print("Model loaded from {ckpt_path}")
    torch.save(model_module.model.state_dict(), output_path)

    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/nougat_finetune_standard.yaml")
    parser.add_argument("--ckpt", type=str, default="result_models/nougat_finetune_standard/20240517_163208/epoch=21-step=184030-val_bleu=0.76.ckpt")
    parser.add_argument("--output", "-O", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    
    save_model(config, args.ckpt, args.output)
