"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
import datetime
import os
from os.path import basename
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
    GradientAccumulationScheduler,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.plugins import CheckpointIO
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from sconf import Config

from nougat import NougatDataset
from lightning_module import NougatDataPLModule, NougatModelPLModule

import logging

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)



def test(config, ckpt_path: str):
    pl.seed_everything(config.get("seed", 42), workers=True)

    print("Loading model and data modules...")
    model_module = NougatModelPLModule(config)
    data_module = NougatDataPLModule(config)

    # add datasets to data_module
    datasets = {"test": []}
    dataset_pathes_by_split = config.dataset_pathes_by_split
    for split in ["test"]:
        datasets[split].append(
            NougatDataset(
                dataset_path=dataset_pathes_by_split[split][0],
                nougat_model=model_module.model,
                max_length=config.max_length,
                split=split,
                image_dir=config.image_dir
            )
        )
    data_module.test_datasets = datasets["test"]

    print("Creating logger...")

    # if not config.debug:
    #     logger = Logger(config.exp_name, project="Nougat", config=dict(config))
    # else:
    logger = TensorBoardLogger(
        save_dir=config.test_result_dir,
        name=config.exp_name,
        version=Path(ckpt_path).stem,
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        accelerator="auto",
        log_every_n_steps=15,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        logger=logger
    )

    print("Starting testing...")
    trainer.test(
        model_module,
        data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--job", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default="last")
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    config.debug = args.debug
    config.job = args.job
    if not config.get("exp_name", False):
        raise ValueError("exp_name is required")
    config.exp_version = Path(args.ckpt_path).stem
    
    test(config, args.ckpt_path)
