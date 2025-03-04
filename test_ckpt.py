"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import argparse
import json
import os
import logging
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nougat import NougatModel
from nougat.metrics import compute_metrics
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import NougatDataset
from nougat.utils.device import move_to_device
from lightning_module import NougatModelPLModule, NougatDataPLModule
from sconf import Config


def load_module(ckpt_path: str, config: str):
    # ckpt_dir = ckpt_path.parent
    ckpt_path = Path(ckpt_path)
    config = Config(config)
    model = NougatModelPLModule.load_from_checkpoint(ckpt_path, config=config)
    return model


def test(args):
    module = load_module(args.checkpoint, args.config)
    config = Config(args.config)
    pretrained_model = module.model

    pretrained_model = move_to_device(pretrained_model)

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    else:
        logging.warning("Results can not be saved. Please provide a -o/--save_path")
    predictions = []
    ground_truths = []
    metrics = defaultdict(list)
    assert args.split in config.dataset_pathes_by_split
    dataset_path = config.dataset_pathes_by_split[args.split][0]
    print("Evaluating model on dataset split: {} {}".format(args.split, dataset_path))
    dataset = NougatDataset(
        dataset_path=dataset_path,
        nougat_model=pretrained_model,
        max_length=config.max_length,
        split=args.split,
        image_dir=config.image_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=6,
        pin_memory=True,
        shuffle=args.shuffle,
        collate_fn=NougatDataPLModule.ignore_none_collate,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        image_tensors, decoder_input_ids, _ = sample
        if image_tensors is None:
            return
        if args.num_samples != -1 and len(predictions) >= args.num_samples:
            break
        ground_truth = pretrained_model.decoder.tokenizer.batch_decode(
            decoder_input_ids, skip_special_tokens=True
        )
        outputs = pretrained_model.inference(
            image_tensors=image_tensors,
            return_attentions=False,
        )["predictions"]
        predictions.extend(outputs)
        ground_truths.extend(ground_truth)
        with Pool(args.batch_size) as p:
            _metrics = p.starmap(compute_metrics, iterable=zip(outputs, ground_truth))
            for m in _metrics:
                for key, value in m.items():
                    metrics[key].append(value)

            print({key: sum(values) / len(values) for key, values in metrics.items()})

    scores = {}
    for metric, vals in metrics.items():
        scores[f"{metric}_accuracies"] = vals
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        print(scores)
        pass

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        with open(args.save_path, "w") as f:
            json.dump(scores, f)

    return predictions


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", "-c", type=Path, default=None, help=".ckpt文件路径"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--save_path",
        "-o",
        type=str,
        default="result/test_result.json",
        help="json file to save results to",
    )
    parser.add_argument("--num_samples", "-N", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, left_argv = parser.parse_known_args()
    # args.checkpoint = get_checkpoint(args.checkpoint)

    predictions = test(args)
