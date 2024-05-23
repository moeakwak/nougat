import sys
from pathlib import Path
import logging
import re
import argparse
import re
import os
from functools import partial
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from lightning_module import NougatModelPLModule
from nougat import NougatModel
from nougat.utils.dataset import ImageDataset, LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible

from sconf import Config

logging.basicConfig(level=logging.INFO)

def load_module(ckpt_path: str, config: str):
    # ckpt_dir = ckpt_path.parent
    ckpt_path = Path(ckpt_path)
    config = Config(config)
    model = NougatModelPLModule.load_from_checkpoint(ckpt_path, config=config)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=default_batch_size(),
        help="Batch size to use.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="Path to checkpoint file.",
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed images, discarding previous predictions.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Use float32 instead of bfloat16. Can speed up CPU conversion for some setups.",
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="Do not add postprocessing step for markdown compatibility.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Add postprocessing step for markdown compatibility (default).",
    )
    parser.add_argument(
        "--no-skipping",
        dest="skipping",
        action="store_false",
        help="Don't apply failure detection heuristic.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="Image(s) to process.")
    args = parser.parse_args()
    
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if len(args.images) == 1 and args.images[0].is_dir():
        # args.images = list(args.images[0].rglob("*.png"))
        args.images = list(args.images[0].rglob("*.[jpeg jpg png]*"))
        logging.info(f"Found {len(args.images)} files in the directory.")
    # Confirm all files are PNG images
    for img_path in args.images:
        if img_path.suffix not in [".png", ".jpeg", ".jpg"]:
            logging.error(f"File {img_path} is not a image.")
            sys.exit(1)
    return args


def main():
    args = get_args()
    module = load_module(args.checkpoint, args.config)
    model = module.model
    del module
    model = move_to_device(model, bf16=not args.full_precision, cuda=args.batchsize > 0)
    if args.batchsize <= 0:
        args.batchsize = 1
    model.eval()

    # Use ImageDataset for images.
    dataset = ImageDataset(
        args.images, partial(model.encoder.prepare_input, random_padding=False)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
    )

    predictions = []
    all_final_output = []
    for i, sample in enumerate(tqdm(dataloader)):
        model_output = model.inference(
            image_tensors=sample, early_stopping=args.skipping
        )
        final_output = []
        for j, output in enumerate(model_output["predictions"]):
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                final_output.append(
                    f"\n\n[MISSING_PAGE_EMPTY:{args.images[i]}[{j}]\n\n"
                )
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(
                        f"Skipping {args.images[i]}[{j}] due to repetitions."
                    )
                    final_output.append(
                        f"\n\n[MISSING_PAGE_FAIL:{args.images[i]}[{j}]]\n\n"
                    )
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    final_output.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )

            if args.markdown:
                output = markdown_compatible(output)
            final_output.append(output)
            all_final_output.append(output)
        predictions.append("".join(final_output).strip())
        print(f"image {i} done")

    # Save or print the predictions.
    for filepath, pred in zip(args.images, predictions):
        filename = filepath.stem
        if args.out:
            out_path = args.out / f"{filename}.mmd"
            out_path.write_text(pred, encoding="utf-8")
        else:
            print(pred, "\n\n")
    
    # output all
    out_dir = args.out / "all_final_output"
    out_dir.mkdir(exist_ok=True)
    for i, o in enumerate(all_final_output):
        out_path = out_dir / f"{i}.mmd"
        o = markdown_compatible(o)
        out_path.write_text(o, encoding="utf-8")


if __name__ == "__main__":
    main()
