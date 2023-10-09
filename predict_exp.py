"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
from pathlib import Path
import logging
import re
import argparse
import re
import os
from functools import partial
from typing import Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers.file_utils import ModelOutput
from transformers import StoppingCriteriaList, StoppingCriteria
from export.ModifiedStoppingCriteriaScores import ModifiedStoppingCriteriaScores
from nougat import NougatModel
from nougat.model import StoppingCriteriaScores, batch, subdiv
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible, postprocess
import pypdf

logging.basicConfig(level=logging.INFO)


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
        type=Path,
        default=None,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="0.1.0-small",
        help=f"Model tag to use.",
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed PDF, discarding previous predictions.",
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
    parser.add_argument(
        "--pages",
        "-p",
        type=str,
        help="Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDF input.",
    )
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    args = parser.parse_args()
    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        # input is a list of pdfs
        try:
            pdfs_path = args.pdf[0]
            if pdfs_path.is_dir():
                args.pdf = list(pdfs_path.rglob("*.pdf"))
            else:
                args.pdf = [
                    Path(l) for l in open(pdfs_path).read().split("\n") if len(l) > 0
                ]
            logging.info(f"Found {len(args.pdf)} files.")
        except:
            pass
    if args.pages and len(args.pdf) == 1:
        pages = []
        for p in args.pages.split(","):
            if "-" in p:
                start, end = p.split("-")
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(p) - 1)
        args.pages = pages
    else:
        args.pages = None
    return args


def inference_exp(
    model,
    image: Image.Image = None,
    image_tensors: Optional[torch.Tensor] = None,
    return_attentions: bool = False,
    early_stopping: bool = True,
):
    """
    Generate a token sequence in an auto-regressive manner.

    Args:
        image: input document image (PIL.Image)
        image_tensors: (1, num_channels, height, width)
            convert prompt to tensor if image_tensor is not fed
    """
    output = {
        "predictions": list(),
        "sequences": list(),
        "repeats": list(),
        "repetitions": list(),
    }
    if image is None and image_tensors is None:
        logging.warn("Image not found")
        return output

    if image_tensors is None:
        image_tensors = model.encoder.prepare_input(image).unsqueeze(0)

    if model.device.type != "mps":
        image_tensors = image_tensors.to(next(model.parameters()).dtype)

    image_tensors = image_tensors.to(model.device)

    last_hidden_state = model.encoder(image_tensors)

    encoder_outputs = ModelOutput(
        last_hidden_state=last_hidden_state, attentions=None
    )

    if len(encoder_outputs.last_hidden_state.size()) == 1:
        encoder_outputs.last_hidden_state = (
            encoder_outputs.last_hidden_state.unsqueeze(0)
        )

    # get decoder output
    modifiedStoppingCriteriaScores = ModifiedStoppingCriteriaScores()

    decoder_output = model.decoder.model.generate(
        encoder_outputs=encoder_outputs,
        min_length=1,
        max_length=model.config.max_length,
        pad_token_id=model.decoder.tokenizer.pad_token_id,
        eos_token_id=model.decoder.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[
            [model.decoder.tokenizer.unk_token_id],
        ],
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=return_attentions,
        do_sample=False,
        stopping_criteria=StoppingCriteriaList(
            [modifiedStoppingCriteriaScores] if early_stopping else []
        ),
    )
    output["repetitions"] = decoder_output.sequences.clone()
    output["sequences"] = decoder_output.sequences.clone()
    batch_size = len(decoder_output.sequences)

    logits = torch.stack(decoder_output.scores, 1).cpu().max(-1)
    values = logits.values
    indices = logits.indices

    for b in range(batch_size):
        mask = indices[b] != model.decoder.tokenizer.pad_token_id
        N = mask.sum().item()
        var = np.array(
            [np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())]
        )
        if len(var) < 10:
            output["repeats"].append(None)
            continue
        varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1])
        minlen = 120
        if (
            indices[b] == model.decoder.tokenizer.eos_token_id
        ).any() and N + 1 < indices.shape[1]:
            # there is an end to the generation, likely no repetitions
            output["repeats"].append(None)
            continue
        small_var = np.where(varvar < 0.045)[0]
        if early_stopping and len(small_var) > 1:
            if np.all(np.diff(small_var) < 2):
                idx = int(min(max(small_var[0], 1) * 1.08 + minlen, 4095))
                if idx / N > 0.9:  # at most last bit
                    output["repeats"].append(None)
                    continue
                elif small_var[0] < 30:
                    idx = 0
                logging.warn("Found repetitions in sample %i" % b)
                output["repeats"].append(idx)
                output["sequences"][b, idx:] = model.decoder.tokenizer.pad_token_id
                output["repetitions"][b, :idx] = model.decoder.tokenizer.pad_token_id
            else:
                output["repeats"].append(None)
        else:
            output["repeats"].append(None)
    output["repetitions"] = model.decoder.tokenizer.batch_decode(
        output["repetitions"], skip_special_tokens=True
    )
    output["predictions"] = postprocess(
        model.decoder.tokenizer.batch_decode(
            output["sequences"], skip_special_tokens=True
        ),
        markdown_fix=False,
    )

    if return_attentions:
        output["attentions"] = {
            "self_attentions": decoder_output.decoder_attentions,
            "cross_attentions": decoder_output.cross_attentions,
        }
    
    print(modifiedStoppingCriteriaScores.count)

    return output

def main():
    args = get_args()
    model = NougatModel.from_pretrained(args.checkpoint)
    model = move_to_device(model, bf16=not args.full_precision, cuda=args.batchsize > 0)
    if args.batchsize <= 0:
        # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
        args.batchsize = 1
    model.eval()
    datasets = []
    for pdf in args.pdf:
        if not pdf.exists():
            continue
        if args.out:
            out_path = args.out / pdf.with_suffix(".mmd").name
            if out_path.exists() and not args.recompute:
                logging.info(
                    f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                )
                continue
        try:
            dataset = LazyDataset(
                pdf,
                partial(model.encoder.prepare_input, random_padding=False),
                args.pages,
            )
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)
    if len(datasets) == 0:
        return
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    predictions = []
    file_index = 0
    page_num = 0
    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
        model_output = inference_exp(model,
            image_tensors=sample, early_stopping=args.skipping
        )
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)
            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                if args.out:
                    out_path = args.out / Path(is_last_page[j]).with_suffix(".mmd").name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")
                else:
                    print(out, "\n\n")
                predictions = []
                page_num = 0
                file_index += 1


if __name__ == "__main__":
    main()
