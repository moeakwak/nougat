from functools import partial
import torch
from PIL import Image
import io

import logging

from nougat.model import NougatModel
from nougat.utils.dataset import ImageDataset
from nougat.utils.device import default_batch_size, move_to_device


def load_model(model_path="./model/nougat-standard-0529", full_precision=True, device="cuda:0"):
    model: NougatModel = NougatModel.from_pretrained(model_path)
    if not full_precision:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()
    return model


def inference(model: NougatModel, images: list[Image.Image], skipping=True):
    prepare_input = partial(model.encoder.prepare_input, random_padding=False)
    inputs = [prepare_input(image) for image in images]

    image_tensor = torch.stack(inputs)
    model_output = model.inference(image_tensors=image_tensor, early_stopping=skipping)
    predictions = []
    for i, output in enumerate(model_output["predictions"]):
        remark = ""
        if output.strip() == "[MISSING_PAGE_POST]":
            # uncaught repetitions -- most likely empty page
            remark = f"\n\n[MISSING_PAGE_EMPTY:[{i}]\n\n"
        elif model_output["repeats"][i] is not None:
            if model_output["repeats"][i] > 0:
                # If we end up here, it means the output is most likely not complete and was truncated.
                logging.warning(f"Skipping [{i}] due to repetitions.")
                remark = f"\n\n[MISSING_PAGE_FAIL:[{i}]]\n\n"
            else:
                # If we end up here, it means the document page is too different from the training domain.
                # This can happen e.g. for cover pages.
                remark = f"\n\n[MISSING_PAGE_EMPTY:{i}]\n\n"
        predictions.append((output.strip() + remark).strip())

    return predictions, model_output


def main():
    model = load_model()
    image_paths = [
        "../nougat-data/paired_book/营销计量/14.png",
        "../nougat-data/paired_book/营销计量/45.png",
    ]
    images = [Image.open(path) for path in image_paths]
    predictions = inference(model, images)
    return predictions
