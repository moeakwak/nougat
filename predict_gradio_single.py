import gradio as gr
from PIL import Image
from nougat.postprocessing import markdown_compatible, close_envs
import logging
from pathlib import Path
from predict_pretrained import inference, load_model

logging.basicConfig(level=logging.INFO)


print("loading model...")
model = load_model(device="cuda:2")


def get_examples():
    examples = Path("gradio_examples").glob("*")
    results = []
    for item in examples:
        if item.is_file():
            results.append(item)
        else:
            results += [str(p) for p in item.glob("*.png")]
    return results


examples = get_examples()


def run_inference(image_inputs: list[Image.Image] | Image.Image):
    if isinstance(image_inputs, Image.Image):
        imgs = [image_inputs]
    else:
        [tup[0] for tup in image_inputs]
    preds, model_outputs = inference(model, imgs)
    # preds = ["1", "2", "3"]
    outputs = []

    for i, pred in enumerate(preds):
        outputs.append(markdown_compatible(pred))
        outputs.append("---")

    result = "\n\n".join(outputs)

    return result, result

    # submit_button.click(run_inference, inputs=images, outputs=output_boxes)


print(examples)


iface = gr.Interface(
    run_inference,
    inputs=[gr.Image(label="上传页面图像", type="pil")],
    outputs=[
        gr.Markdown(
            label="预测结果",
            latex_delimiters=[
                {"left": "$", "right": "$", "display": False},
                {"left": "$$", "right": "$$", "display": True},
            ],
            line_breaks=True,
        ),
        gr.TextArea(label="原始预测结果"),
    ],
    examples=examples,
    cache_examples="lazy",
).queue(default_concurrency_limit=1)


iface.launch(server_name="0.0.0.0", server_port=7861)
