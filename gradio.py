from functools import partial
import gradio as gr
import torch
from PIL import Image
from nougat import NougatModel
from nougat.utils.device import move_to_device, default_batch_size
from nougat.postprocessing import markdown_compatible, close_envs
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def process_image(img):
    # 将Numpy数组转为PIL图像
    pil_img = Image.fromarray(np.uint8(img * 255))
    
    # 使用model.encoder.prepare_input来准备图像
    prepare_input = partial(model.encoder.prepare_input, random_padding=False)
    img_tensor = prepare_input(pil_img)
    
    # 用模型进行推断
    model_output = model.inference(image_tensors=img_tensor.unsqueeze(0))  # 可能需要调整，使其匹配实际的模型输入
    
    for j, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = (
                        "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"
                    )
                rest = close_envs(model_output["repetitions"][j]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

             (
                markdown_compatible(output) + disclaimer
            )

    # 获取文本输出并进行Markdown渲染
    text_output = model_output['predictions'][0]
    
    if args.markdown:
        markdown_output = markdown_compatible(text_output)
    else:
        markdown_output = text_output

    return text_output, markdown_output


# 获取模型和其它设置，可以参考原始代码
model_tag = "0.1.0-small"  # 可以根据需要调整
checkpoint = None  # 可以根据需要调整
full_precision = True  # 可以根据需要调整
model: NougatModel = NougatModel.from_pretrained(checkpoint, model_tag=model_tag)
model = move_to_device(model, bf16=not full_precision, cuda=default_batch_size() > 0)
model.eval()

# 创建Gradio界面
iface = gr.Interface(
    fn=process_image,
    inputs=gr.inputs.Image(type="numpy", label="Upload an image"),  
    outputs=[
        gr.outputs.Textbox(label="Raw Text"),
        gr.outputs.Textbox(label="Markdown Text", type="markdown")
    ],
    live=True,
    capture_session=True,
)

# 启动Gradio界面
iface.launch()
