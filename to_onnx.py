import torch
import sys

# sys.path.append('..')
# sys.path.append('.')

from nougat import NougatModel
from nougat.model import RunningVarTorch, StoppingCriteriaScores
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import LazyDataset
from pathlib import Path
import pypdf
import torch.onnx
from transformers.file_utils import ModelOutput
from transformers import StoppingCriteriaList, StoppingCriteria

print("torch version:", torch.__version__)

checkpoint = get_checkpoint(None, model_tag="0.1.0-base")
print("checkpoint:", checkpoint)

device="cuda"

model = NougatModel.from_pretrained(checkpoint)
model = model.to(device)
model.eval()


pad_token_id = model.decoder.tokenizer.pad_token_id
eos_token_id = model.decoder.tokenizer.eos_token_id
unk_token_id = model.decoder.tokenizer.unk_token_id
print("pad_token_id:", pad_token_id)
print("eos_token_id:", eos_token_id)
print("unk_token_id:", unk_token_id)


class ONNXExportableNougatModel(torch.nn.Module):
    def __init__(self, nougat_model):
        super().__init__()
        self.encoder = nougat_model.encoder
        self.decoder = nougat_model.decoder

    def forward(self, image_tensors):
        last_hidden_state = self.encoder(image_tensors)
        return last_hidden_state
        # encoder_outputs = ModelOutput(
        #     last_hidden_state=last_hidden_state, attentions=None
        # )
        # encoder_outputs = (last_hidden_state, None, None)
        # decoder_output = self.decoder.model.generate(
        #     encoder_outputs=encoder_outputs,
        #     min_length=1,
        #     max_length=4096,
        #     pad_token_id=pad_token_id,
        #     eos_token_id=eos_token_id,
        #     use_cache=True,
        #     bad_words_ids=[
        #         [unk_token_id],
        #     ],
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     output_attentions=False,
        #     do_sample=False,
        #     stopping_criteria=StoppingCriteriaList(
        #         [StoppingCriteriaScores()]
        #     ),
        # )
        # return decoder_output

exportable_model = ONNXExportableNougatModel(model).to(device).float()

pdf_path = Path("example.pdf")

dataset = LazyDataset(pdf_path, model.encoder.prepare_input)
print(dataset, len(dataset))

example_inputs = dataset[0][0].unsqueeze(0).to(device).float()
print(example_inputs.shape, example_inputs.dtype)

onnx_file_path = "nougat_model_encoder.onnx"

# torch.onnx.dynamo_export(
#     exportable_model, 
#     example_inputs
# ).save(onnx_file_path)

torch.onnx.export(
    exportable_model, 
    example_inputs,
    onnx_file_path,
    # verbose=True,
    verbose=False,
    input_names=["input"],
    output_names=["output"],
)


print(f"Model exported to {onnx_file_path}")
