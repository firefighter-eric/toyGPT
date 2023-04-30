import os

import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import OPTForCausalLM

# %% config
model_path = 'facebook/opt-1.3b'
onnx_path = 'data/trt/opt-1.3b/opt-1.3b.flash.onnx'
convert = True

os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
if 'flash' in onnx_path:
    model = OPTForCausalLM.from_pretrained(model_path)
    print('flash model loaded')
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)

model.config.use_cache = False
tokenizer.model_max_length = 1024

input_ids_pt = tokenizer('hello world', return_tensors='pt', truncation=True)['input_ids']
input_ids_np = tokenizer('hello world', return_tensors='np', truncation=True)['input_ids']
# %% hf
o_pt = model(input_ids_pt)

# %% to onnx

if convert:
    torch.onnx.export(
        model=model,
        args=input_ids_pt,
        f=onnx_path,
        verbose=True,
        opset_version=15,
        input_names=['input_ids'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'seq_size'},
                      'output': {0: 'batch_size', 1: 'seq_size'}}
    )

# %%

session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
o_onnx = session.run(['output'], {'input_ids': input_ids_np})[0]

print('mean error:', torch.mean(torch.abs(o_pt.logits - torch.tensor(o_onnx))))
print('size', o_onnx.shape)
print('all close:', torch.allclose(o_pt.logits, torch.tensor(o_onnx), atol=1e-3))

# %%

print('done')
