import os
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models import OPTForCausalLM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# %%
model_path = 'facebook/opt-125m'
# model_path = 'facebook/opt-1.3b'

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.model_max_length = 1024

input_ids = tokenizer.encode('hello world' * 1024, truncation=True, return_tensors='pt').cuda()
input_ids_np = tokenizer.encode('hello world' * 1024, truncation=True, return_tensors='np').astype('int32')

# model_fp32 = AutoModelForCausalLM.from_pretrained(model_path).cuda()
# model = AutoModelForCausalLM.from_pretrained(model_path).cuda().to(torch.bfloat16)
# model_compiled = torch.compile(model)
# flash_model = OPTForCausalLM.from_pretrained(model_path).cuda().to(torch.bfloat16)
# flash_model_compiled = torch.compile(flash_model)

models = [
    # {'name': 'model_fp32', 'model': model_fp32, 'input': input_ids},
#     {'name': 'model', 'model': model, 'input': input_ids},
#     {'name': 'model_compiled', 'model': model_compiled, 'input': input_ids},
#     {'name': 'flash_model', 'model': flash_model, 'input': input_ids},
#     {'name': 'flash_model_compiled', 'model': flash_model_compiled, 'input': input_ids},
]

trt = True
if trt:
    from models.trt.tensorrt_model import TRTModel

    trt_model = TRTModel('data/trt/opt-1.3b.fp16.trt')
    models.append({'name': 'trt_model', 'model': trt_model, 'input': input_ids_np})

print('model loaded')


# %% benchmark


def timeit(func, *args, n=100):
    o = func(*args)
    start = time.time()
    for _ in range(n):
        func(*args)
    end = time.time()
    t = (end - start) / n
    # print(f'mean time: {t:.4f} s')
    return t


time_dict = {}
for example in models:
    t = timeit(example['model'], example['input'])
    time_dict[example['name']] = t

df = pd.DataFrame.from_dict(time_dict, orient='index', columns=['time'])
print(df.to_markdown())

"""
opt-125m
|                      |       time |
|:---------------------|-----------:|
| model                | 0.0157292  |
| model_compiled       | 0.00941361 |
| flash_model          | 0.0100424  |
| flash_model_compiled | 0.00804039 |


opt-1.3b
|                      |      time |
|:---------------------|----------:|
| model                | 0.0767244 |
| model_compiled       | 0.0647843 |
| flash_model          | 0.0560017 |
| flash_model_compiled | 0.0540989 |
"""
