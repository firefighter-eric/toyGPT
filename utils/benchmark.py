import time
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = ArgumentParser()
parser.add_argument('--model_path', '-m', type=str, default='/mnt/h/models/llama-7b')
parser.add_argument('--precision', '-p', type=str, default='bf16')

args = parser.parse_args()

model_args = {}
if args.precision == 'bf16':
    model_args['torch_dtype'] = torch.bfloat16
elif args.precision == 'fp16':
    model_args['torch_dtype'] = torch.float16
elif args.precision == 'fp32':
    model_args['torch_dtype'] = torch.float32
elif args.precision == 'int8':
    model_args['load_in_8bit'] = True
elif args.precision == 'int4':
    model_args['load_in_4bit'] = True
else:
    raise ValueError(f'Unsupported precision {args.precision}')

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto', **model_args)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
print('Model loaded')

input_ids = torch.tensor([[1]], dtype=torch.long, device='cuda:0')

o = model.generate(input_ids, do_sample=False, max_length=512, top_p=1, num_return_sequences=1)

input_length = input_ids.size(1)
output_length = len(o[0])
print('Input length:', input_length)
print('Output length:', output_length)

start_time = time.time()
for _ in tqdm(range(5)):
    model.generate(input_ids, do_sample=False, max_length=512, top_p=1, num_return_sequences=1)
end_time = time.time()

cost_time_per_seq = (end_time - start_time) / 5
ms_per_token = cost_time_per_seq / (output_length - input_length) * 1000
print(f'Cost time per seq: {cost_time_per_seq:.2f}s')
print(f'MS per token: {ms_per_token:.2f}ms')
