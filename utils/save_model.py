from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = ArgumentParser()
parser.add_argument('--model_name', '-m', type=str, required=True)
parser.add_argument('--tokenizer_name', '-t', type=str, default='')
parser.add_argument('--precision', '-p', type=str, default='')
parser.add_argument('--output_path', '-o', type=str, default='')
args = parser.parse_args()
if args.tokenizer_name == '':
    args.tokenizer_name = args.model_name
if args.output_path == '':
    args.output_path = args.model_name + '-o'

model_args = {}

match args.precision:
    case 'bf16':
        model_args['torch_dtype'] = torch.bfloat16
    case 'fp16':
        model_args['torch_dtype'] = torch.float16
    case 'fp32':
        model_args['torch_dtype'] = torch.float32
    case 'int8':
        model_args['load_in_8bit'] = True
    case 'int4':
        model_args['load_in_4bit'] = True
    case '':
        pass
    case _:
        raise ValueError(f'Unsupported precision {args.precision}')

model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_args)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

model.save_pretrained(args.output_path, safe_serialization=False)
tokenizer.save_pretrained(args.output_path)
