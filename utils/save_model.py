from argparse import ArgumentParser

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = ArgumentParser()
parser.add_argument('--model_name', '-m', type=str, default=r'H:\models\llama-7b-pt')
parser.add_argument('--tokenizer_name', '-t', type=str, default='huggyllama/llama-7b')
parser.add_argument('--output_path', '-o', type=str, default='H:/models/llama-7b')

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

model.save_pretrained(args.output_path, safe_serialization=True)
tokenizer.save_pretrained(args.output_path)
