from argparse import ArgumentParser

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# %%
parser = ArgumentParser()
parser.add_argument('--model_path', '-m', type=str, default='/mnt/h/models/llama-7b')
parser.add_argument('--lora_path', '-l', type=str, default='')
# parser.add_argument('--precision', '-p', type=str, default='bf16')
parser.add_argument('--prompt', '-t', type=str, default='你好')

args = parser.parse_args()
print(args)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto', torch_dtype=torch.bfloat16)
if args.lora_path:
    # model = get_peft_model(model, args.lora_path)
    model = PeftModel.from_pretrained(model, args.lora_path, device_map='auto', torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

output = pipeline(args.prompt, max_length=512, do_sample=False, top_p=1, num_return_sequences=1)
print(output)
while True:
    prompt = input('Human: ')
    output = pipeline(prompt, max_new_tokens=32, do_sample=False, top_p=1, num_return_sequences=1, return_full_text=False)
    for o in output:
        print('Bot:', o['generated_text'])
    print('-' * 100)
