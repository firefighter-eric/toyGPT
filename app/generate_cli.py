from argparse import ArgumentParser

from transformers import pipeline

from models import OPTForCausalLM

parser = ArgumentParser()
parser.add_argument('--model_class', '-mc', type=str, default='auto')
parser.add_argument('--model_path', '-m', type=str)
parser.add_argument('--tokenizer_path', '-t', type=str, required=False)
parser.add_argument('--num_return_sequences', '-n', type=int, default=5)
args = parser.parse_args()

model_path = args.model_path
tokenizer_path = args.tokenizer_path or model_path

if args.model_class == 'auto':
    model = model_path
else:
    model = OPTForCausalLM.from_pretrained(model_path)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer_path, device=0)

while True:
    text = input('Intput:')
    generate_output = generator(text,
                                num_return_sequences=args.num_return_sequences,
                                do_sample=True,
                                max_length=200,
                                top_p=0.6)
    for output in generate_output:
        print(output['generated_text'])
        print('-' * 100)

# python -m app.generate_cli -mc opt -m gpt2 -t gpt2 -n 5
# python -m app.generate_cli -m gpt2 -t gpt2 -n 5
