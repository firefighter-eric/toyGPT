import glob
import json
from itertools import chain

from datasets import Dataset
from transformers import AutoTokenizer

# %% config

NUM_PROC = 8
MAX_LENGTH = 512
BATCH = 10000
tokenizer_path = 'facebook/opt-125m'
exp_dir = 'data/chinese-poetry/quan_tang_shi'
raw_data_dir = f'{exp_dir}/json'
p0_json_dir = f'{exp_dir}/p0_json'
p1_dataset_dir = f'{exp_dir}/p1_opt_l{MAX_LENGTH}_dataset'

# %% load data

data_paths = glob.glob(f'{raw_data_dir}/*.json')
data = list(chain(*[json.load(open(_)) for _ in data_paths]))
ds = Dataset.from_list(data)
ds = ds.select_columns(['title', 'author', 'paragraphs'])


# %% process
def gen_poem_text(example):
    title = example['title']
    content = '\n'.join(example['paragraphs'])
    text = f'{title}\n{content}\n'
    return {'text': text}


ds = ds.map(gen_poem_text, num_proc=NUM_PROC)
ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
# %% save

ds['train'].to_json(f'{p0_json_dir}/train.json', force_ascii=False)
ds['test'].to_json(f'{p0_json_dir}/test.json', force_ascii=False)

# %% tokenize

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
eos_token = tokenizer.eos_token


def clm_tokenize(example):
    text = example['text']
    input_ids = tokenizer(text)['input_ids']
    input_ids = list(chain(*input_ids))
    token_length = len(input_ids) // MAX_LENGTH * MAX_LENGTH
    input_ids_list = [input_ids[i: i + MAX_LENGTH] for i in range(0, token_length, MAX_LENGTH)]
    return {'input_ids': input_ids_list}


ds = ds.map(clm_tokenize, batched=True, batch_size=BATCH, num_proc=NUM_PROC, remove_columns=ds['train'].column_names)
ds = ds.shuffle(seed=42)

# example
print(tokenizer.decode(ds['train'][0]['input_ids']))

# save
ds.save_to_disk(p1_dataset_dir)

# python -m exp.poem.process
