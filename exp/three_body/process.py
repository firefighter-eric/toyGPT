from itertools import chain

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# %% config

NUM_PROC = 8
MAX_LENGTH = 1024
BATCH = 10000
tokenizer_path = 'facebook/opt-125m'
exp_dir = 'data/three-body'
input_data_path = f'{exp_dir}/p0/three_body.txt'
p1_dataset_dir = f'{exp_dir}/p1_opt_l{MAX_LENGTH}_dataset'

# %% load data
ds: Dataset = load_dataset('text', data_files=input_data_path, split='train')
print(ds)

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


ds = ds.map(clm_tokenize, batched=True, batch_size=BATCH, num_proc=NUM_PROC, remove_columns=ds.column_names)
ds = ds.shuffle(seed=42)

# %% example
dsd = ds.train_test_split(test_size=0.01, shuffle=True, seed=42)
print(tokenizer.decode(dsd['train'][0]['input_ids']))
print(dsd)
# %% save
dsd.save_to_disk(p1_dataset_dir)

# python exp/three_body/process.py
