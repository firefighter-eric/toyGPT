import glob
import json
from itertools import chain

from datasets import Dataset
import opencc


# %% config

NUM_PROC = 8
MAX_LENGTH = 1024
BATCH = 10000
tokenizer_path = 'facebook/opt-125m'
exp_dir = 'data/chinese-poetry/quan_tang_shi'
raw_data_dir = f'{exp_dir}/json'
p0_json_dir = f'{exp_dir}/p0_json_zh_cn'

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
    text = converter.convert(text)
    return {'text': text}

converter = opencc.OpenCC('t2s.json')
ds = ds.map(gen_poem_text, num_proc=NUM_PROC)
ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
# %% save

ds['train'].to_json(f'{p0_json_dir}/train.json', force_ascii=False)
ds['test'].to_json(f'{p0_json_dir}/test.json', force_ascii=False)
