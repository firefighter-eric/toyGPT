import os
from argparse import ArgumentParser

from datasets import DatasetDict, Dataset, load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer


class DataProcessor:

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 1024, test_size: float = 0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_size = test_size
        assert self.tokenizer.pad_token_id is not None
        assert self.tokenizer.eos_token_id is not None

    def tokenize_clm(self, examples):
        input_ids = self.tokenizer(examples['text'])['input_ids']
        for _ in input_ids:
            _.append(self.tokenizer.eos_token_id)
            _ += [self.tokenizer.pad_token_id] * (self.max_length - len(_))
            _ = _[:self.max_length]

        examples['input_ids'] = input_ids

        return examples

    def load_text(self, path: str) -> DatasetDict:

        with open(path, 'r') as f:
            text = f.read()

        input_ids = self.tokenizer(text)['input_ids']
        data = []
        for i in range(0, len(input_ids), self.max_length):
            batch = input_ids[i: i + self.max_length]
            if len(batch) < self.max_length:
                batch += [self.tokenizer.pad_token_id] * (self.max_length - len(batch))
            data.append({'input_ids': batch})
        dataset = Dataset.from_list(data)
        dataset = dataset.train_test_split(test_size=self.test_size, shuffle=True, seed=42)
        return dataset

    def load_json(self, path: str, split: bool = False) -> DatasetDict:
        if os.path.isdir(path):
            dataset = load_dataset('json', data_files={'train': f'{path}/train.json', 'test': f'{path}/test.json'})
        else:
            dataset = load_dataset('json', data_files=path, split='train')
            dataset.train_test_split(test_size=self.test_size, shuffle=True, seed=42)

        dataset = dataset.map(self.tokenize_clm, batched=True, num_proc=8)
        return dataset

    def load_dataset_dict(self, path: str) -> DatasetDict:
        dataset = DatasetDict.load_from_disk(dataset_dict_path=path)
        return dataset


def load_data(path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024, test_size: float = 0.1) -> DatasetDict:
    processor = DataProcessor(tokenizer=tokenizer, max_length=max_length, test_size=test_size)
    if path.endswith('.txt'):
        data = processor.load_text(path=path)
    else:
        data = processor.load_json(path=path)
    # else:
    #     data = processor.load_dataset_dict(path=path)
    print(data)
    print(data['train'][0]['input_ids'])
    print(tokenizer.decode(data['train'][0]['input_ids']))
    return data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default='data/chinese-poetry/quan_tang_shi/p0_json')
    parser.add_argument('--tokenizer_path', '-t', type=str, default='/mnt/h/models/chinese-llama-2-7b')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    d = load_data(path=args.data_path, tokenizer=tokenizer, max_length=64, test_size=0.1)

