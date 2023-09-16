from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer


class DataProcessor:

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 1024, test_size: float = 0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_size = test_size

    def load_text(self, path: str) -> DatasetDict:
        assert self.tokenizer.pad_token_id is not None

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

    def load_dataset_dict(self, path: str) -> DatasetDict:
        dataset = DatasetDict.load_from_disk(dataset_dict_path=path)
        return dataset


def load_data(path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024, test_size: float = 0.1) -> DatasetDict:
    processor = DataProcessor(tokenizer=tokenizer, max_length=max_length, test_size=test_size)
    if path.endswith('.txt'):
        data = processor.load_text(path=path)
    else:
        data = processor.load_dataset_dict(path=path)
    return data
