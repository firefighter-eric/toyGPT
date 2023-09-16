from datasets import DatasetDict


def load_data(path: str) -> DatasetDict:
    data = DatasetDict.load_from_disk(dataset_dict_path=path)
    return data
