import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from os import path as osp
from pprint import pprint
from typing import Optional

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from transformers import HfArgumentParser
from transformers import TrainingArguments

sys.path.append(osp.join(osp.dirname(__file__), '..'))

from utils.data import load_data
from utils.model import load_model


# %% config
@dataclass
class DataTrainingArguments:
    data_path: str = field()
    max_length: int = 1024


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = ''
    model_type: str = 'auto'
    tokenizer_path: Optional[str] = ''
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )

    def __post_init__(self):
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_name_or_path


@dataclass
class MyTrainingArguments(TrainingArguments):
    project_name: str = ''

    def __post_init__(self):
        super().__post_init__()
        # wandb
        os.environ['WANDB_PROJECT'] = self.project_name
        self.run_name += f'-{time.time()}'


parser = ArgumentParser()
parser.add_argument('--config_path', '-c', type=str)
cli_args = parser.parse_args()

data_args: DataTrainingArguments
model_args: ModelArguments
training_args: TrainingArguments
hf_parser = HfArgumentParser([DataTrainingArguments, ModelArguments, TrainingArguments])
data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=cli_args.config_path)

pprint(data_args.__dict__)

# %% data
data = load_data(data_args.data_path)
print('data loaded')
print(data)

# %% model
tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
tokenizer.max_length = data_args.max_length

model = load_model(model_name_or_path=model_args.model_name_or_path, model_type=model_args.model_type)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %% optimizer
# if data_args.optim == 'adam_bf16':


# %% trainer


trainer = Trainer(
    args=training_args,
    model=model,
    data_collator=data_collator,
    train_dataset=data['train'],
    eval_dataset=data['test'],

)

# %% train

trainer.train()
trainer.save_model(output_dir=f'{training_args.output_dir}/best')
tokenizer.save_pretrained(f'{training_args.output_dir}/best')

"""
CONFIG=''
torchrun --nnodes 1 --nproc-per-node 1 tasks/train_opt_fast.py -c $CONFIG
"""
