import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from os import path as osp
from pprint import pprint
from typing import Optional

from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer
from transformers import HfArgumentParser
from transformers import TrainingArguments

sys.path.append(osp.join(osp.dirname(__file__), '..'))

from utils.data import load_data
from utils.model import load_model, load_tokenizer


# %% config
@dataclass
class DataTrainingArguments:
    data_path: str = ''
    max_length: int = 1024

    def __post_init__(self):
        assert self.data_path


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
        assert self.model_name_or_path
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_name_or_path


@dataclass
class MyTrainingArguments(TrainingArguments):
    project_name: str = ''
    # lora
    target_modules: list[str] = None

    def __post_init__(self):
        super().__post_init__()
        # wandb
        os.environ['WANDB_PROJECT'] = self.project_name
        self.run_name += f'-{time.time()}'


# %% config
parser = ArgumentParser()
parser.add_argument('--config_path', '-c', type=str)
cli_args = parser.parse_args()
print(cli_args)

data_args: DataTrainingArguments
model_args: ModelArguments
training_args: MyTrainingArguments
hf_parser = HfArgumentParser([DataTrainingArguments, ModelArguments, MyTrainingArguments])
data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=cli_args.config_path)

print(data_args)
print(model_args)
print(training_args)

pprint(data_args.__dict__)

# %% tokenizer
tokenizer = load_tokenizer(tokenizer_path=model_args.tokenizer_path, max_length=data_args.max_length)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %% data
data = load_data(path=data_args.data_path, tokenizer=tokenizer, max_length=data_args.max_length, test_size=0.1)
print('Data loaded')
print(data)
print(data['train'][0])
print(tokenizer.decode(data['train'][0]['input_ids']))

# %% model
model = load_model(model_name_or_path=model_args.model_name_or_path, model_type=model_args.model_type)
model.enable_input_require_grads()
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules=training_args.target_modules
    # bias="all"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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
torchrun --nnodes 1 --nproc-per-node 1 training/train_gpt_hf.py -c $CONFIG
"""
