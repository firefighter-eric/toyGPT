import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from os import path as osp
from pprint import pprint

from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from transformers import HfArgumentParser
from transformers import TrainingArguments

sys.path.append(osp.join(osp.dirname(__file__), '..'))
from models import OPTForCausalLM


# %% config
@dataclass
class CustomTrainArguments:
    data_path: str = ''
    tokenizer_path: str = ''
    model_path: str = ''

    output_dir: str = ''
    num_train_epochs: int = 1
    total_batch_size: int = -1
    mini_batch_size: int = -1

    project_name: str = ''
    run_name: str = ''

    learning_rate: float = 3e-5

    torch_compile: bool = True

    def __post_init__(self):
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_path

        if self.mini_batch_size == -1:
            self.total_batch_size = self.mini_batch_size
        self.gradient_accumulation_steps = self.total_batch_size // self.mini_batch_size

        self.learning_rate = float(self.learning_rate)

        # wandb
        os.environ['WANDB_PROJECT'] = self.project_name
        self.run_name += f'-{time.time()}'


parser = ArgumentParser()
parser.add_argument('--config_path', '-c', type=str)
args = parser.parse_args()

hf_parser = HfArgumentParser([CustomTrainArguments])
custom_train_args, = hf_parser.parse_yaml_file(args.config_path)
custom_train_args: CustomTrainArguments

pprint(custom_train_args.__dict__)

# %% data
data = DatasetDict.load_from_disk(dataset_dict_path=custom_train_args.data_path)
print('data loaded')
print(data)

# %% model
tokenizer = AutoTokenizer.from_pretrained(custom_train_args.tokenizer_path)
model = OPTForCausalLM.from_pretrained(custom_train_args.model_path)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %% trainer
train_args = TrainingArguments(
    output_dir=custom_train_args.output_dir,
    overwrite_output_dir=False,

    # data
    per_device_train_batch_size=custom_train_args.mini_batch_size,
    per_device_eval_batch_size=custom_train_args.mini_batch_size,
    gradient_accumulation_steps=custom_train_args.gradient_accumulation_steps,
    dataloader_num_workers=2,

    # precision
    bf16=True,
    bf16_full_eval=True,

    # eval
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=100,
    save_steps=100,
    save_total_limit=1,
    metric_for_best_model='eval_loss',
    prediction_loss_only=True,

    # log
    logging_strategy='steps',
    logging_steps=10,
    logging_dir=custom_train_args.output_dir,
    logging_first_step=True,
    run_name=custom_train_args.run_name,
    report_to=['wandb'],

    # hyper-parameters
    learning_rate=custom_train_args.learning_rate,
    num_train_epochs=custom_train_args.num_train_epochs,
    warmup_steps=0,
    weight_decay=0.0,

    greater_is_better=False,

    # flags
    do_train=True,
    do_eval=True,
    do_predict=False,
    load_best_model_at_end=True,
    torch_compile=custom_train_args.torch_compile,
    seed=42,
)

trainer = Trainer(
    args=train_args,
    model=model,
    data_collator=data_collator,
    train_dataset=data['train'],
    eval_dataset=data['test'],

)

# %% train

trainer.train()

"""
torchrun --nnodes 1 --nproc-per-node 1 tasks/train_opt_fast.py -c $config_path
"""
