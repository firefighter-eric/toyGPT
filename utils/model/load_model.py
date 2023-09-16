import os

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def load_model(model_name_or_path, model_type, ) -> PreTrainedModel:
    args = {'device_map': 'auto'}
    if True:
        args['load_in_8bit'] = True

    if model_type == 'auto':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **args)
    elif model_type == 'flash_opt':
        from models import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model_name_or_path, **args)
    elif model_type == 'flash_llama':
        from models.llama.modeling_llama_flash import load_llama_flash_class
        LlamaForCausalLM = load_llama_flash_class()

        model = LlamaForCausalLM.from_pretrained(model_name_or_path, **args)
    else:
        raise ValueError(f'Unknown model class name: {model_name_or_path}')
    return model


def load_tokenizer(tokenizer_path, max_length=1024) -> PreTrainedTokenizer:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.max_length = max_length

    tokenizer.padding_side = 'right'
    if 'llama' in tokenizer_path or 'alpaca' in tokenizer_path:
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
    elif 'opt' in tokenizer_path:
        tokenizer.pad_token_id = 1
    return tokenizer
