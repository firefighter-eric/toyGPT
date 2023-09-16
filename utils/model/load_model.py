from transformers import PreTrainedModel, AutoModelForCausalLM


def load_model(model_name_or_path, model_type) -> PreTrainedModel:
    if model_type == 'auto':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    elif model_type == 'flash opt':
        from models import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f'Unknown model class name: {model_name_or_path}')
    return model