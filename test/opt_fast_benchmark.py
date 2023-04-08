import torch
from models import OPTForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_same_result():
    model_path = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    opt_model_hf = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()
    opt_model_fast = OPTForCausalLM.from_pretrained(model_path).cuda().eval()

    print('model loaded')
    input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt").cuda()
    with torch.no_grad():
        hf_outputs = opt_model_hf(input_ids)[0].max(dim=-1)
        fast_outputs = opt_model_fast(input_ids)[0].max(dim=-1)
    # assert torch.allclose(hf_outputs[0], fast_outputs[0])
    return hf_outputs[0].cpu().numpy(), fast_outputs[0].cpu().numpy()


hf, fast = check_same_result()
