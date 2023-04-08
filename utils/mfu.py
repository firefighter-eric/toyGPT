from transformers import OPTConfig, OPTForCausalLM

device_flops_map = {
    '3090': 142e12,
    '4090': 330e12,
    'V100 PCIe': 112e12,
    'V100 SXM': 125e12,
    'A100': 312e12,
}


def estimate_mfu(
        n_params,
        n_layer,
        n_head,
        n_embd,
        block_size,
        fwdbwd_per_iter,
        device_name,
        dt
) -> tuple[float, float, float]:
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    L, H, Q, T = n_layer, n_head, n_embd // n_head, block_size
    flops_per_token = 6 * n_params + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second

    flops_promised = device_flops_map[device_name]
    mfu = flops_achieved / flops_promised
    return mfu, flops_achieved, flops_promised


def opt_mfu(model, block_size, fb_per_iter, device, dt, n_params: float = -1):
    config = OPTConfig.from_pretrained(model)
    if n_params < 0:
        model = OPTForCausalLM.from_pretrained(model)
        n_params = sum(p.numel() for p in model.parameters())
        # n_params -= sum(p.numel() for p in model.model.decoder.embed_tokens.parameters())
    n_layer = config.num_hidden_layers
    n_head = config.num_attention_heads
    n_embd = config.hidden_size
    mfu, flops_achieved, flops_promised = estimate_mfu(n_params, n_layer, n_head, n_embd, block_size, fb_per_iter,
                                                       device, dt)
    print('mfu:', mfu, '\t',
          'flops achieved:', flops_achieved / 1e12, 'TFLOPS', '\t',
          'flops promised:', flops_promised / 1e12, 'TFLOPS')
    return mfu, flops_achieved, flops_promised


if __name__ == '__main__':
    opt_mfu(model='facebook/opt-125m', block_size=512, fb_per_iter=32, device='4090', dt=0.3, n_params=125e6)
    opt_mfu(model='facebook/opt-125m', block_size=1024, fb_per_iter=32, device='4090', dt=0.668, n_params=125e6)
    opt_mfu(model='facebook/opt-1.3b', block_size=1024, fb_per_iter=512 / 8, device='V100 PCIe', dt=30, n_params=1.3e9)
    opt_mfu(model='facebook/opt-125m', block_size=1024, fb_per_iter=32, device='3090', dt=0.837, n_params=125e6)
    opt_mfu(model='facebook/opt-125m', block_size=1024, fb_per_iter=32, device='3090', dt=0.641, n_params=125e6)
    opt_mfu(model='facebook/opt-350m', block_size=1024, fb_per_iter=32, device='3090', dt=1.57, n_params=350e6)
