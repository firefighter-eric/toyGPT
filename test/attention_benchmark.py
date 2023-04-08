import math

import torch
from torch import nn
from torch.utils import benchmark


def flash_attention(q, k, v):
    # efficient attention using Flash Attention CUDA kernels
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                         attn_mask=None,
                                                         dropout_p=0.0,
                                                         is_causal=True)
    return y


def torch_attention(q, k, v):
    L, S = q.size(-2), k.size(-2)
    dropout_p = 0
    attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    attn_mask = attn_mask.float().masked_fill(~attn_mask, -float('inf')) if attn_mask.dtype == torch.bool else attn_mask
    attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p)
    return attn_weight @ v


class Attention(torch.nn.Module):
    def __init__(self, block_size, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        T = q.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        return y


def attention_benchmark(bs, n_head, seq_len, head_dim):
    print('benchmark')
    print(f'bs: {bs}, n_head: {n_head}, seq_len: {seq_len}, head_dim: {head_dim}')
    attention = Attention(block_size=seq_len).cuda()

    torch.manual_seed(42)
    q, k, v = [torch.randn(bs, n_head, seq_len, head_dim, dtype=torch.bfloat16).cuda() for _ in range(3)]

    timer1 = benchmark.Timer(
        stmt="flash_attention(q, k, v)",
        globals={'q': q, 'k': k, 'v': v, 'flash_attention': flash_attention}
    )

    print(timer1.timeit(100))

    timer2 = benchmark.Timer(
        stmt="attention(q, k, v)",
        globals={'q': q, 'k': k, 'v': v, 'attention': attention}
    )

    print(timer2.timeit(100))
    print('#' * 100)


def check_same_result(bs, n_head, seq_len, head_dim):
    print('check_same_result')
    torch.manual_seed(42)
    q, k, v = [torch.randn(bs, n_head, seq_len, head_dim, dtype=torch.bfloat16).cuda() for _ in range(3)]

    # flash attention
    y1 = flash_attention(q, k, v)
    # torch attention
    # y2 = torch_attention(q, k, v)
    # nanogpt attention
    attention = Attention(block_size=seq_len).cuda().to(torch.bfloat16)
    y3 = attention(q, k, v)

    print('error:', (y1 - y3).abs().sum())
    print(torch.allclose(y1, y3, atol=1e-3))


if __name__ == "__main__":
    # attention_benchmark(bs=1, n_head=12, seq_len=1024, head_dim=64)
    # attention_benchmark(bs=64, n_head=12, seq_len=1024, head_dim=64)
    # attention_benchmark(bs=1, n_head=32, seq_len=2048, head_dim=64)
    # attention_benchmark(bs=2, n_head=32, seq_len=2048, head_dim=64)
    bs, n_head, seq_len, head_dim = 1, 12, 1024, 64
    check_same_result(bs=1, n_head=12, seq_len=1024, head_dim=64)
