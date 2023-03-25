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
    attention = Attention(block_size=1024).cuda()

    torch.manual_seed(42)
    q, k, v = [torch.randn(bs, n_head, seq_len, head_dim, dtype=torch.bfloat16).cuda() for _ in range(3)]

    # warmup
    # print("warmup...")
    #
    # y1 = flash_attention(q, k, v)
    # y2 = attention(q, k, v)
    # print((y1 - y2).sum())

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


if __name__ == "__main__":
    attention_benchmark(bs=1, n_head=12, seq_len=1024, head_dim=64)
    attention_benchmark(bs=64, n_head=12, seq_len=1024, head_dim=64)
