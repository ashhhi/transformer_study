import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head                        # 原论文中的 h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, n_head)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        n_d = self.d_model // self.n_head                   # 原论文中的 d_k
        q = q.view(batch, self.n_head, time, n_d)
        k = k.view(batch, self.n_head, time, n_d)
        v = v.view(batch, self.n_head, time, n_d)
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        return score


if __name__ == '__main__':
    x = torch.rand(128, 32, 512)
    d_model = 512
    n_head = 8

    attention = MultiHeadAttention(d_model, n_head)
    out = attention(x, x, x)
    print(out)