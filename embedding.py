import torch
from torch import nn
import torch.nn.functional as f
import math

# 将输入的词汇表索引准换为指定纬度的embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False     # 编码不需要梯度
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocal_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocal_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        print(tok_emb, '\n', pos_emb)
        return self.drop_out(tok_emb + pos_emb)


if __name__ == '__main__':
    embedding = TransformerEmbedding(20, 512, 15, 0.1, 'cpu')
    embedding(torch.tensor([[1, 2, 3], [3,4,5]]))