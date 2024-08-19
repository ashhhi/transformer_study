from torch import nn
from attention import MultiHeadAttention
from layernorm import LayerNorm
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, hidden=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.norm1(x)
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x + _x)

        _x = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x+_x)

        return x

class Encoder(nn.Module):
    def __init__(self, patch_size, n_head, n_layer):
        super(Encoder, self).__init__()
        d_model = patch_size * patch_size * 3
        ffn_hidden = d_model * 4
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head) for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask=None):
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
