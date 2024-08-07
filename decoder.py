from torch import nn
from attention import MultiHeadAttention
from embedding import TransformerEmbedding
from layernorm import LayerNorm
from encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        _x = x

        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, drop_prob):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head) for _ in range(n_layer)
            ]
        )
        self.fc1 = nn.Linear(d_model, dec_voc_size)

    def forward(self, enc, t_mask, s_mask):
        dec = self.embedding(enc)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc1(dec)
        return dec

