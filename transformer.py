import torch
from torch import nn
from decoder import Decoder
from encoder import Encoder


d_model = 512


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 drop_prob,
                 device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, device, drop_prob)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, device, drop_prob)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(3)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        mask = torch.trill(torch.ones(len(q), len(k)).type(torch.BoolTensor).to(self.device))
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, src, trg_mask, src_mask)
        return out

if __name__ == '__main__':
    Transformer(src_pad_idx=1,
                trg_pad_idx=1,
                enc_voc_size=100,
                dec_voc_size=100,
                d_model=512,
                max_len=20,
                n_heads=8,
                ffn_hidden=2048,
                n_layers=6,
                drop_prob=0.1,
                device='cpu')