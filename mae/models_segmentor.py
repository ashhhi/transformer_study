# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
        '''
        Add a segmenter decoder
        '''
        # nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dec_cls = nn.Parameter(torch.zeros(1, 3, kwargs['embed_dim']))
        self.upsample = nn.Sequential(
            nn.Upsample(self.img_size // 8),
            nn.Upsample(self.img_size // 4),
            nn.Upsample(self.img_size // 2),
            nn.Upsample(self.img_size)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        # print('forward里的输入形状：', x.shape, 'Dtype：', x.dtype)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # print('forward里的embed+cls+pos形状：', x.shape)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            # outcome = x[:, 0]
            x = x[:, 1:, :]
            # 定义连续的反卷积层
            dec_cls = self.dec_cls.expand(B, 3, -1)
            x = torch.cat((dec_cls, x), dim=1)
            for blk in self.blocks:
                x = blk(x)
            x = torch.bmm(x[:, 3:, :], torch.transpose(x[:, :3, :], 1, 2))  # Batch, n_Token, n_Class
            x = x.reshape((-1, C, H//self.patch_size, W//self.patch_size))
            x = self.upsample(x)
            outcome = F.softmax(x, dim=1)
            # print('forward里的output形状：', x.shape)
        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    model = vit_base_patch16(img_size=224)
    tensor = torch.rand((2, 3, 224, 224))
    res = model(tensor)
    print(res.shape)