import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.distribute_utils import is_main_process
from common.utils.dct import dct
import torch.utils.checkpoint as cp
import matplotlib.pyplot as plt
import numpy as np


def dct_patch(x, norm='ortho'):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -3), norm=norm)
    return X2.transpose(-1, -3)


def patchfy_and_dct(x, p=8, dct=True):
    n, c, t, h, w = x.size()
    assert h % p == 0 and w % p == 0
    x = x.view(n, c, t, h // p, p, w // p, p)
    if dct:
        x = dct_patch(x)
    return x.permute(0, 1, 4, 6, 2, 3, 5).flatten(1, 3)


class CrossAttn(nn.Module):
    def __init__(self, C, r, thw):
        super().__init__()
        assert C % r == 0
        self.conv_key1 = nn.Conv3d(C, C // r, 1)
        self.conv_key2 = nn.Conv3d(C, C // r, 1)
        self.conv_val1 = nn.Conv3d(C, C, 1)
        self.conv_val2 = nn.Conv3d(C, C, 1)

        self.cor_weight1 = nn.Linear(thw, thw, bias=False)
        self.cor_weight2 = nn.Linear(thw, thw, bias=False)

    def forward(self, x1, x2):
        n, c, t, h, w = x1.shape
        k1 = self.conv_key1(x1) # n c/r t h w
        v1 = self.conv_val1(x1) # n c t h w

        k2 = self.conv_key2(x2) # n c/r t h w
        v2 = self.conv_val2(x2) # n c t h w

        cor = torch.bmm(k1.flatten(2).transpose(1, 2), k2.flatten(2)) # n thw thw
        attn1 = F.softmax(self.cor_weight1(cor), dim=2)
        attn2 = F.softmax(self.cor_weight2(cor.transpose(1, 2)), dim=2).transpose(1, 2)
        r1 = torch.bmm(v1.flatten(2), attn1).reshape((n, c, t, h, w))
        r2 = torch.bmm(v2.flatten(2), attn2.transpose(1, 2)).reshape((n, c, t, h, w))

        return x1 + r1, x2 + r2


class TwoStreamPatchFreq(nn.Module):
    def __init__(self, name='x3d_m', num_class=2, inj_at=3, attn_at=[3], r=12, pretrain=None, thw=16*14*14):
        super().__init__()
        x3d = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=pretrain is None and is_main_process())
        rgb_x3d = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        fc_feature_dim = x3d.blocks[5].proj.in_features
        x3d.blocks[5].proj = nn.Identity()
        x3d.blocks[5].activation = nn.Identity()
        x3d.blocks[5].output_pool = None
        rgb_x3d.blocks[5].proj = nn.Identity()
        rgb_x3d.blocks[5].activation = nn.Identity()
        rgb_x3d.blocks[5].output_pool = None
        self.fusion = nn.Sequential(
            nn.Linear(fc_feature_dim * 2, fc_feature_dim),
            nn.ReLU(),
            nn.Linear(fc_feature_dim, num_class),
        )
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.inj_at = inj_at
        self.attn_at = attn_at
        self.freq_stem = nn.Sequential(
            nn.Conv3d(12 * 4**inj_at, 12 * 2**inj_at, 1, bias=False),
            nn.BatchNorm3d(12 * 2**inj_at))
        self.blocks = x3d.blocks[inj_at:]
        self.blocks[0].res_blocks = self.blocks[0].res_blocks[1:]
        self.rgb_blocks = rgb_x3d.blocks
        cross_attn = {}
        for i in attn_at:
            cross_attn[str(i)] = CrossAttn(12 * 2**i, r, thw)
        self.cross_attn = nn.ModuleDict(cross_attn)
        if pretrain and is_main_process():
            state_dict = torch.load(pretrain, map_location='cpu')
            state_dict = {k: v for k, v in state_dict.items() if 'proj.' not in k}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        f = patchfy_and_dct(x, 2 * 2**self.inj_at)
        f = self.freq_stem(f)
        x.requires_grad_(True)
        for i , blk in enumerate(self.rgb_blocks):
            x = cp.checkpoint(blk, x)
            if i >= self.inj_at:
                f = self.blocks[i - self.inj_at](f)
            if i in self.attn_at:
                x, f = self.cross_attn[str(i)](x, f)
        output = torch.cat((x, f), dim=1)
        output = output.permute((0, 2, 3, 4, 1))
        output = self.fusion(output)
        output = output.permute((0, 4, 1, 2, 3))
        output = self.output_pool(output)
        output = output.view(output.shape[0], -1)
        return output

    def set_segment(self, _):
        pass


class TwoStreamPatchFreqDS(nn.Module):
    def __init__(self, name='x3d_m', num_class=2, inj_at=3, attn_at=[3], r=12, pretrain=None, thw=16*14*14):
        super().__init__()
        x3d = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=pretrain is None and is_main_process())
        rgb_x3d = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        fc_feature_dim = x3d.blocks[5].proj.in_features
        x3d.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        x3d.blocks[5].activation = nn.Identity()
        rgb_x3d.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        rgb_x3d.blocks[5].activation = nn.Identity()
        self.inj_at = inj_at
        self.attn_at = attn_at
        self.freq_stem = nn.Sequential(
            nn.Conv3d(12 * 4**inj_at, 12 * 2**inj_at, 1, bias=False),
            nn.BatchNorm3d(12 * 2**inj_at))
        self.blocks = x3d.blocks[inj_at:]
        self.blocks[0].res_blocks = self.blocks[0].res_blocks[1:]
        self.rgb_blocks = rgb_x3d.blocks
        cross_attn = {}
        for i in attn_at:
            cross_attn[str(i)] = CrossAttn(12 * 2**i, r, thw)
        self.cross_attn = nn.ModuleDict(cross_attn)
        if pretrain and is_main_process():
            state_dict = torch.load(pretrain, map_location='cpu')
            state_dict = {k: v for k, v in state_dict.items() if 'proj.' not in k}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        f = patchfy_and_dct(x, 2 * 2**self.inj_at)
        f = self.freq_stem(f)
        x.requires_grad_(True)
        for i , blk in enumerate(self.rgb_blocks):
            x = cp.checkpoint(blk, x)
            if i >= self.inj_at:
                f = self.blocks[i - self.inj_at](f)
            if i in self.attn_at:
                x, f = self.cross_attn[str(i)](x, f)
        return (x, f)

    def set_segment(self, _):
        pass
