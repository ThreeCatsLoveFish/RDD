from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from real_deepfakes_detect.models import x3d_hub
from .distribute_utils import is_main_process
from .dct import dct


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
    def __init__(self, C, r):
        super().__init__()
        assert C % r == 0
        self.conv_key1 = nn.Conv3d(C, C // r, 1)
        self.conv_key2 = nn.Conv3d(C, C // r, 1)
        self.conv_val1 = nn.Conv3d(C, C, 1)
        self.conv_val2 = nn.Conv3d(C, C, 1)

    def forward(self, x1, x2):
        n, c, t, h, w = x1.shape
        k1 = self.conv_key1(x1) # n c/r t h w
        v1 = self.conv_val1(x1) # n c t h w

        k2 = self.conv_key2(x2) # n c/r t h w
        v2 = self.conv_val2(x2) # n c t h w

        cor = torch.bmm(k1.flatten(2).transpose(1, 2), k2.flatten(2)) # n thw thw
        attn1 = F.softmax(cor, dim=2)
        attn2 = F.softmax(cor, dim=1)
        r1 = torch.bmm(v1.flatten(2), attn1).reshape((n, c, t, h, w))
        r2 = torch.bmm(v2.flatten(2), attn2.transpose(1, 2)).reshape((n, c, t, h, w))

        return x1 + r1, x2 + r2


class TwoStreamPatchFreq(nn.Module):
    def __init__(self, name='x3d_m', num_class=2, inj_at=3, deep_supervision=False, mlp=True):
        super().__init__()
        x3d = x3d_hub.__dict__[name](pretrained=is_main_process())
        rgb_x3d = x3d_hub.__dict__[name](pretrained=is_main_process())
        fc_feature_dim = x3d.blocks[5].proj.in_features
        x3d.blocks[5].proj = nn.Identity()
        x3d.blocks[5].activation = nn.Identity()
        x3d.blocks[5].output_pool = None
        rgb_x3d.blocks[5].proj = nn.Identity()
        rgb_x3d.blocks[5].activation = nn.Identity()
        rgb_x3d.blocks[5].output_pool = None
        if mlp:
            self.fusion = nn.Sequential(
                nn.Linear(fc_feature_dim * 2, fc_feature_dim),
                nn.ReLU(),
                nn.Linear(fc_feature_dim, num_class),
            )
        else:
            self.fusion = nn.Linear(fc_feature_dim * 2, num_class)
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.inj_at = inj_at
        self.freq_stem = nn.Sequential(
            nn.Conv3d(12 * 4**inj_at, 12 * 2**inj_at, 1, bias=False),
            nn.BatchNorm3d(12 * 2**inj_at), nn.ReLU(inplace=True))
        self.blocks = x3d.blocks[inj_at:]
        self.blocks[0].res_blocks = self.blocks[0].res_blocks[1:]
        self.rgb_blocks = rgb_x3d.blocks
        self.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        f = patchfy_and_dct(x, 2 * 2**self.inj_at)
        f = self.freq_stem(f)
        for i, blk in enumerate(self.rgb_blocks):
            x = blk(x)
            if i >= self.inj_at:
                f = self.blocks[i - self.inj_at](f)
            # if i >= self.inj_at and i < len(self.rgb_blocks) - 1:
            #     x, f = self.cross_attn[i - self.inj_at](x, f)
        output = torch.cat((x, f), dim=1)
        if self.training and self.deep_supervision:
            output = torch.cat([
                output,
                torch.cat((x.mul(2), torch.zeros_like(f)), dim=1),
                torch.cat((torch.zeros_like(x), f.mul(2)), dim=1),
            ])
        output = output.permute((0, 2, 3, 4, 1))
        output = self.fusion(output)
        output = output.permute((0, 4, 1, 2, 3))
        output = self.output_pool(output)
        output = output.view(n, -1)
        if self.training and self.deep_supervision:
            output = output.reshape(3, n, -1)
        return output

    def set_segment(self, _):
        pass
