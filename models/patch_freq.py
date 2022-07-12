import torch
import torch.nn as nn
from common.utils.distribute_utils import is_main_process
from common.utils.dct import dct


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


class PatchFreq(nn.Module):
    def __init__(self, name='x3d_s', num_class=2, inj_at=2, pretrain=None):
        super().__init__()
        x3d = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=pretrain is None and is_main_process())
        fc_feature_dim = x3d.blocks[5].proj.in_features
        x3d.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        x3d.blocks[5].activation = nn.Identity()
        self.inj_at = inj_at
        self.freq_stem = nn.Sequential(
            nn.Conv3d(12 * 4**inj_at, 12 * 2**inj_at, 1, bias=False),
            nn.BatchNorm3d(12 * 2**inj_at), nn.ReLU(inplace=True))
        self.blocks = x3d.blocks[inj_at:]
        self.blocks[0].res_blocks = self.blocks[0].res_blocks[1:]
        if pretrain and is_main_process():
            state_dict = torch.load(pretrain, map_location='cpu')
            # state_dict = {k: v for k, v in state_dict.items() if 'proj.' not in k}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        f = patchfy_and_dct(x, 2 * 2**self.inj_at)
        f = self.freq_stem(f)
        for blk in self.blocks:
            f = blk(f)
        return f

    def set_segment(self, _):
        pass
