import torch
import torch.nn as nn
from common.utils.distribute_utils import is_main_process
from common.utils.dct import dct_3d, dct


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


class X3D_FreqInj(nn.Module):
    def __init__(self, name='x3d_s', num_class=2):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        fc_feature_dim = self.base_model.blocks[5].proj.in_features
        self.base_model.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        self.base_model.blocks[5].activation = nn.Identity()
        self.inj_at = [3]
        self.inj = nn.ModuleList([
            nn.Sequential(nn.BatchNorm3d(3 * 4**i), nn.Conv3d(3 * 4**i, 6 * 2**i, 1))
            if i in self.inj_at else None for i in range(6)])

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        t = x.size(2)
        output = x
        for i, blk in enumerate(self.base_model.blocks):
            if i in self.inj_at:
                p = 2**i
                f = dct_3d(x.view(n, 3, t, h // p, p, w // p, p))
                f = f.permute(0, 1, 4, 6, 2, 3, 5).flatten(1, 3)
                output = output + self.inj[i](f)
            print(i, output.shape)
            output = blk(output)
        return output

    def set_segment(self, _):
        pass
