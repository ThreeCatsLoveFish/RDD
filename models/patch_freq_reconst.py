import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.distribute_utils import is_main_process
from models.patch_freq import PatchFreq, patchfy_and_dct


class PatchFreqReconst(PatchFreq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconst = nn.Conv3d(192, 3 * 32**2, 1)

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        f = patchfy_and_dct(x, 2 * 2**self.inj_at)
        f = self.freq_stem(F.dropout3d(f, 0.9, inplace=True))
        for blk in self.blocks[:-1]:
            f = blk(f)
        aux_loss = F.mse_loss(self.reconst(f), patchfy_and_dct(x, 32, dct=False))
        return self.blocks[-1](f), aux_loss
