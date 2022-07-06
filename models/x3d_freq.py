from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from einops import rearrange
from common.utils.dct import dct_3d, idct_3d
from common.utils.distribute_utils import is_main_process

@torch.no_grad()
def plot(x: torch.Tensor):
    mean = 0.45
    std = 0.225
    if x.ndim == 3:
        x = rearrange(x * std + mean, 'c h w -> h w c')
        x.clamp_(0, 1)
    plt.imsave("demo.png", x.cpu().numpy())


class X3D_Freq(nn.Module):
    def __init__(self, name='x3d_s', num_class=2, fad=True):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        fc_feature_dim = self.base_model.blocks[5].proj.in_features
        self.base_model.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        self.base_model.blocks[5].activation = nn.Identity()

        self.fad = fad
        if self.fad:
            conv0: nn.Conv3d = self.base_model.blocks[0].conv.conv_t
            w = conv0.weight.data.repeat_interleave(5, 1)
            new_conv0 = nn.Conv3d(3 * 5, 24, (1, 3, 3), (1, 2, 2), (0, 1, 1), bias=False)
            new_conv0.load_state_dict({'weight': w})
            self.base_model.blocks[0].conv.conv_t = new_conv0

            mask_cum = torch.zeros((16, 224, 224), dtype=bool)
            masks = []
            for i in range(5):
                mask_last = mask_cum.clone()
                to = 2 ** i
                mask_cum[:to, :to*14, :to*14] = 1
                masks.append(mask_cum & ~mask_last)
            self.fad_masks = nn.Parameter(torch.stack(masks), requires_grad=False)

    def forward(self, x):
        x = rearrange(x, 'n (t c) h w -> n c t h w', c=3)
        if self.fad:
            f = dct_3d(x)
            x_fad = idct_3d(f[:, :, None] * self.fad_masks)
            x = rearrange(x_fad, 'n c f t h w -> n (c f) t h w')
        return self.base_model(x)

    def set_segment(self, _):
        pass
