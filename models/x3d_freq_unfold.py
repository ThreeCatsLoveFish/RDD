import torch
import torch.nn as nn
from common.utils.distribute_utils import is_main_process
from common.utils.dct import dct_3d


class X3D_FreqUnfold(nn.Module):
    def __init__(self, name='x3d_xs', num_class=2):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        self.base_model.blocks[0] = nn.Identity()
        fc_feature_dim = self.base_model.blocks[5].proj.in_features
        self.base_model.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        self.base_model.blocks[5].activation = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.view(n, -1, 3, h, w).transpose(1, 2)  # n 3 t h w
        f = dct_3d(x, norm='ortho')
        f = f.view(n, 3, -1, 2, h // 2, 2, w // 2, 2)
        f = f.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return self.base_model(f)

    def set_segment(self, _):
        pass
