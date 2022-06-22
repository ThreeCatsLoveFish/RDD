import torch
import torch.nn as nn
import torch_dct as dct
from common.utils.distribute_utils import is_main_process

class X3D_2Stream(nn.Module):
    def __init__(self, name='x3d_s', num_class=2):
        super().__init__()
        self.rgb_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        self.freq_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        fc_feature_dim = self.base_model.blocks[5].proj.in_features

        self.rgb_model.blocks[5].proj = nn.Identity()
        self.rgb_model.blocks[5].activation = nn.Identity()
        self.rgb_model.blocks[5].output_pool = nn.Identity()

        self.freq_model.blocks[5].proj = nn.Identity()
        self.freq_model.blocks[5].activation = nn.Identity()
        self.freq_model.blocks[5].output_pool = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(fc_feature_dim * 2, fc_feature_dim),
            nn.ReLU(),
            nn.Linear(fc_feature_dim, num_class),
            nn.AdaptiveAvgPool3d(output_size=1)
        )

    def forward(self, x):
        n, _, h, w = x.shape
        rgb = x.view(n, -1, 3, h, w).transpose(1, 2) # (N, C, T, H, W)
        
        # calculate DCT-3D on last 3 dimensions (spatial + temporal)
        freq = dct.dct_3d(rgb) 
        
        rgb_embd = self.rgb_model(rgb)
        freq_embd = self.freq_model(freq)
        embd = torch.cat((rgb_embd, freq_embd), dim=1)
        return self.fusion(embd)

    def set_segment(self, _):
        pass