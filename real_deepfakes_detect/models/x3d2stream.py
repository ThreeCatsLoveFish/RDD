import torch
import torch.nn as nn
from .dct import dct_3d
from .distribute_utils import is_main_process

class X3D_2Stream(nn.Module):
    def __init__(self, name='x3d_s', num_class=2):
        super().__init__()
        self.rgb_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        self.freq_model = torch.hub.load('facebookresearch/pytorchvideo',
            name, pretrained=is_main_process())
        fc_feature_dim = self.rgb_model.blocks[5].proj.in_features

        self.rgb_model.blocks[5].proj = nn.Identity()
        self.rgb_model.blocks[5].activation = nn.Identity()
        self.rgb_model.blocks[5].output_pool = None

        self.freq_model.blocks[5].proj = nn.Identity()
        self.freq_model.blocks[5].activation = nn.Identity()
        self.freq_model.blocks[5].output_pool = None

        self.fusion = nn.Sequential(
            nn.Linear(fc_feature_dim * 2, fc_feature_dim),
            nn.ReLU(),
            nn.Linear(fc_feature_dim, num_class),
        )
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        rgb = x.view(n, -1, 3, h, w).transpose(1, 2) # (N, C, T, H, W)
        
        # calculate DCT-3D on last 3 dimensions (spatial + temporal)
        freq = dct_3d(rgb, norm='ortho') 
        
        rgb_embd = self.rgb_model(rgb)
        freq_embd = self.freq_model(freq)
        embd = torch.cat((rgb_embd, freq_embd), dim=1)
        embd = embd.permute((0, 2, 3, 4, 1))
        embd = self.fusion(embd)
        embd = embd.permute((0, 4, 1, 2, 3))
        embd = self.output_pool(embd)
        embd = embd.view(embd.shape[0], -1)
        return embd

    def set_segment(self, _):
        pass