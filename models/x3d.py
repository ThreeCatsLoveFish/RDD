import torch
import torch.nn as nn
from common.utils.distribute_utils import is_main_process


class X3D(nn.Module):
    def __init__(self, name='x3d_s', num_class=2):
        super().__init__()
        self.base_model = torch.hub.load('/home/zhouzhanhui/.cache/torch/hub/facebookresearch_pytorchvideo_main', 
            name, source='local', pretrained=is_main_process())
        fc_feature_dim = self.base_model.blocks[5].proj.in_features
        self.base_model.blocks[5].proj = nn.Linear(fc_feature_dim, num_class)
        self.base_model.blocks[5].activation = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        return self.base_model(x.view(n, -1, 3, h, w).transpose(1, 2))

    def set_segment(self, _):
        pass


if __name__ == '__main__':
    x3d = X3D()
