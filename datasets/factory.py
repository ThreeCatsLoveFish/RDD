import sys
import os
from omegaconf import OmegaConf

from common.data import create_base_transforms, create_base_dataloader

from . import video_dataset

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

def get_dataloader(args, split):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
    """
    transform = create_base_transforms(args.transform_params, split=split)

    dataset_cfg = getattr(args, split).dataset
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    dataset_params['transform'] = transform
    if dataset_cfg.name == 'FFPP_Dataset_Preprocessed_Multiple':
        dataset_params['methods'] = [m for m in METHODS if m is not args['method']]
    else:
        dataset_params['method'] = args['method']
    dataset_params['compression'] = args['compression']

    _dataset = video_dataset.__dict__[dataset_cfg.name](**dataset_params)
    _dataloader = create_base_dataloader(args, _dataset, split=split)

    return _dataloader
