from omegaconf import OmegaConf

from common.data import create_base_transforms, create_base_dataloader

from .ffpp import *
from .celeb_df import *

LOADER = {
    'FFPP_Dataset': FFPP_Dataset,
    'FFPP_Dataset_Preprocessed': FFPP_Dataset_Preprocessed,
    'FFPP_Dataset_Preprocessed_Multiple': FFPP_Dataset_Preprocessed_Multiple,
    'CelebDF_Dataset': CelebDF_Dataset,
}

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
        dataset_params['methods'] = [m for m in METHODS if m != args['method']]
        if 'method' in dataset_params.keys():
            dataset_params.pop('method')
        assert len(dataset_params['methods']) == 3
    else:
        dataset_params['method'] = args['method']
    dataset_params['compression'] = args['compression']

    _dataset = LOADER[dataset_cfg.name](**dataset_params)
    _dataloader = create_base_dataloader(args, _dataset, split=split)

    return _dataloader
