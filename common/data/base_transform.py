import numpy as np
import torch
import torchvision.transforms as T
import cv2
import augly.image as imaugs
import PIL






def create_base_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """

    if split == 'train':
        COLOR_JITTER_PARAMS = {
            "brightness_factor": 1.2,
            "contrast_factor": 1.2,
            "saturation_factor": 1.4,
        }
        base_transform = [
            imaugs.Resize(args.image_size, args.image_size, resample=PIL.Image.BILINEAR),    
            # imaugs.HFlip(p=0.5),
            # imaugs.VFlip(p=0.5),
            # imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
            T.ToTensor()
        ]
    else:
        base_transform = [
            imaugs.Resize(args.image_size, args.image_size, resample=PIL.Image.BILINEAR),
            T.ToTensor()
        ]

    return base_transform
