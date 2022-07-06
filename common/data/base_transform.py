import numpy as np
import torch
import torchvision.transforms as T
import cv2


def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    return cv2.resize(img, dsize=(width, height))


def augment_hsv(imgs, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    for img in imgs:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2RGB, dst=img)  # no return needed


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return [i[:, ::-1] for i in img]
        return img


class RandomTemporalFlip:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return img[::-1]
        return img


class AugmentHSV:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            augment_hsv(img)
        return img


class Resize(T.Resize):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        super().__init__(size, interpolation)

    def forward(self, img):
        return [resize(i, *self.size, self.interpolation) for i in img]


class ToTensor:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32) * 255
        self.std = np.array(std, dtype=np.float32) * 255

    def to_tensor(self, img):
        img = (np.float32(img) - self.mean) / self.std
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img)

    def __call__(self, img):
        return [self.to_tensor(i) for i in img]


def create_base_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """

    if split == 'train':
        base_transform = T.Compose([
            Resize((args.image_size, args.image_size)),
            AugmentHSV(),
            RandomTemporalFlip(),
            RandomHorizontalFlip(),
            ToTensor(mean=args.mean, std=args.std)
        ])

    else:
        base_transform = T.Compose([
            Resize((args.image_size, args.image_size)),
            ToTensor(mean=args.mean, std=args.std)
        ])

    return base_transform
