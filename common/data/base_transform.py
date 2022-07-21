import numpy as np
import torch
import torchvision.transforms as T
import cv2
import augly.image as imaugs
import PIL
import random


COLOR_JITTER_PARAMS = {
            "brightness": 1.2,
            "contrast": 1.2,
            "saturation": 1.1,
        }


def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    return cv2.resize(img, dsize=(width, height), interpolation=interpolation)


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


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return np.array([np.flipud(i) for i in img])
        return img


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return np.array([np.fliplr(i) for i in img])
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


class ColorJitter:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.applier = T.ColorJitter(**COLOR_JITTER_PARAMS)

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return np.array([np.array(self.applier(PIL.Image.fromarray(im))) for im in img])
        return img


class GaussianBlur:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.applier = T.GaussianBlur(kernel_size=(4,4), sigma=(0.1,1))

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return np.array([np.array(self.applier(PIL.Image.fromarray(im))) for im in img])
        return img



class RandomSwitch:
    def __init__(self,p=0.5,split=16):
        super().__init__()
        self.p = p
        self.split = split

    def __call__(self,img):
        if torch.rand(1) < self.p:
            F, H, W, C = img.shape
            img = np.transpose(img,axes=[1,2,3,0])
            img = np.array(np.split(img,self.split,axis=0))
            img = np.array(np.split(img,self.split,axis=2))
            img = img.reshape((-1,H//self.split,W//self.split,C,F))
            np.random.shuffle(img)
            img = img.reshape((self.split,self.split,H//self.split,W//self.split,C,F))
            img = np.transpose(img,axes=[0,2,1,3,4,5])
            img = img.reshape((H,W,C,F))
            img = np.transpose(img,axes=[3,0,1,2])
        return img



class Resize(T.Resize):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        super().__init__(size, interpolation)

    def forward(self, img):
        return np.array([resize(i, *self.size, self.interpolation) for i in img])


class ToTensor:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32) * 255

    def to_tensor(self, img):
        img = (np.float32(img) - self.mean) / self.std
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img)

    def __call__(self, img):
        return [self.to_tensor(i) for i in img]


def create_base_transforms(args, split='train', aug=None):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    # print(args.augmentation)
    if split == 'train':
        transform = [Resize((args.image_size, args.image_size))]
        if not aug is None:
            if "CJHSV" in aug:
                transform.append(random.choice([ColorJitter(),AugmentHSV()]))
            if "VF" in aug:
                transform.append(RandomVerticalFlip())
            if "HF" in aug:
                transform.append(RandomHorizontalFlip())
            if "TF" in aug:
                transform.append(RandomTemporalFlip())
            # if "BlUR" in aug:
            #     transform.append(GaussianBlur())
            if "RS" in aug:
                transform.append(RandomSwitch())
        transform.append(ToTensor(mean=args.mean, std=args.std))
        base_transform = T.Compose(transform)
    else:
        base_transform = T.Compose([
            Resize((args.image_size, args.image_size)),
            ToTensor(mean=args.mean, std=args.std)
        ])

    return base_transform
