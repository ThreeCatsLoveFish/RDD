# -*- coding: UTF-8 -*-
import argparse
import json
import pickle
import math
import os
import subprocess
import sys

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .utils.datasets import letterbox
from .utils.general import check_img_size, xywh2xyxy, scale_coords, xyxy2xywh
from glob import glob
from decord import VideoReader


def load_model(url, map_location='cpu'):
    sys.path.append(os.path.dirname(__file__))
    if url.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            url, map_location=map_location)
    else:
        checkpoint = torch.load(url, map_location=map_location)
    return checkpoint


class FFPPDataset(Dataset):
    def __init__(self) -> None:
        vid_list = sorted(glob("data/ffpp_videos/*/*/c40/videos/*.mp4"))
        rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
        total = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
        self.vid_list = vid_list[rank::total]
        self.img_size = 800

    def __len__(self):
        return len(self.vid_list)
    
    def __getitem__(self, index):
        path = self.vid_list[index]
        vr = VideoReader(path)
        imgs = [i.asnumpy() for i in vr]
        imgsz = check_img_size(self.img_size, s=32)  # check img_size
        clip = []
        for img0 in imgs:
            h0, w0 = img0.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            img = letterbox(img0, new_shape=imgsz)[0]
            clip.append(img.transpose(2, 0, 1))

        # Run inference
        clip = np.ascontiguousarray(np.stack(clip))
        img = torch.from_numpy(clip)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img, imgs[0].shape, path


class Detector:

    def __init__(self, weights, device='cpu', img_size=800) -> None:
        self.device = device
        self.img_size = img_size
        self.model = load_model(weights, device)['model'].float().fuse().eval()
    
    def detect_preprocessed(self, img, orig_shape):
        pred = self.model(img)[0]
        index = pred[..., 4].max(1).indices
        dets = pred[range(index.size(0)), index, :4]
        dets = scale_coords(img.shape[2:], xywh2xyxy(dets), orig_shape)
        return xyxy2xywh(dets).cpu().numpy()

    def __call__(self, imgs):
        imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        clip = []
        for img0 in imgs:
            h0, w0 = img0.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            img = letterbox(img0, new_shape=imgsz)[0]
            clip.append(img.transpose(2, 0, 1))

        # Run inference
        clip = np.ascontiguousarray(np.stack(clip))
        img = torch.from_numpy(clip).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return self.detect_preprocessed(img, imgs[0].shape)


@torch.no_grad()
def detect_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str,
                        default='https://github.com/zyayoung/oss/releases/download/rdd/yolov5n-0.5.pt')
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = Detector(opt.detector, device)

    dataloader = DataLoader(FFPPDataset(), 1,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda b: b[0])
    for img, orig_size, path in tqdm(dataloader):
        boxes = []
        while img.size(0):
            for res in detector.detect_preprocessed(img[:64].to(device, non_blocking=True), orig_size):
                boxes.append(res)
            img = img[64:]
        with open(path[:-4] + '.txt', 'w') as f:
            for b in boxes:
                f.write('{:.1f} {:.1f} {:.1f} {:.1f}\n'.format(*b))


if __name__ == '__main__':
    detect_all()
