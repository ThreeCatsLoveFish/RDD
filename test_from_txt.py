import argparse
import math
import os
from pathlib import Path

import cv2
import torch
import numpy as np
from decord import VideoReader
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

import models

def load_model(url, map_location='cpu'):
    if url.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            url, map_location=map_location)
    else:
        checkpoint = torch.load(url, map_location=map_location)
    return checkpoint


def make_divisible(x, divisor=32):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


class Detector:

    def __init__(self, weights, device='cpu', img_size=640) -> None:
        self.device = device
        self.img_size = img_size
        self.model = load_model(weights, device)['model'].float().fuse().eval()

    def __call__(self, imgs):
        h0, w0 = imgs[0].shape[:2]
        r = self.img_size / max(h0, w0)  # resize image to img_size
        w, h = make_divisible(w0 * r), make_divisible(h0 * r)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            imgs = [cv2.resize(img, (w, h), interpolation=interp) for img in imgs]
        scale = torch.tensor([w0 / w, h0 / h] * 2)
        img = np.stack(imgs).transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device, non_blocking=True)
        img = img.float() / 255.

        # Inference
        pred = self.model(img)[0]

        # Process detections
        index = pred[..., 4].max(1).indices
        return pred[range(index.size(0)), index, :4].cpu().mul(scale).numpy()


class FaceVideo:

    def __init__(self, src, detector, n_frames=16, img_size=224) -> None:
        self.src = src
        self.n_frames = n_frames
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.detector = detector
        self.mean = np.float32([0.485, 0.456, 0.406]) * 255
        self.std = np.float32([0.229, 0.224, 0.225]) * 255
        self._frames = None
        self._boxes = None

    @property
    def frames(self):
        if self._frames is None:
            vr = VideoReader(self.src)
            sampled_idxs = np.linspace(0, len(vr) - 1, self.n_frames, dtype=int).tolist()
            self._frames = list(vr.get_batch(sampled_idxs).asnumpy())
        return self._frames
    
    @property
    def boxes(self):
        if self._boxes is None:
            self._boxes = self.detector(self.frames)
        return self._boxes

    def load_cropped_frames(self, margin=1.3):
        cx, cy = self.boxes[:, 0], self.boxes[:, 1]
        hw = self.boxes[:, 2:].max(-1) * margin
        rois = np.stack([cx - hw / 2, cy - hw /2, cx + hw / 2, cy + hw / 2], 1).clip(0)
        clip = []
        for frame, roi in zip(self.frames, rois.tolist()):
            x0, y0, x1, y1 = map(int, roi)
            clip.append(cv2.resize(frame[y0:y1, x0:x1], self.img_size, interpolation=cv2.INTER_LINEAR))
        clip = (np.float32(clip) - self.mean) / self.std
        clip = np.ascontiguousarray(clip.transpose(0, 3, 1, 2))
        return torch.from_numpy(clip)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('test_label_txt', type=str)
    parser.add_argument('-c', '--config', type=str, default='configs/ffpp_x3d_inference.yaml')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--detector', type=str,
                        default='https://github.com/zyayoung/oss/releases/download/rdd/yolov5n-0.5.pt')
    parser.add_argument('--resume', type=str,
                        default='https://github.com/zyayoung/oss/releases/download/rdd/ffpp_x3d.pth')
    parser.add_argument('--demo', default=False, action='store_true')
    args = parser.parse_args()

    # load video
    with open(args.test_label_txt) as f:
        imdb = [line.split() for line in f]

    # load model config
    oc_cfg = OmegaConf.load(args.config)
    oc_cfg.merge_with(vars(args))
    args = oc_cfg

    # load face detection model
    device = torch.device(args.device)
    print("Loading face detection model...")
    detector = Detector(args.detector, device)

    # load forgery detection model
    print("Loading forgery detection model...")
    model = models.__dict__[args.model.name](**args.model.params)
    state_dict = load_model(args.resume, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()})
    model.set_segment(args.n_frames)
    model.to(device).eval()

    print("Detecting...")
    base_dir = os.path.dirname(args.test_label_txt)
    gt = [label == '1' for label, _ in imdb]
    pred = []
    for label, path in tqdm(imdb):
        video = FaceVideo(os.path.join(base_dir, path), detector, n_frames=args.n_frames)
        frames = video.load_cropped_frames()
        frames = frames.flatten(0, 1).to(device, non_blocking=True)

        real_prob = model(frames[None])[0].softmax(-1)[0].item()
        pred.append(real_prob)

    acc = accuracy_score(gt, [p > 0.5 for p in pred])
    auc = roc_auc_score(gt, pred)
    print("Acc:", acc)
    print("Auc:", auc)


if __name__ == "__main__":
    main()
