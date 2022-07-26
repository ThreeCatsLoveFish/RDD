import argparse
import math
import os

import cv2
import torch
import numpy as np
from decord import VideoReader
from omegaconf import OmegaConf
from .face_recog import Detector, load_model

from . import models


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
    basedir = os.path.dirname(__file__)
    parser.add_argument('-i', '--input', type=str, default='test.mp4')
    parser.add_argument('-c', '--config', type=str, default=basedir+'/configs/ffpp_2s_inference.yaml')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--detector', type=str,
                        default='https://github.com/zyayoung/oss/releases/download/rdd/yolov5s-face.pt')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--demo', default=False, action='store_true')
    args = parser.parse_args()

    oc_cfg = OmegaConf.load(args.config)
    oc_cfg.merge_with(vars(args))
    args = oc_cfg

    # load face detection model
    device = torch.device(args.device)
    print("Loading face detection model...", end=' ', flush=True)
    detector = Detector(args.detector, device)
    print("Done")

    # load forgery detection model
    print("Loading forgery detection model...", end=' ', flush=True)
    model = models.__dict__[args.model.name](**args.model.params)
    state_dict = load_model(args.resume or args.model.resume, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()})
    model.set_segment(args.n_frames)
    model.to(device).eval()
    print("Done")

    # load video
    print("Detecting...", end=' ', flush=True)
    video = FaceVideo(args.input, detector, n_frames=args.n_frames)
    frames = video.load_cropped_frames()
    frames = frames.flatten(0, 1).to(device, non_blocking=True)

    real_prob = model(frames[None])[0].softmax(-1)[0].item()
    print("Done")

    label = 'Fake' if real_prob < 0.5 else 'Real'
    confidence = 1 - real_prob if real_prob < 0.5 else real_prob
    print(f'Result: {label}; Confidence: {confidence:.2f}')

    if args.demo:
        print("Saving results to figs/")
        os.makedirs('figs', exist_ok=True)
        h, w = video.frames[0].shape[:2]
        tl = max(1, round(0.002 * (h + w)) )  # line/font thickness
        for i, (frame, box) in enumerate(zip(video.frames, video.boxes)):
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, (x-w/2, y-h/2, x+w/2, y+h/2))
            color = (0, 0, 255) if real_prob < 0.5 else (0, 255, 0)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
            img = cv2.putText(img, label, (x1, y1 - tl * 2), 0, tl, color, tl, cv2.LINE_AA)
            cv2.imwrite(f"figs/{i:03d}.png", img)

if __name__ == "__main__":
    main()
