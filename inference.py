import argparse
import math
import os
import cv2
import torch
import numpy as np
from decord import VideoReader
from omegaconf import OmegaConf

import models
from models.experimental import attempt_load


def make_divisible(x, divisor=32):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


class Detector:

    def __init__(self, weights, device='cpu', img_size=640) -> None:
        if not os.path.exists(weights):
            print(f"Face detector weights not found. ({weights})\n"
                "You may need to download from "
                "https://drive.google.com/file/d/1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO/view?usp=sharing.\n"
                "Face detector repo: https://github.com/deepcam-cn/yolov5-face.")
            exit(1)
        self.device = device
        self.img_size = img_size
        self.model = attempt_load(weights, map_location=device)

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


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', type=str,
                        default='data/ffpp_videos/manipulated_sequences/Deepfakes/c40/videos/000_003.mp4')
    parser.add_argument('-c', '--config', type=str, default='configs/ffpp_stil.yaml')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--detector', type=str, default='data/yolov5s-face.pt')
    parser.add_argument('--resume', type=str,
                        default='exps/ffpp_stil/STIL_Model_Deepfakes_c40/ckpt/best.pth')
    parser.add_argument('--demo', default=False, action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    detector = Detector(args.detector, device)

    oc_cfg = OmegaConf.load(args.config)
    oc_cfg.merge_with(vars(args))
    args = oc_cfg

    # set model and wrap it with DistributedDataParallel
    model = models.__dict__[args.model.name](**args.model.params)
    state_dict = torch.load(args.resume, map_location='cpu')['state_dict']
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()})
    model.set_segment(args.n_frames)
    model.to(device)

    video = FaceVideo(args.input, detector, n_frames=args.n_frames)
    frames = video.load_cropped_frames()
    frames = frames.flatten(0, 1).to(device, non_blocking=True)

    real_prob = model(frames[None])[0].softmax(-1)[0].item()
    print('real prob:', real_prob)

    if args.demo:
        os.makedirs('figs', exist_ok=True)
        h, w = video.frames[0].shape[:2]
        tl = max(1, round(0.002 * (h + w) / 2) )  # line/font thickness
        for i, (frame, box) in enumerate(zip(video.frames, video.boxes)):
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, (x-w/2, y-h/2, x+w/2, y+h/2))
            label = 'Fake' if real_prob < 0.5 else 'Real'
            color = (0, 255, 0) if real_prob < 0.5 else (255, 0, 0)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
            img = cv2.putText(img, label, (x1, y1 - tl * 2), 0, tl, color, tl, cv2.LINE_AA)
            cv2.imwrite(f"figs/{i:03d}.png", img)
        print("Demo images are saved to ./figs.\n"
            "You may use the following command to greate a GIF:\n"
            "ffmpeg -r 2 -i figs/%03d.png figs/demo.gif")


if __name__ == "__main__":
    main()
