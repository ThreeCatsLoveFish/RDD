import argparse
from glob import glob
import math
import os
import subprocess

import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
from decord import VideoReader
from omegaconf import OmegaConf

from pytorch_grad_cam import ActivationsAndGradients

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

    def crop(self, margin=1.3):
        cx, cy = self.boxes[:, 0], self.boxes[:, 1]
        hw = self.boxes[:, 2:].max(-1) * margin
        rois = np.stack([cx - hw / 2, cy - hw /2, cx + hw / 2, cy + hw / 2], 1).clip(0)
        clip = []
        for frame, roi in zip(self.frames, rois.tolist()):
            x0, y0, x1, y1 = map(int, roi)
            clip.append(cv2.resize(frame[y0:y1, x0:x1], self.img_size, interpolation=cv2.INTER_LINEAR))
        return clip

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


class VideoWriter:
    def __init__(self, filename, fps=24) -> None:
        self.filename = filename
        if self.filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fps = fps
        self.p = None
    
    def write(self, frame):
        if not self.filename:
            return
        if self.p is None:
            h, w, _ = frame.shape
            self.p = subprocess.Popen([
                "ffmpeg",
                '-y',  # overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', f'{w}x{h}',  # size of one frame
                '-pix_fmt', 'bgr24',
                '-r', f'{self.fps}',  # frames per second
                '-i', '-',  # The imput comes from a pipe
                '-s', f'{w}x{h}',
                '-an',  # Tells FFMPEG not to expect any audio
                '-loglevel', 'error',
                '-b:v', '800k',
                '-pix_fmt', 'yuv420p',
                self.filename
            ], stdin=subprocess.PIPE)
        self.p.stdin.write(frame.tobytes())

    def close(self):
        if self.p:
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', type=str,
                        default='data/ffpp_videos/manipulated_sequences/Deepfakes/c40/videos/000_003.mp4')
    parser.add_argument('-c', '--config', type=str, default='configs/ffpp_x3d_inference.yaml')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--detector', type=str,
                        default='https://github.com/zyayoung/oss/releases/download/rdd/yolov5s-face.pt')
    parser.add_argument('--resume', type=str,
                        default='https://github.com/zyayoung/oss/releases/download/rdd/ffpp_x3d.pth')
    args = parser.parse_args()

    oc_cfg = OmegaConf.load(args.config)
    oc_cfg.merge_with(vars(args))
    args = oc_cfg

    device = torch.device(args.device)
    print("Loading face detection model...", end=' ', flush=True)
    detector = Detector(args.detector, device)
    print("Done")

    print("Loading fogery detection model...", end=' ', flush=True)
    model = models.__dict__[args.model.name](**args.model.params)
    state_dict = load_model(args.resume, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()})
    model.set_segment(args.n_frames)
    model.to(device).eval()
    print("Done")
    target_layers = [model.rgb_blocks[-2], model.blocks[-2]]

    os.makedirs('figs/df_det', exist_ok=True)
    os.makedirs('figs/face_det', exist_ok=True)
    os.makedirs('figs/cropped', exist_ok=True)
    os.makedirs('figs/cam', exist_ok=True)
    for src in glob("data/ffpp_videos/*/*/c40/videos/00*.mp4"):
        print("Detecting...", end=' ', flush=True)
        video = FaceVideo(src, detector, n_frames=args.n_frames)
        frames = video.load_cropped_frames()
        frames = frames.flatten(0, 1).to(device, non_blocking=True)

        cam_model = ActivationsAndGradients(model, target_layers, None)
        pred = model(frames[None])[0]
        real_prob = pred.softmax(-1)[0].item()
        print("Done")

        label = 'Fake' if real_prob < 0.5 else 'Real'
        confidence = 1 - real_prob if real_prob < 0.5 else real_prob
        print(f'Result: {label}; Confidence: {confidence:.2f}')
        _, _, _, method, _, _, name = src.split('/')

        vw_cam = VideoWriter(os.path.join('figs/cam', f'{method}_{name}'), 2)
        with torch.enable_grad():
            cam_model(frames[None])[0, 1 if real_prob < 0.5 else 0].backward(retain_graph=True)

        # pull the gradients out of the cam_model
        heat_map_sum = 0
        for gradient, activation in zip(cam_model.gradients, cam_model.activations):
            activation *= gradient.mean((2, 3, 4), True)
            heat_map = activation[0].mean(0).relu()
            heat_map_sum = heat_map_sum + heat_map
        heat_map_sum.div_(heat_map_sum.max())
        for heat_map, frame in zip(heat_map_sum.cpu().numpy(), video.crop()):
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            heatmap1 = cv2.resize(heat_map, (frame.shape[1], frame.shape[0]))
            heatmap1 = np.uint8(255 * (heatmap1))
            heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
            vw_cam.write(heatmap1 // 2 + img // 2)
        vw_cam.close()
        cam_model.release()

        h, w = video.frames[0].shape[:2]
        tl = max(1, round(0.002 * (h + w)) )  # line/font thickness
        video = FaceVideo(src, detector, n_frames=128)
        vw_df = VideoWriter(os.path.join('figs/df_det', f'{method}_{name}'), 16)
        vw_face = VideoWriter(os.path.join('figs/face_det', f'{method}_{name}'), 16)
        vw_crop = VideoWriter(os.path.join('figs/cropped', f'{method}_{name}'), 16)
        for i, (frame, cropped, box) in enumerate(zip(video.frames, video.crop(), video.boxes)):
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, (x-w/2, y-h/2, x+w/2, y+h/2))
            color = (0, 0, 255) if real_prob < 0.5 else (0, 255, 0)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_df = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), color, tl)
            img_df = cv2.putText(img_df, label, (x1, y1 - tl * 2), 0, tl, color, tl, cv2.LINE_AA)
            vw_df.write(img_df)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            vw_crop.write(cropped)
            img_face = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (255, 255, 255), tl)
            vw_face.write(img_face)
        vw_df.close()
        vw_face.close()
        vw_crop.close()


if __name__ == "__main__":
    main()
