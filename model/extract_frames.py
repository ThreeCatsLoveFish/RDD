import argparse
import torch
import pickle
import numpy as np

from loguru import logger
from decord import VideoReader
from face_recog import Detector
from tqdm import tqdm
from glob import glob

from torch.multiprocessing import Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


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
            sampled_idxs = np.linspace(
                0, len(vr) - 1, len(vr), dtype=int).tolist()
            self._frames = list(vr.get_batch(sampled_idxs).asnumpy())
        return self._frames

    @property
    def boxes(self):
        if self._boxes is None:
            self._boxes = self.detector(self.frames)
            cx, cy = self._boxes[:, 0], self._boxes[:, 1]
            hw = self._boxes[:, 2:].max(-1) * 1.2 # margin
            rois = np.stack([cx - hw / 2, cy - hw / 2, cx +
                            hw / 2, cy + hw / 2], 1).clip(0)
            # turn to int and update bbox
            self._boxes = rois.astype(int)
        return self._boxes

def extract_face_infos(detector, video_paths, output_path):
    res = {}
    for video_path in tqdm(video_paths):
        try:
            video = FaceVideo(video_path, detector)
            res[video_path] = video.boxes
        except Exception as e:
            logger.error(e)
    pickle.dump(res, open(output_path, 'wb'))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', type=str,
                        default='./data/*/*.mp4')
    parser.add_argument('-o', '--output', type=str,
                        default='../../data/meta_info/face_info.pkl')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    detectors = []
    for device_id in range(args.workers):
        device = torch.device('cuda:{}'.format(device_id))
        logger.info("Loading face detection model...", end=' ', flush=True)
        detector = Detector('https://github.com/zyayoung/oss/releases/download/rdd/yolov5n-0.5.pt', device)
        detector.model.eval()
        detector.model.share_memory()
        detectors.append(detector)

    # load video
    video_paths = glob(args.input)
    video_paths = np.array_split(video_paths, args.workers)
    process = []
    for device_id in range(args.workers):
        p = Process(target=extract_face_infos,
                    args=(detectors[device_id], video_paths[device_id], f'{args.output}'.replace('.pkl', f'_{device_id}.pkl')))
        p.start()
        process.append(p)
    for p in process:
        p.join()

    # merge
    res = {}
    for device_id in range(args.workers):
        res_tmp = pickle.load(open(f'{args.output}'.replace('.pkl', f'_{device_id}.pkl'), 'rb'))
        res.update(res_tmp)
    
    pickle.dump(res, open(args.output, 'wb'))


if __name__ == "__main__":
    main()
