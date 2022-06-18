"""
SBIs at the video level, where each frame is self-blended with a delayed frame.

Prereq:
    Cropped faces and cropped masks after running 
        pre_create_masks.py and pre_crop_videos.py in order.
"""
import os
import torch
import torchvision.transforms.functional as F
import random
import numpy as np
import glob
import argparse
import tqdm
import multiprocessing
from functools import partial
from decord import VideoReader
from tools.utils import VideoWriter

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def transform(frame):
    # TODO: change to albumtation for for choices.
    # frame = F.adjust_hue(frame, np.random.uniform(-0.1, 0.1))
    # frame = F.adjust_saturation(frame, np.random.uniform(0.9, 1.1))
    frame = F.adjust_brightness(frame, np.random.uniform(0.7, 1.3))
    frame = F.adjust_contrast(frame, np.random.uniform(0.7, 1.3))
    return frame


def resize(src, msk):
    # add gaussian blur to msk and 
    # same affine transform to both src, msk (TODO)
    return src, F.gaussian_blur(msk, [35, 35])


def delayed_sbis(frames, msk, delay=5):
    # frames: T, 3, H, W
    # msk: T, 1, H, W
    src, tgt = transform(frames), frames
    src, msk = resize(src, msk)

    src = torch.concat([src[delay:], src[-1].unsqueeze(0).repeat(delay, 1, 1, 1)])
    msk = torch.concat([msk[delay:], msk[-1].unsqueeze(0).repeat(delay, 1, 1, 1)])

    cand_coef = [0.25, 0.5, 0.75, 1]
    coef = cand_coef[np.random.choice(4, p=[.2, .2, .2, .4])]

    msk *= coef

    return src * msk + tgt * (1-msk)


def worker(i, faces_path, masks_path, target_dir, delay=4, use_moviepy=False):
    face_path = faces_path[i]
    mask_path = masks_path[i]
    target_path = os.path.join(target_dir, os.path.basename(face_path))
    if os.path.exists(target_path):
        return
    device = f'cuda:{np.random.choice(torch.cuda.device_count())}' \
        if torch.cuda.is_available() else 'cpu'
    vr = VideoReader(face_path, num_threads=4)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames = torch.from_numpy(frames).to(device)
    frames = frames.permute(0, 3, 1, 2).contiguous() / 255.
    vr = VideoReader(mask_path, num_threads=4)
    msk = vr.get_batch(range(len(vr))).asnumpy()
    msk = torch.from_numpy(msk).to(device)
    msk = msk.permute(0, 3, 1, 2)[:, 0].unsqueeze(1).contiguous() / 255.

    fake = delayed_sbis(frames, msk, delay=delay)
    fake = fake.permute(0, 2, 3, 1).cpu().numpy()
    fake = (fake*255).astype(np.int32) 

    writer = VideoWriter(target_path, use_moviepy=use_moviepy)
    for f in fake:
        writer.write(f)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--moviepy', action='store_true')
    parser.add_argument('--delay', type=int, default=4)
    args = parser.parse_args()

    target_dir = 'data/ffpp_videos/manipulated_sequences/SBIs/c40/faces/'

    masks_dir = 'data/ffpp_videos/original_sequences/youtube/c40/face_masks/'
    faces_dir = 'data/ffpp_videos/original_sequences/youtube/c40/faces/'

    masks_path = sorted(glob.glob(os.path.join(masks_dir, '*.mp4')))
    faces_path = sorted(glob.glob(os.path.join(faces_dir, '*.mp4')))

    if args.processes == 1:
        for i in tqdm.tqdm(range(len(faces_path))):
            worker(i, faces_path, masks_path, target_dir, args.delay, args.moviepy)
    else:
        worker_partial = partial(
            worker, 
            faces_path=faces_path, 
            masks_path=masks_path, 
            target_dir=target_dir,
            delay=args.delay,
            use_moviepy=args.moviepy,
        )

        pool = multiprocessing.Pool(processes=args.processes)

        pool.map(worker_partial, range(len(faces_path)))
        
        pool.close()
