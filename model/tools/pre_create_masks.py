"""
Create uncropped face mask videos.

Prereq:
    data/ffpp_face_landmarks81.pkl

Usage:
    python3 -m tools.pre_create_masks
"""
import os
import pickle
import numpy as np
import cv2
import multiprocessing
import argparse
import tqdm
import glob
from decord import VideoReader
from functools import partial

from tools.utils import VideoWriter


def worker(i, video_paths, landmarkss, use_moviepy=False):

    # c40/videos/000/000.mp4
    video_path = video_paths[i]
    mask_video_path  = video_path.replace('/videos', '/masks')

    if os.path.exists(mask_video_path):
        return

    landmarks = landmarkss[str(i).zfill(3)]

    frame_shape = VideoReader(video_path)[0].shape

    frame_writer = VideoWriter(mask_video_path, use_moviepy=use_moviepy)

    # masks = []

    last_landmark = np.zeros((1, 81, 2))
    last_mask = np.zeros(frame_shape[:-1])
    for landmark in landmarks:
        mask = np.zeros(frame_shape[:-1])
        if landmark:
            landmark = np.array(landmark, dtype=np.int32)
            if landmark.shape[0] > 1:
                # if more than two faces present
                # (n, 81, 2) - (1, 81, 2)
                diff = pow((landmark - last_landmark), 2).sum((1, 2))
                idx = diff.argmin()
                landmark = landmark[idx][None, ...]
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
            mask = mask*255
            last_landmark = landmark
            last_mask = mask
        else:
            # no face detected
            mask = last_mask
        # grayscale -> rgb
        frame_writer.write(np.tile(mask[..., None], (1,1,3)))
    
    # to video
    frame_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['FaceForensics++', 'ffpp_videos'], default='ffpp_videos')
    parser.add_argument('--compression', choices=['raw','c23','c40'], default='c40')
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--moviepy', action='store_true')
    args = parser.parse_args()

    dataset_folder = f'data/{args.dataset}/original_sequences/youtube/{args.compression}/' 

    video_paths = sorted(glob.glob(os.path.join(dataset_folder, 'videos/*.mp4')))

    os.makedirs(os.path.join(dataset_folder, 'masks'), exist_ok=True)

    print('loading data/ffpp_face_landmarks81.pkl')
    landmarkss = pickle.load(open('data/ffpp_face_landmarks81.pkl', 'rb'))

    if args.processes == 1:

        for i in tqdm.tqdm(range(len(video_paths))):
            worker(i, video_paths, landmarkss, use_moviepy=args.moviepy)

    else:

        worker_partial = partial(
            worker, 
            video_paths=video_paths, 
            landmarkss=landmarkss,
            use_moviepy=args.moviepy,
        )

        pool = multiprocessing.Pool(processes=args.processes)

        pool.map(worker_partial, range(len(video_paths)))
        
        pool.close()
