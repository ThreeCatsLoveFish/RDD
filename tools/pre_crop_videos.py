"""
Create cropped videos.

Prereq:
    data/ffpp_face_rects_yolov5_s.pkl

Note:
    run "python tools/pre_create_masks.py" first, then this script 
    (with --mask) flag will create cropped face mask videos in ${compression}/face_masks.

Usage:
    python -m tools.pre_crop_videos --video

    or

    RLAUNCH_REPLICA_TOTAL=$1
    for RLAUNCH_REPLICA in $(seq 0 $(($1 - 1))); do
        python -m tools.pre_crop_videos --video &
    done
    wait
"""
from glob import glob
import os
import pickle
import cv2
import argparse
from tqdm import tqdm
from decord import VideoReader
from tools.utils import VideoWriter


def get_enclosing_box(img_h, img_w, box, margin=1.3):
    """Get the square-shape face bounding box after enlarging by a certain margin.

    Args:
        img_h (int): Image height.
        img_w (int): Image width.
        box (list): [x0, y0, x1, y1] format face bounding box.
        margin (int): The margin to enlarge.

    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    max_size = max(w, h)

    cx = x0 + w / 2
    cy = y0 + h / 2
    x0 = cx - max_size / 2
    y0 = cy - max_size / 2
    x1 = cx + max_size / 2
    y1 = cy + max_size / 2

    offset = max_size * (margin - 1) / 2
    x0 = int(max(x0 - offset, 0))
    y0 = int(max(y0 - offset, 0))
    x1 = int(min(x1 + offset, img_w))
    y1 = int(min(y1 + offset, img_h))

    return [x0, y0, x1, y1]


def to_pngs(pathlist, mask=False, use_moviepy=False):
    with open("data/ffpp_face_rects_yolov5_s.pkl", "rb") as f:
        video_face_info_d = pickle.load(f)
    for src_path in tqdm(pathlist):
        vid = os.path.basename(src_path)[:-4]
        src_vid = vid.split('_')[0]
        if mask:
            dst_path = src_path.replace("/masks/", f"/face_masks/")
        else:
            dst_path = src_path.replace("/videos/", f"/faces/")
        writer = VideoWriter(dst_path, use_moviepy=use_moviepy)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        for frame_id, frame in enumerate(VideoReader(src_path)):
            frame = frame.asnumpy()
            img_h, img_w, _ = frame.shape
            if frame_id >= len(video_face_info_d[src_vid]):
                break
            x0, y0, x1, y1 = get_enclosing_box(img_h, img_w, video_face_info_d[src_vid][frame_id])
            img = frame[y0:y1, x0:x1]
            # cv2.imwrite(dst_path + f"/{frame_id:06d}.png", img)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            writer.write(img)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moviepy', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--mask', action='store_true')
    args = parser.parse_args()

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    total = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))

    if args.video:
        pathlist = sorted(glob("data/ffpp_videos/*/*/c40/videos/*.mp4"))
        pathlist = pathlist[rank::total]
        to_pngs(pathlist, args.mask, use_moviepy=args.moviepy)

    if args.mask:
        pathlist = sorted(glob("data/ffpp_videos/original_sequences/youtube/c40/masks/*.mp4"))
        pathlist = pathlist[rank::total]
        to_pngs(pathlist, args.mask, use_moviepy=args.moviepy)
