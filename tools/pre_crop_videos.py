from glob import glob
import os
import pickle
import subprocess
import cv2
from tqdm import tqdm
from decord import VideoReader


class VideoWriter:
    def __init__(self, filename, fps=24) -> None:
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fps = fps
        self.p = None
        self.shape = None
    
    def write(self, frame):
        if not self.filename:
            return
        if self.p is None:
            h, w, _ = self.shape = frame.shape
            self.p = subprocess.Popen([
                "/usr/bin/ffmpeg",
                '-y',  # overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', f'{w}x{h}',  # size of one frame
                '-pix_fmt', 'rgb24',
                '-r', f'{self.fps}',  # frames per second
                '-i', '-',  # The imput comes from a pipe
                '-an',  # Tells FFMPEG not to expect any audio
                '-loglevel', 'error',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                '-tune', 'fastdecode',
                # '-x264opts', 'keyint=0:min-keyint=0:no-scenecut',
                self.filename
            ], stdin=subprocess.PIPE)
        assert self.shape == frame.shape
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.flush()
        self.p.stdin.close()
        return self.p.wait()


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


def to_pngs(pathlist):
    with open("data/ffpp_face_rects_yolov5_s.pkl", "rb") as f:
        video_face_info_d = pickle.load(f)
    for src_path in tqdm(pathlist):
        vid = os.path.basename(src_path)[:-4]
        src_vid = vid.split('_')[0]
        dst_path = src_path.replace("/videos/", f"/faces/")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        writer = VideoWriter(dst_path)
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
    pathlist = sorted(glob("data/ffpp_videos/*/*/c40/videos/*.mp4"))
    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    total = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    pathlist = pathlist[rank::total]
    to_pngs(pathlist)
