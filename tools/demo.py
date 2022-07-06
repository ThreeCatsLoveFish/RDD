from glob import glob
import cv2
import numpy as np
from decord import VideoReader
from matplotlib import pyplot as plt
import scipy.fft
from tqdm import tqdm


t_size = 4

if __name__ == '__main__':
    video_list = sorted(glob("data/ffpp_videos/manipulated_sequences/Deepfakes/c23/faces/*.mp4"))
    pathname = video_list[4]
    vr = VideoReader(pathname)
    # t_stride = len(vr) // t_size
    t_stride = 1
    clip = vr.get_batch(range(0, t_size * t_stride, t_stride)).asnumpy()
    # clip = vr.get_batch(range(0, t_size)).asnumpy()
    for i, img in enumerate(clip):
        cv2.imwrite(f"figs/s{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    _l = [cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0] for img in clip]
    _l = np.float32(np.stack(_l))
    freq = np.abs(scipy.fft.dctn(_l, norm="forward"))
    freq = np.log(freq)
    for i, img in enumerate(freq):
        plt.imsave(f"figs/f{i}.png", img, vmax=5, vmin=-9)
