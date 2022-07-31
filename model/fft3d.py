from glob import glob
import cv2
import numpy as np
from decord import VideoReader
from matplotlib import pyplot as plt
import scipy.fft
from tqdm import tqdm

def get_freq_response(path, t_size=224):
    vr = VideoReader(path)

    clip = []
    for img in vr:
        clip.append(cv2.cvtColor(img.asnumpy(), cv2.COLOR_RGB2LAB)[..., 0])
        if len(clip) == t_size:
            break
    while len(clip) < t_size:
        clip.append(clip[-1])
    clip = np.float32(np.stack(clip))
    # clip = torch.from_numpy(clip)
    freq = scipy.fft.fftn(clip, norm="forward")
    # clip_ = scipy.fft.ifftn(freq, norm="forward")
    # print(clip_)
    mi = scipy.fft.fftn(clip[..., 112], norm="forward")
    # hw, tw, th, mi = map(np.log, map(np.abs, (freq[0], freq[:, 0], freq[..., 0].T, mi.T)))
    hw, tw, th, mi = map(np.abs, (freq[0], freq[:, 0], freq[..., 0].T, mi.T))
    return hw, tw, th, mi


if __name__ == '__main__':
    for i, pathname in enumerate(tqdm(glob("data/ffpp_videos/manipulated_sequences/Deepfakes/c40/videos/*.mp4"))):
        if i == 20: break
        maps = get_freq_response(pathname, t_size=224)
        for j in range(4):
            plt.imsave(f"figs/fft_3d_{i}_{j}_fake.png", np.log(maps[j]), vmax=4, vmin=-9)
    lists = [[], [], [], []]
    for i, pathname in enumerate(tqdm(glob("data/ffpp_videos/original_sequences/youtube/c40/videos/*.mp4"))):
        if i == 20: break
        maps = get_freq_response(pathname, t_size=224)
        for j in range(4):
            plt.imsave(f"figs/fft_3d_{i}_{j}_real.png", np.log(maps[j]), vmax=4, vmin=-9)
