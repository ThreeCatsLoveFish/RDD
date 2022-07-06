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
    freq = scipy.fft.dctn(clip, norm="forward")
    mi = scipy.fft.dctn(clip[..., 112], norm="forward")
    # hw, tw, th, mi = map(np.log, map(np.abs, (freq[0], freq[:, 0], freq[..., 0].T, mi.T)))
    hw, tw, th, mi = map(np.abs, (freq[0], freq[:, 0], freq[..., 0].T, mi.T))
    return hw, tw, th, mi


if __name__ == '__main__':
    lists = [[], [], [], []]
    for pathname in tqdm(glob("data/ffpp_videos/manipulated_sequences/Deepfakes/c23/faces/*.mp4")):
        maps = get_freq_response(pathname, t_size=224)
        [l.append(m) for l, m in zip(lists, maps)]
    for i, l in enumerate(lists):
        plt.imsave(f"figs/fake_{i}.png", np.log(np.mean(l, 0)), vmax=4, vmin=-9)
    lists = [[], [], [], []]
    for pathname in tqdm(glob("data/ffpp_videos/original_sequences/youtube/c23/faces/*.mp4")):
        maps = get_freq_response(pathname, t_size=224)
        [l.append(m) for l, m in zip(lists, maps)]
    for i, l in enumerate(lists):
        plt.imsave(f"figs/real_{i}.png", np.log(np.mean(l, 0)), vmax=4, vmin=-9)
