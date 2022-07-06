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
    return np.abs(scipy.fft.dctn(clip, axes=0, norm="forward")[1:])


if __name__ == '__main__':
    freqs = []
    for pathname in tqdm(glob("data/ffpp_videos/manipulated_sequences/Deepfakes/c23/faces/*.mp4")):
        freq = get_freq_response(pathname, t_size=224)
        freqs.append(freq.mean((1, 2)))
    plt.plot(np.mean(freqs, 0), label='fake')
    freqs = []
    for pathname in tqdm(glob("data/ffpp_videos/original_sequences/youtube/c23/faces/*.mp4")):
        freq = get_freq_response(pathname, t_size=224)
        freqs.append(freq.mean((1, 2)))
    plt.plot(np.mean(freqs, 0), label='real')
    plt.legend()
    plt.show()
