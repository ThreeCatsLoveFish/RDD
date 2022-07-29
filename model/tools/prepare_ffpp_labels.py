
from glob import glob
import json


with open("data/ffpp_videos/splits/test.json", 'r') as f:
    json_data = json.load(f)

with open("data/ffpp_nt_test.txt", "w") as f:
    for i, j in json_data:
        f.write(f"1 ffpp_videos/original_sequences/youtube/c40/videos/{i}.mp4\n")
        f.write(f"0 ffpp_videos/manipulated_sequences/NeuralTextures/c40/videos/{i}_{j}.mp4\n")
