import os
import numpy as np
import torch
import torch.utils.data as data
import random

from decord import VideoReader


class CelebDF_Dataset(data.Dataset):
    def __init__(self,
                 root,
                 balance=True,
                 split='train',
                 num_segments=16,
                 transform=None,
                 sparse_span=150,
                 dense_sample=0,
                 test_margin=1.3):
        """Dataset class for ffpp dataset.

        Args:
            root (str): 
                The root path for ffpp data.
            balance (bool, optional):
                Whether to balance the number of real and fake videos. Defaults to True.
            split (str, optional): 
                Data split. One if ['train', 'test']. Defaults to 'train'.
            num_segments (int, optional): 
                How many frames to choose from each video. Defaults to 16.
            transform (function, optional): 
                Data augmentation. Defaults to None.
            sparse_span (int, optional): 
                How many frames to sparsely select from the whole video. Defaults to 150.
            dense_sample (int, optional): 
                How many frames to densely select. Defaults to 0.
            test_margin (float, optional): 
                The margin to enlarge the face bounding box at test stage. Defaults to 1.3.
        """
        super().__init__()

        self.root = root
        self.balance = balance
        self.split = split
        self.num_segments = num_segments
        self.transform = transform
        self.sparse_span = sparse_span
        self.dense_sample = dense_sample
        self.test_margin = test_margin

        self.parse_dataset_info()

    def parse_dataset_info(self):
        """Parse the video dataset information
        """
        self.youtube_real = os.path.join(self.root, 'YouTube-real')
        self.celeb_real = os.path.join(self.root, 'Celeb-real')
        self.celeb_fake = os.path.join(self.root, 'Celeb-synthesis')
        self.split_txt_path = os.path.join(
            self.root, 'List_of_testing_videos.txt')

        assert os.path.exists(self.youtube_real)
        assert os.path.exists(self.celeb_real)
        assert os.path.exists(self.celeb_fake)
        assert os.path.exists(self.split_txt_path)

        test_ids = self.get_test_ids()

        assert self.split in ['test']
        self.real = []
        self.fake = []
        # YouTube-real
        for basename in test_ids[0]:
            self.real.extend(os.path.join(self.youtube_real, basename))
        # Celeb-real
        for basename in test_ids[1]:
            self.real.extend(os.path.join(self.celeb_real, basename))
        # Celeb-synthesis
        for basename in test_ids[2]:
            self.fake.extend(os.path.join(self.celeb_fake, basename))

        print(f"Real: {len(self.real)}, Fake: {len(self.fake)}")

        # Balance the number of real and fake videos
        if self.balance:
            self.fake = np.random.choice(
                self.fake, size=len(self.real), replace=False)
            print(
                f"After Balance | Real: {len(self.real)}, Fake: {len(self.fake)}")

        self.dataset_info = [
            [x, 'real'] for x in self.real] + [[x, 'fake'] for x in self.fake]

    def get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(self.split_txt_path, "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                basename = name.split("/")[-1]
                if "YouTube-real" in name:
                    youtube_real.add(basename)
                elif "Celeb-real" in name:
                    celeb_real.add(basename)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(basename)
                else:
                    raise ValueError(
                        "'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake

    def sample_indices_train(self, video_len):
        """Frame sampling strategy in training stage.

        Args:
            video_len (int): Video frame length.

        """
        base_idxs = np.array(range(video_len), np.int)
        if self.sparse_span:
            base_idxs = np.linspace(
                0, video_len - 1, self.sparse_span, dtype=np.int)
        base_idxs_len = len(base_idxs)

        def over_sample_strategy(total_len):
            if total_len >= self.num_segments:
                offsets = np.sort(random.sample(
                    range(total_len), self.num_segments))
            else:
                inv_ratio = self.num_segments // total_len
                offsets = []
                for idx in range(total_len):
                    offsets.extend([idx] * inv_ratio)
                tail = [total_len - 1] * (self.num_segments - len(offsets))
                offsets.extend(tail)
                offsets = np.asarray(offsets)
            return offsets

        def dense_sample(total_len):
            # print(f'dense! total_len: {total_len}')
            if total_len > self.dense_sample:
                start_idx = np.random.randint(0, total_len - self.dense_sample)
                average_duration = self.dense_sample // self.num_segments
                assert average_duration > 1
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                    np.random.randint(average_duration, size=self.num_segments)
                offsets += start_idx
            else:
                offsets = over_sample_strategy(total_len)
            # print(f'dense offsets: {offsets}')
            return offsets

        def non_dense_sample(total_len):
            average_duration = total_len // self.num_segments
            if average_duration > 1:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                    np.random.randint(average_duration, size=self.num_segments)
            else:
                offsets = over_sample_strategy(total_len)
            return offsets

        if self.dense_sample:
            if random.random() < 0.5:
                offsets = dense_sample(base_idxs_len)
            else:
                offsets = non_dense_sample(base_idxs_len)
        else:
            offsets = non_dense_sample(base_idxs_len)

        return base_idxs[offsets].tolist()

    def sample_indices_test(self, video_len):
        """Frame sampling strategy in test stage.

        Args:
            video_len (int): Video frame count.

        """
        base_idxs = np.array(range(video_len), np.int)
        if self.sparse_span:
            base_idxs = np.linspace(
                0, video_len - 1, self.sparse_span, dtype=np.int)
        base_idxs_len = len(base_idxs)

        if self.dense_sample:
            start_idx = max(base_idxs_len // 2 - self.dense_sample // 2, 0)
            end_idx = min(base_idxs_len // 2 +
                          self.dense_sample // 2, base_idxs_len)
            base_idxs = base_idxs[start_idx: end_idx]
            base_idxs_len = len(base_idxs)

        tick = base_idxs_len / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x)
                           for x in range(self.num_segments)])
        offsets = base_idxs[offsets].tolist()

        return offsets

    def get_enclosing_box(self, img_h, img_w, box, margin):
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

    def decode_selected_frames(self, vr, sampled_idxs, video_face_info_d):
        """Decode image frames from a given video on the fly.

        Args:
            vr (object): 
                Decord VideoReader instance.
            sampled_idxs (list): 
                List containing the frames to extract from the given video.
            video_face_info_d (dict): 
                Dict containing the face bounding box information of each frame from the given video.
        """
        frames = vr.get_batch(sampled_idxs).asnumpy()
        img_h, img_w, _ = frames[0].shape

        if self.split == 'train':
            margin = random.uniform(1.0, 1.5)
        else:
            margin = self.test_margin

        imgs = []
        for idx in range(len(frames)):
            img = frames[idx]
            x0, y0, x1, y1 = self.get_enclosing_box(
                img_h, img_w, video_face_info_d[sampled_idxs[idx]], margin)
            img = img[y0:y1, x0:x1]
            imgs.append(img)
        return imgs

    def __getitem__(self, index):
        video_path, video_label = self.dataset_info[index]

        vr = VideoReader(video_path, num_threads=1)
        video_len = len(vr)

        if self.split == 'train':
            sampled_idxs = self.sample_indices_train(video_len)
        else:
            sampled_idxs = self.sample_indices_test(video_len)

        frames = self.decode_selected_frames(vr, sampled_idxs, None)

        if self.transform is not None:
            frames = self.transform(frames)
        frames = torch.cat(frames)  # TC, H, W

        video_label_int = 0 if video_label == 'real' else 1

        return frames, video_label_int, video_path, sampled_idxs

    def __len__(self):
        return len(self.dataset_info)
