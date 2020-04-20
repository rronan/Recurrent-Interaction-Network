#!/usr/bin/env python3

from multiprocessing import Manager

import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw

from pyronan.utils.misc import mp_cache
from Positional_Reconstruction_Network.src.datasets.dataset import Dataset
from Positional_Reconstruction_Network.src.datasets.dataset_pil import draw_occluder


def make_mask(f, hw):
    im = Image.new("L", (hw, hw))
    draw_mask = Draw(im)
    for i, o in enumerate(f.transpose()):
        if np.linalg.norm(o[:3]) == 0:
            continue
        r = int(o[-1] * hw / 2)
        p = [int((x + 1) * hw / 2.0) for x in o]
        box = (p[0] - r, p[1] - r, p[0] + r, p[1] + r)
        if i < 5:
            draw_mask.ellipse(box, fill=i + 1, outline=i + 1)
        else:
            draw_mask.rectangle(box, fill=i + 1, outline=i + 1)
    return np.array(im)


class Dataset_real(Dataset):
    shared_dict = Manager().dict()

    def __init__(self, args, set_):
        super().__init__(args, "train")
        self.nc_out = args.nc_out
        self.draw_occluder = args.draw_occluder
        self.select_features = getattr(args, "select_features", None)
        self.clip = getattr(args, "clip", None)
        self.init_data(args.input_prefix)
        self.n_frames = args.n_frames
        self.count_data()
        self.print_stats()

    def init_data(self, video_dir):
        self.dir_list = sorted(video_dir.dirs())
        self.state_list = [np.load(f) for d in self.dir_list for f in d.files("*.npy")]
        for i, state in enumerate(self.state_list):
            self.state_list[i][:, 4:] = state[:1, 4:]
        assert len(self.state_list) == len(self.dir_list)

    def count_data(self, c=None):
        self.n_videos = len(self.dir_list)
        self.count = self.n_videos * self.n_frames

    @mp_cache(shared_dict)
    def get_state(self, i, j):
        if len(self.state_list) <= i or j >= len(self.state_list[i]):
            return np.zeros()
        state = self.state_list[i][j].copy()
        mask = state.any(1, keepdims=True)
        pad = np.array([[0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]]).t()
        state = np.concatenate([state, pad], axis=1)
        return state[mask].astype(np.float)

    def get_input(self, i, j):
        return self.get_state(i, j)

    def get_target(self, i, j):
        return [self.get_mask(i, j)]

    def get_detection(self, video_idx, frame_idx):
        raise NotImplementedError

    def get_image(self, i, j):
        im_list = sorted((self.dir_list[i] / "color").files("crop*"))
        j = min(j, len(im_list) - 1)
        im = Image.open(im_list[j]).convert("RGB")
        return im.transpose(Image.FLIP_TOP_BOTTOM)

    def get_depth(self, video_idx, frame_idx):
        raise NotImplementedError

    @mp_cache(shared_dict)
    def get_mask(self, video_idx, frame_idx):
        f = self.get_state(video_idx, frame_idx)[:4].copy()
        f[1] *= -1
        mask = make_mask(f, self.hw)
        mask = mask.astype("uint8")
        mask = (mask + (mask >= self.nc_out).astype("uint8")) % self.nc_out
        if self.draw_occluder is not None:
            mask = draw_occluder(mask, self.draw_occluder)
        return mask.astype("uint8")
