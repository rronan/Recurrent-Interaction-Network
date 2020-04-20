from multiprocessing import Manager

import numpy as np
from scipy.special import binom

from pyronan.utils.misc import init_shared_dict, mp_cache


def backward_derivatives(array):
    k = array.shape[0] - 1
    res = np.sum([(-1) ** i * binom(k, i) * x for i, x in enumerate(array[::-1])], 0)
    return res


def make_derivatives(array, logvar_bias):
    n = array.shape[0]
    o_list = [backward_derivatives(array[n - k - 1 :]) for k in range(n)]
    o_list[0] += o_list[1] * 0
    if logvar_bias is not None:
        o_list.append(o_list[0] * 0 + logvar_bias)
    res = np.array(o_list).astype("float32").swapaxes(0, 1)
    return res


def add_direction(item):
    fwd = np.pad(item, ((0, 0), (0, 0), (0, 1)), constant_values=1)
    bwd = np.pad(item, ((0, 0), (0, 0), (0, 1)), constant_values=-1)
    return fwd, bwd


class Sequence:
    def __init__(self, dataset, order, seq_length, logvar_bias, bidirectional=False):
        self.dataset = dataset
        self.seq_length = seq_length
        self.order = order
        self.n_frames = dataset.n_frames - seq_length - order + 1
        self.count = dataset.n_videos * self.n_frames
        self.logvar_bias = logvar_bias
        self.bidirectional = bidirectional

    shared_dict = init_shared_dict()

    def make_input(self, item):
        if not self.bidirectional:
            return make_derivatives(item[: self.order], self.logvar_bias)
        fwd, bwd = add_direction(item)
        input_ = np.array(
            [
                make_derivatives(fwd[: self.order], self.logvar_bias),
                make_derivatives(bwd[: -self.order - 1 : -1], self.logvar_bias),
            ]
        )
        return input_

    def make_target(self, item):
        if not self.bidirectional:
            return item[self.order :]
        fwd, bwd = add_direction(item)
        target = np.array([fwd[self.order :], bwd[-self.order - 1 :: -1]])
        return target

    def getitem(self, video_idx, frame_idx):
        idx_seq = range(frame_idx, frame_idx + self.seq_length + self.order)
        data = [list(self.dataset.getitem(video_idx, i)) for i in idx_seq]
        data = [np.array(x) for x in zip(*data)]
        input_, target = [], []
        for item in data:
            input_.append(self.make_input(item))
            target.append(self.make_target(item))
        return tuple(input_ + target)

    @mp_cache(shared_dict)
    def __getitem__(self, index):
        video_idx = index // self.n_frames
        frame_idx = index % self.n_frames
        item = self.getitem(video_idx, frame_idx)
        return item

    def __len__(self):
        return self.count

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)
