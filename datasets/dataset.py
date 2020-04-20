import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, set_):
        self.hw = args.hw
        self.set_ = set_
        self.select_features = getattr(args, "select_features", None)
        self.item_list = getattr(args, "item_list", [])
        self.getfunc_list = [getattr(self, "get_" + item) for item in self.item_list]
        self.clip = getattr(args, "clip", None)
        self.step = getattr(args, "step", 1)

    def init_data(
        self,
        input_prefix,
        set_,
        video_slice=slice(None),
        backward=False,
        select_features=None,
        clip=None,
    ):
        features = np.load(input_prefix + f"_{set_}.npy")[video_slice]
        if backward:
            features = features[:, ::-1]
        if select_features is not None:
            features = features[..., select_features]
        if clip is not None:
            features = features.clip(-clip, clip)
        self.state = np.nan_to_num(features)

    def print_stats(self):
        if hasattr(self, "state") and self.state is not None:
            d = {"input": self.state}
            print("*****", self.set_, sep="\n")
            for key, value in d.items():
                print(f"{key} max :", np.max(value))
                print(f"{key} min :", np.min(value))
                print(f"{key} mean :", np.mean(value))
                print(f"{key} std :", np.std(value))
                print(f"{key} shape :", value.shape)
            print(f"n samples {self.set_}: {self.count}")

    def count_data(self, c=None):
        self.n_videos, self.n_frames, *_ = self.state.shape
        self.n_frames = self.frames // self.step
        self.count = self.n_videos * self.n_frames

    def get_state(self, video_idx, frame_idx):
        x = self.state[video_idx, frame_idx]
        return x.astype(np.float32)

    def getitem(self, video_idx, frame_idx):
        item_list = [f(video_idx, frame_idx) for f in self.getfunc_list]
        return tuple(item_list)

    def __getitem__(self, index):
        video_idx = index // self.n_frames
        frame_idx = (index % self.n_frames) * self.step
        return self.getitem(video_idx, frame_idx)

    def __len__(self):
        return self.count
