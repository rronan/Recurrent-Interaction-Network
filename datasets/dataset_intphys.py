#!/usr/bin/env python3

import json

from Positional_Reconstruction_Network.src.datasets.dataset_ue import Dataset_ue


def make_filter_func(visibility, movement, nobjects):
    def filter_func(dict_):
        if visibility is not None and dict_["visibility"] != visibility:
            return False
        if movement is not None and dict_["movement"] != movement:
            return False
        if nobjects is not None and dict_["nobjects"] != nobjects:
            return False
        return True

    return filter_func


class Dataset_intphys(Dataset_ue):
    def __init__(self, args, set_):
        self.nc_out = args.nc_out
        self.select_features = getattr(args, "select_features", None)
        self.clip = getattr(args, "clip", None)
        with open(args.metadata, "r") as f:
            metadata = json.load(f)
        filter_func = make_filter_func(args.visibility, args.movement, args.nobjects)
        blocks = [f"O{i:d}" for i in args.block]
        subdirs = [k for k, v in metadata.items() if filter_func(v) and k[:2] in blocks]
        self.q_list = subdirs
        self.metadata = metadata
        self.init_data(args.input_prefix, subdirs, set_, args.video_slice)
        self.step = getattr(args, "step", 1)
        self.count_data()
        self.segmentation_model = None
        self.detection_model = None
        self.print_stats()
        self.hw = args.hw
        self.set_ = set_
        self.item_list = args.item_list
        self.relative_depth = getattr(args, "relative_depth", False)
        self.depth_noise = getattr(args, "depth_noise", None)
        if "depth" in self.item_list:
            assert not self.relative_depth
        self.getfunc_list = [getattr(self, "get_" + item) for item in args.item_list]

    def init_data(self, input_prefix, subdirs, set_, video_slice=slice(None)):
        self.quadruplet_dir_list = (input_prefix / x for x in subdirs)
        dir_list = sorted(
            sum([list(q.iterdir()) for q in self.quadruplet_dir_list], [])
        )
        if set_ == "test":
            self.dir_list = dir_list[video_slice]
        else:
            n = int(len(dir_list) * 0.9)
            dir_list = dir_list[:n] if set_ == "train" else dir_list[n:]
            self.dir_list = dir_list[video_slice]

    def count_data(self, c=None):
        self.n_videos = len(self.dir_list)
        self.n_frames = 100 // getattr(self, "step", 1)
        self.count = self.n_videos * self.n_frames

    def get_path(self, video_idx, frame_idx):
        return str(self.dir_list[video_idx] / f"semantic_mask_{frame_idx+1:03d}.png")

    def get_is_possible(self, video_idx, *_):
        with open(self.dir_list[video_idx] / "status.json", "r") as f:
            status = json.load(f)
        return status["header"]["is_possible"]

    def get_name(self, video_idx, *_):
        path = str(self.dir_list[video_idx])
        return "/".join(path.split("/")[-3:])

    def get_metadata(self, video_idx, *_):
        name = "/".join(str(self.dir_list[video_idx]).split("/")[-3:])
        qname = "/".join(name.split("/")[-3:-1])
        return {
            "name": name,
            "qname": qname,
            **self.metadata[qname],
            "is_possible": self.get_is_possible(video_idx),
        }

    def get_nobj_max(self, video_idx):
        with open(self.dir_list[video_idx] / "status.json", "r") as f:
            status = json.load(f)
        nobj = [len([k for k in f.keys() if "object" in k]) for f in status["frames"]]
        return max(nobj)
