import json
import random

import cv2
import numpy as np
from PIL import Image

from Positional_Reconstruction_Network.src.datasets import ue_utils
from Positional_Reconstruction_Network.src.datasets.dataset import Dataset
from Positional_Reconstruction_Network.src.datasets.detection import make_semantic_mask
from pyronan.utils.image import tis
from pyronan.utils.misc import init_shared_dict, mp_cache


def process_features(f, select_features=None, clip=None):
    if select_features is not None:
        f = f[select_features]
    if clip is not None:
        f = f.clip(-clip, clip)
    f += f.sum(-1, keepdims=True) * 0
    n_obj = 3
    f = f[:n_obj]
    f = np.pad(f, ((0, n_obj - f.shape[0]), (0, 0)), constant_values=float("nan"))
    return f


def process_depth(f, relative_depth=False, depth_noise=None):
    if not relative_depth and depth_noise is None:
        return f
    depth_mask = np.array([False, False, True] * 4 + [False] * 4)[np.newaxis]
    depth_mask = np.repeat(depth_mask, f.shape[0], axis=0)
    if relative_depth:
        f[depth_mask] -= np.nanmin(f[depth_mask])
    if depth_noise is not None:
        f[depth_mask] *= random.random() * depth_noise
    return f


class Dataset_ue(Dataset):
    shared_dict = init_shared_dict()

    def __init__(self, args, set_):
        super().__init__(args, set_)
        self.clip = getattr(args, "clip", None)
        self.init_data(args.input_prefix, set_, args.video_slice)
        self.count_data()
        self.print_stats()
        self.segmentation_model = None
        self.detection_model = None
        self.relative_depth = getattr(args, "relative_depth", False)
        self.depth_noise = getattr(args, "depth_noise", None)
        if "depth" in self.item_list:
            assert not self.relative_depth

    def init_data(self, video_dir, set_, video_slice=slice(None)):
        dir_list = sorted(video_dir.iterdir())
        n = int(len(dir_list) * 0.9)
        dir_list = dir_list[:n] if set_ == "train" else dir_list[n:]
        self.dir_list = dir_list[video_slice]

    def count_data(self, c=None):
        if hasattr(self, "state"):
            self.n_videos, self.n_frames, *_ = self.state.shape
        else:
            self.n_videos, self.n_frames = len(self.dir_list), 100
        self.n_frames = self.n_frames // self.step
        self.count = self.n_videos * self.n_frames

    @mp_cache(shared_dict)
    def read_status(self, video_idx):
        with open(self.dir_list[video_idx] / "status.json", "r") as f:
            status = json.load(f)
        return status

    def object_state_from_status(self, status, frame_idx):
        f = ue_utils.Status.get_object_state(status, frame_idx)
        if len(f.shape) < 2:
            f = np.zeros((1, 7)) * float("nan")
        f = process_features(f, self.select_features, self.clip)
        return f.astype("float32")

    def occluder_state_from_status(self, status, frame_idx):
        f = ue_utils.Status.get_occluder_state(status, frame_idx)
        if len(f.shape) < 2:
            f = np.zeros((1, 12)) * float("nan")
        f = process_features(f, self.select_features, self.clip)
        return f.astype(np.float32)

    def get_object_state(self, video_idx, frame_idx):
        status = self.read_status(video_idx)
        return self.object_state_from_status(status, frame_idx)

    def _get_mask_and_status(self, dir_, frame_idx):
        if self.detection_model is not None:
            return ue_utils.Frame.get_mask_and_idx_from_detection_model(
                dir_, frame_idx, self.detection_model
            )
        if self.segmentation_model is not None:
            return ue_utils.Frame.get_mask_and_idx_from_segmentation_model(
                dir_, frame_idx, self.segmentation_model
            )
        else:
            return ue_utils.Frame.get_mask_and_idx_from_status(dir_, frame_idx)

    @staticmethod
    def _get_object_detection(raw_mask, idx_list, depth):
        k_shape_list = (
            [(k, [1, 0, 0]) for k in idx_list[1]]
            + [(k, [0, 1, 0]) for k in idx_list[2]]
            + [(k, [0, 0, 1]) for k in idx_list[3]]
        )
        if len(k_shape_list) == 0:
            k_shape_list = [(None, [float("nan")] * 3)]
        res = []
        for k, s in k_shape_list:
            res.append(ue_utils.Detection.predict_object_state(raw_mask, depth, k, s))
        return np.array(res)

    @staticmethod
    def _get_visual_features(raw_mask, image, idx_list):
        if len(idx_list) == 0:
            return np.array([[float("nan")] * 3])
        res = []
        for k in idx_list:
            res.append(ue_utils.Detection.make_visual_features(raw_mask, image, k))
        return np.array(res)

    def get_object_detection(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        depth = ue_utils.Frame.get_depth(dir_, frame_idx, normalize=True)
        return self._get_object_detection(raw_mask, idx_list, depth)

    def get_occluder_detection(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        depth = ue_utils.Frame.get_depth(dir_, frame_idx, normalize=True)
        res = []
        for k in idx_list[-1]:
            res.append(ue_utils.Detection.predict_occluder_state(raw_mask, depth, k))
        if len(res) == 0:
            res = np.zeros((1, 12)) * float("nan")
        return np.array(res).astype("float32")

    def get_occluder_detection_pad(self, video_idx, frame_idx):
        f = self.get_occluder_detection(video_idx, frame_idx)
        pad = np.zeros((f.shape[0], 4))
        f = np.concatenate([f, pad], axis=-1)
        f = process_features(f, self.select_features, self.clip)
        res = process_depth(f.astype(np.float32), self.relative_depth, self.depth_noise)
        return res

    def get_state(self, video_idx, frame_idx):
        status = self.read_status(video_idx)
        object_f = self.object_state_from_status(status, frame_idx)
        object_pad = np.zeros((object_f.shape[0], 9))
        object_f = np.concatenate(
            [object_f[:, :3], object_pad, object_f[:, 3:]], axis=-1
        )
        object_f = process_features(object_f, self.select_features, self.clip)
        occluder_f = self.occluder_state_from_status(status, frame_idx)
        occluder_pad = np.zeros((occluder_f.shape[0], 4))
        occluder_f = np.concatenate([occluder_f, occluder_pad], axis=-1)
        occluder_f = process_features(occluder_f, self.select_features, self.clip)
        f = np.concatenate([object_f, occluder_f], axis=0)
        res = f.astype(np.float32)
        res = process_depth(res, self.relative_depth, self.depth_noise)
        return res

    @mp_cache(shared_dict)
    def get_detection(self, video_idx, frame_idx, raw_mask=None, idx_list=None):
        dir_ = self.dir_list[video_idx]
        if raw_mask is None or idx_list is None:
            raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        depth = ue_utils.Frame.get_depth(dir_, frame_idx, normalize=True)
        object_f = self._get_object_detection(raw_mask, idx_list, depth)
        object_pad = np.zeros((object_f.shape[0], 9))
        object_f = np.concatenate(
            [object_f[:, :3], object_pad, object_f[:, 3:]], axis=-1
        )
        object_f = process_features(object_f, self.select_features, self.clip)
        if len(idx_list[-1]) == 0:
            occluder_f = np.zeros((1, 12)) * float("nan")
        o_list = []
        for k in idx_list[-1]:
            o_list.append(ue_utils.Detection.predict_occluder_state(raw_mask, depth, k))
        if len(o_list) == 0:
            o_list.append(np.zeros(12) * float("nan"))
        occluder_f = np.array(o_list)
        occluder_pad = np.zeros((occluder_f.shape[0], 4))
        occluder_f = np.concatenate([occluder_f, occluder_pad], axis=-1)
        occluder_f = process_features(occluder_f, self.select_features, self.clip)
        f = np.concatenate([object_f, occluder_f])
        res = f.astype(np.float32)
        res = process_depth(res, self.relative_depth, self.depth_noise)
        return res

    @mp_cache(shared_dict)
    def get_visual_features(self, video_idx, frame_idx, raw_mask=None, idx_list=None):
        if raw_mask is None or idx_list is None:
            dir_ = self.dir_list[video_idx]
            raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        image = self.get_image_array(video_idx, frame_idx)
        object_v = self._get_visual_features(raw_mask, image, sum(idx_list[1:-1], []))
        object_v = process_features(object_v, None, None)
        occluder_v = self._get_visual_features(raw_mask, image, idx_list[-1])
        occluder_v = process_features(occluder_v, None, None)
        res = np.concatenate([object_v, occluder_v], axis=0).astype(np.float32)
        return res

    @mp_cache(shared_dict)
    def get_detection_visual_features(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        detection = self.get_detection(video_idx, frame_idx, raw_mask, idx_list)
        visual_features = self.get_visual_features(
            video_idx, frame_idx, raw_mask, idx_list
        )
        res = np.concatenate([detection, visual_features], axis=1)
        return res

    @mp_cache(shared_dict)
    def get_hybrid(self, video_idx, frame_idx):
        object_f = self.get_object_state(video_idx, frame_idx)
        object_pad = np.zeros((object_f.shape[0], 9))
        object_f = np.concatenate(
            [object_f[:, :3], object_pad, object_f[:, 3:]], axis=-1
        )
        object_f = process_features(object_f, self.select_features, self.clip)
        occluder_f = self.get_occluder_detection(video_idx, frame_idx)
        occluder_pad = np.zeros((occluder_f.shape[0], 4))
        occluder_f = np.concatenate([occluder_f, occluder_pad], axis=-1)
        occluder_f = process_features(occluder_f, self.select_features, self.clip)
        f = np.concatenate([object_f, occluder_f])
        res = process_depth(f.astype(np.float32), self.relative_depth, self.depth_noise)
        return res

    def get_input(self, video_idx, frame_idx):
        return self.get_hybrid(video_idx, frame_idx)

    def get_target(self, video_idx, frame_idx):
        if "depth" in self.item_list:
            return [
                self.get_semantic_mask(video_idx, frame_idx),
                self.get_depth(video_idx, frame_idx),
            ]
        else:
            return [self.get_semantic_mask(video_idx, frame_idx)]

    def get_image(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        im = Image.open(dir_ / f"scene/scene_{frame_idx+1:03d}.png").convert("RGB")
        if im.size[0] != self.hw or im.size[1] != self.hw:
            im = im.resize((self.hw, self.hw))
        return im

    def get_image_array(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        img = cv2.imread(
            str(dir_ / f"scene/scene_{frame_idx+1:03d}.png"), cv2.IMREAD_UNCHANGED
        )[:, :, :3]
        if img is None:
            print(f"image {dir_} {frame_idx} not found, setting to zero")
            img = np.zeros((self.hw, self.hw, 3))
        if self.hw is not None:
            img = cv2.resize(img, (self.hw, self.hw))
        arr = np.array(img) / 128.0 - 1
        arr = arr.transpose((2, 0, 1)).astype("float32")
        return arr

    def get_semantic_mask(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        raw_mask, idx_list = self._get_mask_and_status(dir_, frame_idx)
        semantic_mask = make_semantic_mask(raw_mask, idx_list).astype("uint8")
        if self.hw is not None:
            semantic_mask = cv2.resize(semantic_mask, (self.hw,) * 2, cv2.INTER_NEAREST)
        return semantic_mask.astype(np.int)

    def get_instance_mask(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        instance_mask, _ = self._get_mask_and_status(dir_, frame_idx)
        if self.hw is not None:
            instance_mask = cv2.resize(instance_mask, (self.hw,) * 2, cv2.INTER_NEAREST)
        return instance_mask

    def get_predicted_semantic_mask(self, video_idx, frame_idx):
        semantic_mask = ue_utils.Frame.get_semantic_mask(
            self.dir_list[video_idx], frame_idx, self.segmentation_model
        )
        if self.hw is not None:
            semantic_mask = cv2.resize(semantic_mask, (self.hw,) * 2, cv2.INTER_NEAREST)
        return semantic_mask

    def get_depth(self, video_idx, frame_idx):
        dir_ = self.dir_list[video_idx]
        depth = ue_utils.Frame.get_depth(dir_, frame_idx, normalize=True)
        if self.hw is not None:
            depth = cv2.resize(depth, (self.hw,) * 2, cv2.INTER_NEAREST)
        return depth

    def get_camera_status(self, video_idx, *_):
        with open(self.dir_list[video_idx] / "status.json", "r") as f:
            camera_status = json.load(f)["header"]["camera"]["location"]
            camera_status = [camera_status[x] for x in "xyz"]
        camera_status = np.array([float(x) / 100 for x in camera_status])
        camera_status = camera_status[np.newaxis, :].astype("float32")
        return camera_status

    def get_path(self, video_idx, frame_idx):
        return str(self.dir_list[video_idx] / f"semantic_mask_{frame_idx+1:03d}.png")
