# /usr/bin/env python3

import json
import math
import random

import numpy as np
import pybullet as p

from Positional_Reconstruction_Network.src.datasets.dataset import Dataset
from Positional_Reconstruction_Network.src.datasets.dataset_pil import draw_occluder
from Positional_Reconstruction_Network.src.datasets.detection import detection
from pyronan.utils.misc import init_shared_dict, mp_cache

np.warnings.filterwarnings("ignore")


def load_camera(camera_path):
    with open(camera_path, "r") as f:
        camera = json.load(f)
    return camera


def pproj(p, vmat, pmat):
    g = p @ vmat @ pmat
    x, y, d = g[:, 0] / g[:, 3], g[:, 1] / g[:, 3], g[:, 2]
    return np.array([x, y, d]).T


def make_features(f, vmat, pmat):
    ones = np.ones((f.shape[0], 1))
    p = np.concatenate([f[:, :3], ones], axis=1)
    pp = pproj(p, vmat, pmat)
    res = np.concatenate([pp, f[:, 3:]], axis=1)
    return res


class Dataset_pybullet(Dataset):
    shared_dict = init_shared_dict()

    def __init__(self, args, set_):
        super().__init__(args, set_)
        self.nc_out = args.nc_out
        self.draw_occluder = args.draw_occluder
        self.remove_to_depth = args.remove_to_depth
        self.init_pybullet(args)
        self.raw = np.load(args.input_prefix + f"_{set_}.npy")[args.video_slice]
        self.N_o = args.N_o

        print("self.raw.shape", set_, ":", self.raw.shape)
        self.camera_path = args.camera_path
        self.random_camera = args.random_camera
        self.camera = load_camera(self.camera_path)
        self.make_input(
            set_,
            args.select_features,
            args.remove_to_depth,
            vars(args).get("features_prefix"),
        )
        self.count_data()
        self.print_stats()

    def set_camera_from_path(self, camera_path):
        self.camera = load_camera(camera_path)
        c = self.camera
        self.vmat = p.computeViewMatrix(c["position"], c["target"], c["up"])
        self.pmat = p.computeProjectionMatrixFOV(c["fov"], 1, c["near"], c["far"])
        self.vmat_square = np.array(self.vmat).reshape(4, 4)
        self.pmat_square = np.array(self.pmat).reshape(4, 4)

    def init_pybullet(self, args):
        p.connect(p.DIRECT)
        self.set_camera_from_path(args.camera_path)
        p.resetSimulation()

    def remove_bodies(self):
        for i in range(p.getNumBodies() - 1, -1, -1):
            p.removeBody(i)

    def make_input(
        self, set_, select_features=None, remove_to_depth=None, features_prefix=None
    ):
        if self.random_camera:
            self.camera_angle = [random.random() * math.pi / 4 for _ in self.raw]
        else:
            self.camera_angle = [None] * self.raw.shape[0]

    def set_camera(self, camera_angle):
        self.camera["position"] = [
            0,
            24 * math.sin(camera_angle),
            24 * math.cos(camera_angle),
        ]
        c = self.camera
        self.vmat = p.computeViewMatrix(c["position"], c["target"], c["up"])
        self.pmat = p.computeProjectionMatrixFOV(c["fov"], 1, c["near"], c["far"])
        self.vmat_square = np.array(self.vmat).reshape(4, 4)
        self.pmat_square = np.array(self.pmat).reshape(4, 4)

    def count_data(self):
        self.n_videos, self.n_frames, *_ = self.raw.shape
        self.count = self.n_videos * self.n_frames

    @staticmethod
    def make_object_pybullet(obj, shape=p.GEOM_SPHERE):
        position, r = obj[:3].tolist(), obj[3]
        if obj.shape[0] > 4:
            if obj[4] == 1:
                shape = p.GEOM_BOX
            elif obj[5] == 1:
                shape = p.GEOM_SPHERE
            else:
                return None
        if shape == p.GEOM_SPHERE:
            vuid = p.createVisualShape(shape, radius=r)
        if shape == p.GEOM_BOX:
            vuid = p.createVisualShape(shape, halfExtents=[r] * 3)
        uid = p.createMultiBody(1, -1, vuid, position)
        p.resetBaseVelocity(uid, linearVelocity=(0, 0, 0))
        return uid

    def update_p(self, video_idx, frame_idx):
        self.remove_bodies()
        raw_pos = self.raw[video_idx, frame_idx]
        for obj in raw_pos:
            self.make_object_pybullet(obj)

    def get_depth_mask(self, video_idx, frame_idx):
        self.update_p(video_idx, frame_idx)
        if self.camera_angle[video_idx] is not None:
            self.set_camera(self.camera_angle[video_idx])
        *_, depth_raw, mask = p.getCameraImage(
            self.hw,
            self.hw,
            self.vmat,
            self.pmat,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        mask += 1
        n, f = self.camera["near"], self.camera["far"]
        depth = 2 * f * n / (f + n - (f - n) * (2 * depth_raw - 1))
        if self.remove_to_depth is not None:
            depth -= self.remove_to_depth
        return mask.astype("uint8"), depth

    @mp_cache(shared_dict)
    def get_state(self, video_idx, frame_idx):
        raw_state = self.raw[video_idx, frame_idx]
        if self.camera_angle[video_idx] is not None:
            self.set_camera(self.camera_angle[video_idx])
        state = make_features(raw_state, self.vmat_square, self.pmat_square)
        c = np.array(self.camera["position"])[np.newaxis, :]
        state[:, 2] = np.linalg.norm(raw_state[..., :3] - c, axis=1)
        if self.remove_to_depth is not None:
            state[..., 2] -= self.remove_to_depth
        if self.clip is not None:
            state[(state >= self.clip).any(1)] = float("nan")
            state = state.clip(-self.clip, self.clip)
        return state.astype("float32")

    def get_depth(self, video_idx, frame_idx):
        _, depth = self.get_depth_mask(video_idx, frame_idx)
        if self.draw_occluder is not None:
            depth = draw_occluder(depth, self.draw_occluder, -2)
        return depth

    def get_mask(self, video_idx, frame_idx):
        mask, _ = self.get_depth_mask(video_idx, frame_idx)
        mask = (mask != 0) + (((mask != 0) * (mask - 1)) % (self.nc_out - 1))
        if self.draw_occluder is not None:
            mask = draw_occluder(mask, self.draw_occluder, self.nc_out - 1)
        return mask.astype(np.int_)

    def get_detection(self, video_idx, frame_idx):
        mask, depth = self.get_depth_mask(video_idx, frame_idx)
        pp = detection(mask, self.N_o, depth)
        res = np.concatenate([pp, self.raw[video_idx, frame_idx, :, 4:]])
        return res

    def get_camera_status(self, video_idx, *_):
        if self.camera_angle[video_idx] is None:
            return np.array([[0]])
        return np.array([self.camera_angle[video_idx]])

    def get_input(self, i, j):
        return self.get_state(i, j)

    def get_target(self, i, j):
        return [self.get_mask(i, j)]
