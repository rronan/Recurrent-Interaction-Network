import json
import math

import cv2
import imutils
import numpy as np
import torch
from PIL import Image

from Positional_Reconstruction_Network.src.datasets import detection

DEPTH_STATS = {
    "maxdepth_mean": 72228.7336040598,
    "maxdepth_std": 121856.77488011424,  # not used
    "depth_mean": 1076.24380130159,
    "depth_std": 1587.6561691828697,
}

CAMERA_ANGLE = 90.0


class Status:
    def _get_camera_loc_rot(camera):
        cloc = np.array([camera["location"][i] for i in "xyz"])
        crot = np.array([camera["rotation"][i] for i in ["roll", "pitch", "yaw"]])
        crot[1] *= -1
        return cloc, crot

    def _get_object_loc_rot(frame_status):
        battr, bloc = [], []
        for k, o in sorted(frame_status.items()):
            if "object" not in k:
                continue
            bloc.append([o["location"][x] for x in "xyz"])
            battr.append([o["scale"][x] * s for x, s in zip("xyz", [100] * 3)])
        return np.array(battr), np.array(bloc)

    def _get_occluder_loc_rot(frame_status):
        oattr, oloc, orot = [], [], []
        for k, o in sorted(frame_status.items()):
            if "occluder" not in k:
                continue
            oloc.append(np.array([o["location"][x] for x in "xyz"]))
            orot.append(np.array([o["rotation"][x] for x in ["roll", "pitch", "yaw"]]))
            oattr.append([o["scale"][x] * s for x, s in zip("xyz", [400, 20, 200])])
        return np.array(oattr), np.array(oloc), np.array(orot)

    def _object_cardinals(c, a, attr):
        h = np.array([a[1] - c[1], c[0] - a[0], 0])
        h /= np.linalg.norm(h) * 2
        v = np.array([0, 0, 1]) / 2
        return [
            a + attr[:2].mean() * h,
            a - attr[:2].mean() * h,
            a + attr[2] * v,
            a - attr[2] * v,
        ]

    def _occ_corners(a, attr, rot):
        shape = []
        rot = [-x for x in rot]
        rot[0] *= -1
        for c in [[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]]:
            shape.append(attr * np.array(c))
        mat0 = np.matrix(
            [
                [1, 0, 0],
                [0, math.cos(rot[0]), math.sin(rot[0])],
                [0, -math.sin(rot[0]), math.cos(rot[0])],
            ]
        )
        mat1 = np.matrix(
            [
                [math.cos(rot[1]), 0, -math.sin(rot[1])],
                [0, 1, 0],
                [math.sin(rot[1]), 0, math.cos(rot[1])],
            ]
        )
        mat2 = np.matrix(
            [
                [math.cos(rot[2]), math.sin(rot[2]), 0],
                [-math.sin(rot[2]), math.cos(rot[2]), 0],
                [0, 0, 1],
            ]
        )
        shape = [
            np.array(mat2.dot(mat1).dot(mat0).dot(np.matrix(x).transpose())).ravel()
            for x in shape
        ]
        return [a + x for x in shape]

    def _pproj(a, c, theta):
        mat1 = np.matrix(
            [
                [math.cos(theta[1]), 0, -math.sin(theta[1])],
                [0, 1, 0],
                [math.sin(theta[1]), 0, math.cos(theta[1])],
            ]
        )
        mat2 = np.matrix(
            [
                [math.cos(theta[2]), math.sin(theta[2]), 0],
                [-math.sin(theta[2]), math.cos(theta[2]), 0],
                [0, 0, 1],
            ]
        )
        d = mat1.dot(mat2).dot(np.matrix(a - c).transpose())
        e = 1 / math.tan(CAMERA_ANGLE * math.pi / 360)
        b_x = d[1] / d[0] * e
        b_y = d[2] / d[0] * e
        return [np.asscalar(b_x), np.asscalar(b_y)]

    def get_object_state(status, frame_idx):
        feat = []
        cloc, crot = Status._get_camera_loc_rot(status["header"]["camera"])
        battr, bloc = Status._get_object_loc_rot(status["frames"][frame_idx])
        shape_list = [
            v["shape"]
            for k, v in sorted(status["frames"][frame_idx].items())
            if "object" in k
        ]
        for a, attr, shape in zip(bloc, battr, shape_list):
            ps = np.dot(
                [np.cos(crot[-1] / 180 * math.pi), np.sin(crot[-1] / 180 * math.pi)],
                (a - cloc)[:2],
            )
            if ps / np.linalg.norm(a[:2] - cloc[:2]) > 0.5:
                p = Status._object_cardinals(cloc, a, attr)
                b = [Status._pproj(x, cloc, crot / 180 * math.pi) for x in p]
                b = list(zip(*b))
                c = [np.mean(b[0]), np.mean(b[1])]
                d = [
                    (np.linalg.norm(a - cloc) - DEPTH_STATS["depth_mean"])
                    / DEPTH_STATS["depth_std"]
                ]
                s = [(np.max(b[0]) - np.min(b[0]) + np.max(b[1]) - np.min(b[1])) / 4]
                t = [
                    float(shape == "Sphere"),
                    float(shape == "Cube"),
                    float(shape == "Cone"),
                ]
                feat.append(c + d + s + t)
        return np.array(feat)

    def get_occluder_state(status, idx):
        feat = []
        cloc, crot = Status._get_camera_loc_rot(status["header"]["camera"])
        oattr, oloc, orot = Status._get_occluder_loc_rot(status["frames"][idx])
        for a, attr, r in zip(oloc, oattr, orot):
            ps = np.dot(
                [np.cos(crot[-1] / 180 * math.pi), np.sin(crot[-1] / 180 * math.pi)],
                (a - cloc)[:2],
            )
            if ps / np.linalg.norm(a[:2] - cloc[:2]) > 0.5:
                p = Status._occ_corners(a, attr, r / 180 * math.pi)
                b = []
                for x in p:
                    b.extend(Status._pproj(x, cloc, crot / 180 * math.pi))
                    b.append(
                        (np.linalg.norm(x - cloc) - DEPTH_STATS["depth_mean"])
                        / DEPTH_STATS["depth_std"]
                    )
                feat.append(b)
        return np.array(feat)

    def get_idx_from_status(status, frame_idx):
        idx_dict = status["header"]["masks"]
        idx = [[], [], [], [], []]
        for k, v in idx_dict.items():
            if "object" in k and k in status["frames"][frame_idx]:
                if status["frames"][frame_idx][k]["shape"] == "Sphere":
                    idx[1].append(v)
                elif status["frames"][frame_idx][k]["shape"] == "Cube":
                    idx[2].append(v)
                elif status["frames"][frame_idx][k]["shape"] == "Cone":
                    idx[3].append(v)
            elif "occluder" in k:
                idx[4].append(v)
            else:
                idx[0].append(v)
        return idx


class Frame:
    def open_mask(dir_, frame_idx):
        raw_mask = cv2.imread(
            str(dir_ / f"masks/masks_{frame_idx+1:03d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if raw_mask is None:
            print(f"mask {dir_}/{frame_idx} not found, setting to zero")
            raw_mask = np.zeros((128, 128))
        return raw_mask.astype("uint8")

    def get_mask_and_idx_from_status(dir_, frame_idx):
        raw_mask = Frame.open_mask(dir_, frame_idx)
        with open(dir_ / "status.json", "r") as f:
            status = json.load(f)
        idx_list = Status.get_idx_from_status(status, frame_idx)
        return raw_mask.astype("uint8"), idx_list

    def get_semantic_mask(dir_, frame_idx, segmentation_model):
        scene = cv2.imread(str(dir_ / f"scene/scene_{frame_idx+1:03d}.png"))[:, :, :3]
        scene = torch.FloatTensor(scene.transpose((2, 0, 1))).to(
            segmentation_model.device
        )
        semantic_mask = segmentation_model(scene.unsqueeze(0))[0].detach().cpu().numpy()
        semantic_mask = semantic_mask.argmax(0)
        return semantic_mask.astype("uint8")

    def get_idx_from_masks(semantic_mask, instance_mask):
        idx = [[], [], [], [], []]
        for i in np.unique(instance_mask):
            v_list = semantic_mask[instance_mask == i]
            bc = np.bincount(v_list).astype("float32")
            bc[3:4] *= 2
            idx[bc.argmax()].append(i)
        return idx

    def get_mask_and_idx_from_detection_model(dir_, frame_idx, detection_model):
        im = Image.open(dir_ / f"scene/scene_{frame_idx+1:03d}.png").convert("RGB")
        detection_model.eval()
        instance_mask = np.zeros(im.size)
        detection = detection_model([im])[0]
        idx_list = [[0], [], [], [], []]
        for i, (label, mask) in enumerate(zip(detection["labels"], detection["masks"])):
            instance_mask[mask > 0.5] = i
            idx_list[label].append(i)
        return instance_mask, idx_list

    def get_mask_and_idx_from_segmentation_model(dir_, frame_idx, segmentation_model):
        semantic_mask = Frame.get_semantic_mask(dir_, frame_idx, segmentation_model)
        instance_mask = Frame.open_mask(dir_, frame_idx)
        idx_list = Frame.get_idx_from_masks(semantic_mask, instance_mask)
        return instance_mask, idx_list

    def open_depth(dir_, frame_idx):
        depth = cv2.imread(
            str(dir_ / f"depth/depth_{frame_idx+1:03d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if depth is None:
            print(f"depth {dir_}/{frame_idx} not found, setting to zero")
            depth = np.zeros((128, 128))
        return depth.astype("float32")

    def get_depth(dir_, frame_idx, normalize=True):
        depth = Frame.open_depth(dir_, frame_idx)
        maxdepth = DEPTH_STATS["maxdepth_mean"]
        depth *= maxdepth / 255
        if normalize:
            depth = (depth - maxdepth / 2) / DEPTH_STATS["maxdepth_mean"]
        return depth


class Detection:
    def make_contours(bin_mask):
        c = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = imutils.grab_contours(c)
        for eps in range(0, 100):
            c_cut = cv2.approxPolyDP(c[0], eps, True)[:, 0]
            if c_cut.shape[0] <= 4:
                break
        assert c_cut.shape[0] > 0
        while c_cut.shape[0] < 4:
            c_cut = np.concatenate([c_cut] * 2, axis=0)
        return c_cut[:4]

    def predict_occluder_state(mask, depth=None, k=None):
        hw = mask.shape[0]
        binmask = (mask == k).astype("uint8")
        if not binmask.any():
            return np.array([float("nan")] * 12)
        c = Detection.make_contours(binmask)
        p = np.mean(c, axis=0).astype("uint8")
        if depth is not None:
            d = depth[p[0], p[1]]
        else:
            d = 0
        features_list = [[x[0] / hw * 2 - 1] + [1 - x[1] / hw * 2] + [d] for x in c]
        features = np.array(sum(features_list, []))
        return features

    def predict_object_state(mask, depth=None, k=None, shape=[1, 0, 0]):
        if k is None or not (mask == k).any():
            return np.array([float("nan")] * 7)
        state_list = [
            detection.g_centroid(mask == k, depth),
            detection.g_radius(mask == k),
            np.array(shape),
        ]
        return np.concatenate(state_list)

    def make_visual_features(mask, image, k):
        if k is None or not (mask == k).any():
            return np.array([float("nan")] * 3)
        return image[:, mask == k].mean(axis=-1)
