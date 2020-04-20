from functools import wraps

import numpy as np


def robustMean(a, outlierConstant):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    return np.clip(a, upper_quartile, lower_quartile).mean()


def checknan(fill=np.array([float("nan")])):
    def check_nan_decorator(f):
        @wraps(f)
        def decorated(binmask, *args):
            if not binmask.any():
                return fill
            return f(binmask, *args)

        return decorated

    return check_nan_decorator


@checknan(np.array([float("nan")] * 2))
def g_centroid_2d(binmask):
    c = [x.astype("float32").mean() for x in binmask.nonzero()]
    xy = np.array(c) * 2 / binmask.shape[-1] - 1
    return np.array([xy[1], -xy[0]])


@checknan(np.array([float("nan")] * 3))
def g_centroid_3d(binmask, depth):
    c = [x.astype("float32").mean() for x in binmask.nonzero()]
    xy = np.array(c) * 2 / binmask.shape[-1] - 1
    return np.array([xy[1], -xy[0], depth[int(c[0]), int(c[1])]])


def g_centroid(binmask, depth=None):
    if depth is None:
        return g_centroid_2d(binmask)
    return g_centroid_3d(binmask, depth)


@checknan()
def g_diameter(binmask):  # NOT RADIUS ANYMORE
    r = np.array(binmask.nonzero()).ptp(1).max() / binmask.shape[-1]
    return np.clip(np.array([r]), 0.04, 0.8)


def detection(mask, n_obj, depth=None):
    f_list = [lambda i: g_centroid(mask == i, depth), lambda i: g_diameter(mask == i)]
    li = [[f(i + 1) for f in f_list] for i in range(n_obj)]
    res = np.concatenate([np.concatenate(x) for x in li])
    return res.astype(np.float32)


def make_semantic_mask(raw_mask, idx_list):
    mask = np.zeros(raw_mask.shape, dtype="uint8")
    for i, idx in enumerate(idx_list):
        for k in idx:
            mask[raw_mask == k] = i
    return mask


def add_noise(x, scale, d):
    noise = np.random.normal(0, scale, x.shape)
    noise[..., d:, :] = 0
    return x + noise


def g_bbox(binmask):
    pixel_min = [x.astype("float32").min() for x in binmask.nonzero()]
    pixel_max = [x.astype("float32").max() for x in binmask.nonzero()]
    return pixel_min + pixel_max
