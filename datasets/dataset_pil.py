#!/usr/bin/env python3

import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw

from Positional_Reconstruction_Network.src.datasets.dataset import Dataset


def border(box):
    return (box[0], box[1], box[2] + 1, box[3] + 1)


def draw_occluder(array, occluder, v=8):
    hw = array.shape[0]
    im = Image.fromarray(array).convert("L")
    draw = Draw(im)
    o = [occluder[0], -occluder[1], occluder[2], -occluder[3]]
    o = [int((x + 1) * hw / 2) for x in o]
    draw.rectangle(tuple(o), fill=v, outline=v)
    return np.array(im)


class Dataset_pil(Dataset):
    def __init__(self, args, set_):
        super().__init__(args, set_)
        self.occluder = args.occluder
        self.nc_out = args.nc_out
        # TODO add occluders
        self.make_input(args.input_prefix, set_, args.select_features, args.video_slice)
        self.count_data()
        self.print_stats()

    def make_mask(self, f):
        im = Image.new("L", (self.hw, self.hw))
        draw_mask = Draw(im)
        box_list = []
        for i, o in enumerate(f.transpose()):
            r = int(o[-1] * self.hw / 2)
            p = [int((x + 1) * self.hw / 2.0) for x in o[1::-1]]
            box = (p[0] - r, p[1] - r, p[0] + r, p[1] + r)
            box_list.append(box)
            if not all(0 <= b <= self.hw for b in box):
                continue
            draw_mask.ellipse(box, fill=i + 1, outline=i + 1)
        return np.array(im), box_list

    def make_paste_attributes(self, i, i_mask, box):
        d = box[2] - box[0]
        if i < len(self.texture_list):
            texture = self.texture_list[i]
            assert texture.size[0] >= d and texture.size[1] >= d
            p_texture = texture.crop((0, 0, d + 1, d + 1))
        else:
            color = tuple(self.color_list[i])
            p_texture = Image.new("RGB", (d + 1, d + 1), color=color)
        pi_mask = i_mask.crop(border(box))
        return p_texture, pi_mask

    def make_image(self, mask, box_list):
        im = Image.new("RGB", (self.hw, self.hw))
        for i, box in enumerate(box_list):
            i_mask = Image.fromarray((mask == (i + 1)).astype("uint8") * 255)
            p_texture, pi_mask = self.make_paste_attributes(i, i_mask, box)
            im.paste(p_texture, border(box), pi_mask)
        return np.array(im)

    def draw_occluder(self, array, occluder):
        im = Image.fromarray(array)
        draw = Draw(im)
        occluder = [int(x * self.hw / 2) for x in occluder]
        draw.rectangle(
            [0, occluder[0], self.hw, sum(occluder)],
            fill=tuple(self.color_list[self.n_obj]),
            outline=tuple(self.color_list[self.n_obj]),
        )
        return np.array(im)

    def get_target(self, video_idx, frame_idx):
        f = self.data[video_idx, frame_idx]
        mask, _ = self.make_mask(f)
        mask = mask.astype("uint8")
        mask = (mask + (mask >= self.nc_out).astype("uint8")) % self.nc_out
        return mask

    def getitem(self, video_idx, frame_idx):
        input_ = self.get_input(video_idx, frame_idx)
        frame = self.get_target(video_idx, frame_idx)
        return input_, frame
