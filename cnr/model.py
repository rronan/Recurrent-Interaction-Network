from copy import copy
from functools import partial

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from torch_scatter import scatter_softmax, segment_csr

from pyronan.model import Model
from pyronan.utils.image import ti


def make_meshgrid(image_size, initial_image_size):
    space = [np.linspace(-1, 1, s) for s in initial_image_size[::-1]]
    xv, yv = np.meshgrid(*space)
    meshgrid = (
        np.array([xv, yv]).astype("float32")
        * (initial_image_size[:, np.newaxis, np.newaxis] - 1)
        / (image_size[:, np.newaxis, np.newaxis] - 1)
    )
    return torch.FloatTensor(meshgrid).unsqueeze(0)


class Core_CNR(nn.Module):
    def __init__(self, args):
        super(Core_CNR, self).__init__()
        self.depth = args.depth
        self.nc_out = args.nc_out
        self.scale = 2 ** (0.5 / args.n_blocks * args.n_layers)
        hw = np.array([args.height, args.width])
        self.image_size_list = [hw // 2 ** i for i in range(args.n_blocks)][::-1]
        self.initial_image_size_list = [self.image_size_list[-1] + 2 * args.n_layers]
        for _ in range(1, args.n_blocks):
            self.initial_image_size_list.append(
                self.initial_image_size_list[-1] // 2 + 2 * args.n_layers
            )
        self.initial_image_size_list = self.initial_image_size_list[::-1]
        self.meshgrid = make_meshgrid(
            self.image_size_list[0], self.initial_image_size_list[0]
        )
        self.conv_info_grid = nn.Conv2d(args.nc_in + 2, args.nf, 1)
        self.block_list = nn.ModuleList()
        for i in range(len(self.initial_image_size_list)):
            layer_list = nn.ModuleList()
            for j in range(args.n_layers):
                module_list = nn.ModuleList()
                module_list.append(nn.Conv2d(args.nf, args.nf // 4, 1))
                module_list.append(nn.Conv2d(args.nf // 4, args.nf // 4, 3))
                module_list.append(nn.Conv2d(args.nf // 4, args.nf, 1))
                layer_list.append(module_list)
            self.block_list.append(layer_list)
        self.upsample = partial(
            nn.functional.interpolate,
            size=self.initial_image_size_list[-1].tolist(),
            mode=args.upsample_mode,
            align_corners=True,
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv_final = nn.Conv2d(args.nf, self.nc_out + 1, 1)
        self.scale_selector = nn.Parameter(torch.FloatTensor([-1]))
        self.final_nl = nn.Sigmoid() if args.mse else nn.LogSoftmax(dim=1)

    def object_reconstruction(self, x):
        insize = self.initial_image_size_list[0]
        x = x.view(-1, x.size(-1), 1, 1).expand(-1, x.size(-1), *insize)
        for i, layer_list in enumerate(self.block_list):
            if i == 0:
                msh = self.meshgrid.expand(x.size(0), 2, *insize)
                features = torch.cat([x, msh], 1)
                features = self.conv_info_grid(features)
            else:
                features = self.upsample(features)
            for layer in layer_list:
                skip = features
                for conv in layer:
                    features = conv(self.relu(features))
                features *= np.sqrt(self.scale ** 2 - 1)
                features += skip[:, :, 1:-1, 1:-1]
                features /= self.scale
        features = self.conv_final(self.relu(features))
        return features

    def object_selection(self, features, n_obj):
        index = torch.LongTensor(sum([[i] * n for i, n in enumerate(n_obj)], []))
        selector = features[:, :1] * self.scale_selector
        selector = scatter_softmax(selector, index.to(self.device), dim=0)
        indptr = torch.LongTensor([0] + np.cumsum(n_obj).tolist()).to(self.device)
        selected_features = segment_csr(selector * features, indptr, reduce="sum")
        image = selected_features.narrow(1, 1, self.nc_out)
        depth = None
        if self.depth:
            depth = selected_features.narrow(1, 0, 1)
        return image, depth

    def forward(self, x):
        objects = self.object_reconstruction(torch.cat(x, dim=0))
        image, depth = self.object_selection(objects, [z.shape[0] for z in x])
        return [self.final_nl(image), depth]


class CNR(Model):
    def __init__(self, args):
        nn_module = Core_CNR(args)
        super().__init__(nn_module, args)
        self.depth_trunc = args.depth_trunc
        self.depth = args.depth
        self.loss_weight = [1]
        self.loss = [nn.MSELoss()] if args.mse else [nn.NLLLoss()]
        if self.depth:
            self.loss_weight = [1 - args.depth_weight, args.depth_weight]
            self.loss.append(nn.MSELoss())

    def step(self, batch, set_):
        self.nn_module.requires_grad_(set_ == "train")
        x_list = [x.to(self.device) for x in batch[0]]
        y_list = [batch[1].to(self.device)]
        out = self.nn_module(x_list)
        self.out = copy(out)
        self.target = copy(y_list)
        if self.depth and self.depth_trunc:
            selection = (y_list[0] != 0).view(-1).byte()
            y_list[1] = torch.masked_select(y_list[1].view(-1), selection)
            out[1] = torch.masked_select(out[1].view(-1), selection)
        loss_list = []
        for out, y, loss in zip(out, y_list, self.loss):
            loss_list.append(loss(out, y))
        loss_dict = {}
        loss = 0
        for i, (l, w) in enumerate(zip(loss_list, self.loss_weight)):
            loss_dict["loss_%d" % i] = l.data.item()
            loss += l * w
        loss_dict["loss"] = loss.data.item()
        if set_ == "train":
            self.update(loss)
        return loss_dict

    @staticmethod
    def scale(depth, min_, max_):
        depth -= min_
        depth /= max_ - min_
        depth = depth * 255 + 127
        depth = depth.clip(0, 255)
        depth = depth.astype("uint8")
        return depth

    def process_depth(self, depth_true, depth_model):
        min_, max_ = depth_true.min(), depth_true.max()
        return self.scale(depth_true, min_, max_), self.scale(depth_model, min_, max_)

    def get_image(self, *_):
        target = self.target[0].data.cpu().numpy()
        mask_model = np.exp(self.out[0].data.cpu().numpy())
        mask_model = ti(mask_model)
        mask_true = ti(target[:, np.newaxis])
        mask = np.concatenate([mask_true, mask_model], axis=0)
        if self.depth:
            depth_true = self.target[1][0].data.cpu().numpy()
            depth_model = self.out[1][0, 0].data.cpu().numpy()
            if self.depth_trunc:
                depth_true *= target != 0
                depth_model *= target != 0
            depth_true, depth_model = self.process_depth(depth_true, depth_model)
            depth = np.concatenate([depth_true, depth_model], axis=0)
            return np.concatenate([mask, depth], axis=1).transpose((2, 0, 1))[
                np.newaxis
            ]

        return mask.transpose((2, 0, 1))[np.newaxis]

    def gpu(self):
        super().gpu()
        self.nn_module.meshgrid = self.nn_module.meshgrid.cuda()

    def cpu(self):
        super().cpu()
        self.nn_module.meshgrid = self.nn_module.meshgrid.cpu()
