import math
from math import factorial as fact

import torch
import torch.nn as nn

from pyronan.model import Model
from Recurrent_Interaction_Networks.src.models import intnet

LOG2PI = math.log(2 * math.pi)
VAR = 1 / 128 ** 2
LOGVAR = math.log(VAR)


def estimate_state(p, m, ghk):
    p_list = [p[:, :, i] + c * (p[:, :, 0] - m) / fact(i) for i, c in enumerate(ghk)]
    res = torch.stack(p_list).permute((1, 2, 0, 3))
    return res


class Recurrent(Model):
    def __init__(self, dyn, args):
        super().__init__()
        self.core_module = dyn
        self.proba = args.logvar_bias is not None
        if self.proba:
            self.loss = intnet.GaussNLL()
        else:
            mse = nn.MSELoss()
            self.loss = lambda x, y: mse(x[:, :, 0], y)
        self.loss_weight_list = args.loss_weight_list
        self.ghk = args.ghk
        self.D_p = args.D_p
        if getattr(self.core_module, "trainable", True):
            super().set_optim(args)

    def step(self, batch, set_):
        batch = intnet.merge_fbwd(self.to_device(batch))
        n = len(batch) // 2
        if n > 1:
            (x, ext), (y, *_) = batch[:n], batch[n:]
        else:
            x, y = batch
            ext = None
        seq_loss = 0
        res = {}
        for j, y_j in enumerate(map(intnet.fillnan, y.permute(1, 0, 2, 3))):
            P = self.core_module.forward(x, ext)
            loss = self.loss(intnet.fillnan(P), y_j)
            seq_loss += self.loss_weight_list[j] * loss
            if self.proba:
                res[f"mu_{j}"] = ((intnet.fillnan(P[:, :, 0]) - y_j) ** 2).mean().item()
                res[f"lv_{j}"] = intnet.fillnan(P[:, :, -1]).mean().item()
            else:
                res[f"loss_{j}"] = loss.item()
            x = P
        if set_ == "train" and self.optimizer is not None:
            self.update(seq_loss)
        res["loss"] = seq_loss.item()
        return res
