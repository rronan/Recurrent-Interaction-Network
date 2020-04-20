import math

import torch
import torch.nn.functional as F
from torch import nn

from pyronan.model import Model
from pyronan.utils.torchutil import fillnan, obj_nan

LOG2PI = math.log(2 * math.pi)
VAR = 1 / 128 ** 2
LOGVAR = math.log(VAR)


def reset_parameters_linear(module, init_scale):
    stdv = 1.0 / math.sqrt(module.weight.size(0))
    nn.init.uniform_(module.weight, -stdv * init_scale, stdv * init_scale)
    if module.bias is not None:
        nn.init.uniform_(module.bias, -stdv * init_scale, stdv * init_scale)


class phi(nn.Module):
    def __init__(self, f, init_scale=1):
        super().__init__()
        self.f = f
        reset_parameters_linear(self.f, init_scale)

    def forward(self, X):
        X = X.unsqueeze(2)
        Z = self.f(X)
        Z = Z.squeeze(2)
        return Z


class GaussNLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # res = 0.5 * (LOG2PI + logvar + (y - mu) ** 2 / torch.exp(logvar))
        mu, logvar = x[:, :, 1], x[:, :, -1]
        res = 0.5 * (logvar - LOGVAR + (VAR + (y - mu) ** 2) / torch.exp(logvar) - 1)
        return res.mean()


def stack_phi(d_in, d_hid, d_out, n, init_scale=1):
    phi_l = [phi(nn.Conv2d(d_in, d_hid, (1, 1), bias=False), init_scale), nn.ReLU()]
    for i in range(n - 2):
        phi_l.extend([phi(nn.Conv2d(d_hid, d_hid, (1, 1)), init_scale), nn.ReLU()])
    phi_l.append(phi(nn.Conv2d(d_hid, d_out, (1, 1)), init_scale))
    return nn.Sequential(*phi_l)


def merge_fbwd(batch):
    return [b.view(-1, *b.shape[-3:]) for b in batch]


def _taylor_approximation(array):
    return sum(a / math.factorial(i) for i, a in enumerate(array))


def taylor_approximation(x, p):
    xp = torch.cat([x.transpose(0, 1), p.transpose(0, 1)])
    res = [_taylor_approximation(xp[i:]) for i in range(x.size(1))]
    res = torch.stack(res).transpose(0, 1)
    return res


class MSEGaussNLL(nn.Module):
    def __init__(self, weight_gauss):
        super().__init__()
        self.weight_gauss = weight_gauss
        self.loss_gauss = GaussNLL()
        self.loss_mse = nn.MSELoss()

    def forward(self, mu, logvar, y):
        res = self.weight_gauss * self.loss_gauss(mu, logvar, y) + self.loss_mse(mu, y)
        return res.mean()


class Core_IntNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cpu"
        self.N_o = args.N_o
        self.N_r = self.N_o * (self.N_o - 1) if self.N_o > 1 else 1
        self.R_r = None
        self.R_s = None
        self.R_a = torch.FloatTensor(1, args.D_r, 1)
        self.logvar_bias = args.logvar_bias
        self.proba = self.logvar_bias is not None
        self.phi_r = stack_phi(
            2 * args.D_s + args.D_r, args.d_r, args.D_e, args.n_r, args.init_scale
        )
        torch.nn.init.normal_(self.R_a, -1, 1)
        self.R_a = nn.Parameter(self.R_a, requires_grad=True)
        self.phi_x = stack_phi(
            args.D_s + args.D_x, args.d_x, args.D_e, args.n_x, args.init_scale
        )
        self.f = stack_phi(2 * args.D_e, args.d_f, args.D_p, args.n_f, args.init_scale)
        self.is_gpu = False

    def compute_R(self, o):
        self.R_r = torch.zeros((o.size(0), self.N_o, self.N_r)).to(self.device)
        self.R_s = torch.zeros((o.size(0), self.N_o, self.N_r)).to(self.device)
        c = 0
        for i in range(self.N_o):
            for j in range(self.N_o):
                if i != j:
                    self.R_r[:, i, c] = 1
                    self.R_s[:, j, c] = 1
                    c += 1
        self.R_r.requires_grad_()
        self.R_s.requires_grad_()

    def _cat_relations(self, o, bsz):
        a = [
            torch.bmm(o, self.R_r),
            torch.bmm(o, self.R_s),
            self.R_a.expand(o.size(0), -1, self.N_r),
        ]
        B = torch.cat(a, 1)
        return B

    @staticmethod
    def _cat_external(o, x):
        x = x.expand(x.size(0), x.size(1), o.size(-1))
        B_x = torch.cat([o, x], 1)
        return B_x

    def _cat_f(self, E_rel, E_ext, o, x):
        r = torch.bmm(E_rel, self.R_r.transpose(1, 2))
        return torch.cat([r, E_ext], 1)

    def _forward(self, x, ext):
        input_rel = self._cat_relations(x, x.size(0))  # ... x (2 * D_s + D_r) x N_r
        input_ext = self._cat_external(x, ext)  # ... x (D_s + D_x) x N_o
        E_rel = self.phi_r(input_rel)  # ..D_e x N_r
        E_ext = self.phi_x(input_ext)  # ..D_e x N_o
        input_f = self._cat_f(E_rel, E_ext, x, ext)
        p = self.f(input_f)  # ... x D_p x N_o
        return p

    def forward(self, x, ext=None):
        bsz, N_o, c, f = x.size()
        if ext is None:
            ext = torch.ones([bsz, 1, 1, 1]).to(self.device)
        isnan = obj_nan(x)
        x.masked_fill_(isnan, 0)
        x = x.permute((0, 2, 3, 1))
        ext = ext.permute((0, 2, 3, 1))
        if self.R_r is None or (self.R_r.size(0) != x.size(0)):
            self.compute_R(x)
        x = x.contiguous().view(bsz, -1, N_o)
        ext = ext.contiguous().view(bsz, -1, 1)
        p = self._forward(x, ext)
        p = p.view(bsz, 1 + self.proba, -1, N_o)
        x = x.view(bsz, c, f, N_o)
        p = F.pad(p, pad=[0, 0, 0, x.size(2) - p.size(2)], mode="constant", value=0)
        if self.proba:
            mean = taylor_approximation(x.narrow(1, 0, c - 1), p.narrow(1, 0, 1))
            logvar = p.narrow(1, 1, 1) + self.logvar_bias
            P = torch.cat([mean, logvar], dim=1)
        else:
            P = taylor_approximation(x, p)
        P = P.permute((0, 3, 1, 2))
        P.masked_fill_(isnan, float("nan"))
        return P


class IntNet(Model):
    def __init__(self, args):
        super().__init__()
        self.core_module = Core_IntNet(args)
        self.proba = args.logvar_bias is not None
        if self.proba:
            self.loss = GaussNLL()
        else:
            mse = nn.MSELoss()
            self.loss = lambda x, y: mse(x[:, :, 0], y)
        super().set_optim(args)

    def step(self, batch, set_):
        batch = merge_fbwd(self.to_device(batch))
        n = len(batch) // 2
        (x, ext), (y, *_) = batch[:n], batch[n:]
        seq_loss = 0
        res = {}
        P = self.core_module.forward(x, ext)
        loss = self.loss(fillnan(P), y)
        if self.proba:
            res[f"mu"] = ((fillnan(P[:, :, 0]) - y) ** 2).mean().item()
            res[f"lv"] = fillnan(P[:, :, -1]).mean().item()
        x = P
        if set_ == "train" and self.optimizer is not None:
            self.update(loss)
        res["loss"] = seq_loss.item()
        return res
