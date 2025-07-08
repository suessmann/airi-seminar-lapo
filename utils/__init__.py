import os
import random
import torch
import torch.nn as nn
import numpy as np
from .ema_t import TimeWeightedEMA
from .schedule import *
from .evaluate import *


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def get_optim_groups(model, weight_decay):
    return [
        # do not decay biases and single-column parameters (rmsnorm), those are usually scales
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {
            "params": (p for p in model.parameters() if p.dim() >= 2),
            "weight_decay": weight_decay,
        },
    ]


def get_grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm()
    return norm


def normalize_img(img):
    return ((img / 255.0) - 0.5) * 2.0


def unnormalize_img(img):
    return ((img / 2.0) + 0.5) * 255.0


def weight_init(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.ConvTranspose2d)
    ):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def merge_time_dim(x):
    # x.shape == (B, T, C, H, W) -> (B, T*C, H, W)
    # x = x.permute(0, 1, 4, 2, 3)
    return x.view(x.shape[0], -1, *x.shape[3:])
