import math

import numpy as np
import torch


def get_projection_matrix(fx, fy):
    P = torch.zeros(4, 4)
    P[0, 0] = fx
    P[1, 1] = fy
    P[3, 2] = 1
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_world2view(r, t):
    rt = np.zeros((4, 4))
    rt[:3, :3] = r
    rt[:3, 3] = t
    rt[3, 3] = 1.0
    return np.float32(rt)
