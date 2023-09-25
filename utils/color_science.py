import numpy as np
from utils import smv_colour
import torch


def gamma(x, colorspace='sRGB'):
    y = np.zeros(x.shape)
    y[x > 1] = 1
    if colorspace in ('sRGB', 'srgb'):
        y[(x >= 0) & (x <= 0.0031308)] = (323 / 25 *
                                          x[(x >= 0) & (x <= 0.0031308)])
        y[(x <= 1) & (x > 0.0031308)] = (
            1.055 * abs(x[(x <= 1) & (x > 0.0031308)])**(1 / 2.4) - 0.055)
    return y


def gamma_reverse(x, colorspace='sRGB'):
    y = np.zeros(x.shape)
    y[x > 1] = 1
    if colorspace in ('sRGB', 'srgb'):
        y[(x >= 0) & (x <= 0.04045)] = x[(x >= 0) & (x <= 0.04045)] / 12.92
        y[(x > 0.04045) & (x <= 1)] = ((x[(x > 0.04045) & (x <= 1)] + 0.055) /
                                       1.055)**2.4
    return y


def calculate_cct(white_point):
    print(white_point)
    white_point_XYZ = smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709")
    white_point_xyY = smv_colour.XYZ2xyY(white_point_XYZ)
    x, y = white_point_xyY[0], white_point_xyY[1]
    n = (x - 0.3320) / (y - 0.1858)
    cct = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33
    return cct
