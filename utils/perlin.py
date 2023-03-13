import numpy as np
import math


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(shape, res, interpolant=interpolant):
    d = (math.ceil(shape[0] / res[0]), math.ceil(shape[1] / res[1]))
    tshape = (d[0] * res[0], d[1] * res[1])
    delta = (res[0] / tshape[0], res[1] / tshape[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]

    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1]-1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    p = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    return p[:shape[0], :shape[1]]
