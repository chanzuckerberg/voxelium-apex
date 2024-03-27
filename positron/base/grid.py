#!/usr/bin/env python

"""
Module for calculations related to grid manipulations.
This is temporary, functions should be organized in separate files.
"""

import numpy as np
import mrcfile as mrc
import torch
from typing import Tuple, Union, List

import matplotlib.pylab as plt


def grid_iterator(
        s0: Union[int, np.ndarray],
        s1: Union[int, np.ndarray],
        s2: Union[int, np.ndarray] = None
):
    if isinstance(s0, int):
        if s2 is None:  # 2D grid
            for i in range(s0):
                for j in range(s1):
                    yield i, j
        else:  # 3D grid
            for i in range(s0):
                for j in range(s1):
                    for k in range(s2):
                        yield i, j, k
    else:
        if s2 is None:  # 2D grid
            for i in range(len(s0)):
                for j in range(len(s1)):
                    yield s0[i], s1[j]
        else:  # 3D grid
            for i in range(len(s0)):
                for j in range(len(s1)):
                    for k in range(len(s2)):
                        yield s0[i], s1[j], s2[k]


def save_mrc(grid, filename, voxel_size=1, origin=0.):
    if isinstance(origin, float) or isinstance(origin, int) or origin is None:
        origin = [origin] * 3
    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin[0]
    o.header['origin'].y = origin[1]
    o.header['origin'].z = origin[2]
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()


def load_mrc(mrc_fn):
    mrc_file = mrc.open(mrc_fn, 'r')

    global_origin = mrc_file.header['origin']
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file.header['nxstart']
    global_origin[1] += mrc_file.header['nystart']
    global_origin[2] += mrc_file.header['nzstart']

    c = mrc_file.header['mapc'] - 1
    r = mrc_file.header['mapr'] - 1
    s = mrc_file.header['maps'] - 1

    if c == 0 and r == 1 and s == 2:
        grid = mrc_file.data
    elif c == 2 and r == 1 and s == 0:
        grid = np.moveaxis(mrc_file.data, [2, 1, 0], [0, 1, 2])
    elif c == 1 and r == 0 and s == 2:
        grid = np.moveaxis(mrc_file.data, [0, 2, 1], [0, 1, 2])
    elif c == 0 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data, [1, 0, 2], [0, 1, 2])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    return grid, float(mrc_file.voxel_size.x), global_origin


def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=int), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0]:c[0] + bz[0], c[1]:c[1] + bz[1], c[2]:c[2] + bz[2]] = box
    return nbox, c, c + bz


def resize_grid(
        grid: np.ndarray,
        shape: Union[List, Tuple],
        pad_value: float = 0
) -> np.ndarray:
    """
    Resize grid to shape, by cropping larger and padding smaller dimensions.
    :param grid: The gird to resize
    :param shape: New shape
    :param pad_value: Value to pad smaller dimensions (default=0)
    :return:
    """
    shape = np.array(shape)
    new_shape = np.max([shape, np.array(grid.shape)], axis=0)
    b = np.ones(new_shape) * pad_value
    c = np.array(b.shape) // 2 - np.array(grid.shape) // 2
    assert np.sum(c < 0) == 0
    s = grid.shape
    b[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] = grid
    if not np.all(np.equal(b.shape, shape)):
        c_ = np.array(b.shape) // 2 - np.array(shape) // 2
        assert np.sum(c < 0) == 0
        s = shape
        b = b[c_[0]:c_[0] + s[0], c_[1]:c_[1] + s[1], c_[2]:c_[2] + s[2]]
        c -= c_

    return b, c


def get_bounds_for_threshold(density, threshold=0.):
    """Finds the bounding box encapsulating volume segment above threshold"""
    xy = np.all(density < threshold, axis=0)
    c = [[], [], []]
    c[0] = ~np.all(xy, axis=0)
    c[1] = ~np.all(xy, axis=1)
    c[2] = ~np.all(np.all(density <= threshold, axis=2), axis=1)

    h = np.zeros(3)
    l = np.zeros(3)
    (h[2], h[1], h[0]) = np.shape(density)

    for i in range(3):
        for j in range(len(c[i])):
            if c[i][j]:
                l[i] = j
                break
        for j in reversed(range(len(c[0]))):
            if c[i][j]:
                h[i] = j
                break

    return l.astype(int), h.astype(int)


def smooth_circular_mask(grid_size, radius, thickness):
    ls = torch.linspace(-grid_size / 2, grid_size / 2, grid_size)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, indexing="ij"), -1).square(), -1)
    r = r2.sqrt()
    band_mask = (radius <= r) & (r <= radius + thickness)
    r_band_mask = r[band_mask]
    mask = torch.zeros((grid_size, grid_size))
    mask[r < radius] = 1
    mask[band_mask] = torch.cos(torch.pi * (r_band_mask - radius) / thickness) / 2 + .5
    mask[radius + thickness < r] = 0
    return mask


def smooth_spherical_mask(grid_size, radius, thickness):
    ls = torch.linspace(-grid_size / 2, grid_size / 2, grid_size)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, ls, indexing="ij"), -1).square(), -1)
    r = r2.sqrt()
    band_mask = (radius <= r) & (r <= radius + thickness)
    r_band_mask = r[band_mask]
    mask = torch.zeros((grid_size, grid_size, grid_size))
    mask[r < radius] = 1
    mask[band_mask] = torch.cos(torch.pi * (r_band_mask - radius) / thickness) / 2 + .5
    mask[radius + thickness < r] = 0
    return mask


def smooth_square_mask(image_size, square_side, thickness):
    square_side_2 = square_side / 2.
    y, x = np.meshgrid(
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
    )
    p = np.max([np.abs(x), np.abs(y)], axis=0)
    band_mask = (square_side_2 <= p) & (p <= square_side_2 + thickness)
    p_band_mask = p[band_mask]
    mask = np.zeros((image_size, image_size))
    mask[p < square_side_2] = 1
    mask[band_mask] = np.cos(np.pi * (p_band_mask - square_side_2) / thickness) / 2 + .5
    mask[square_side_2 + thickness < p] = 0
    return mask


def bilinear_shift_2d(
        grid: torch.tensor,
        shift: torch.tensor,
        y_shift: torch.tensor = None
) -> torch.tensor:
    """
    Shifts a batch of 2D images
    :param grid: Batch of images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D images
    """
    if y_shift is not None:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        shift_ = torch.empty([shift.shape[0], 2])
        shift_[:, 0] = shift
        shift_[:, 1] = y_shift
        shift = shift_

    assert len(shift.shape) == 2 and shift.shape[1] == 2
    int_shift = torch.floor(shift).long()

    s0 = shift - int_shift
    s1 = 1 - s0

    int_shift = int_shift.detach().cpu().numpy()
    g00 = torch.empty_like(grid)
    for i in range(len(grid)):
        g00[i] = torch.roll(grid[i], tuple(int_shift[i]), (-1, -2))

    g01 = torch.roll(g00, (0, 1), (-1, -2))
    g10 = torch.roll(g00, (1, 0), (-1, -2))
    g11 = torch.roll(g00, (1, 1), (-1, -2))

    g = g00 * s1[:, 0, None, None] * s1[:, 1, None, None] + \
        g10 * s0[:, 0, None, None] * s1[:, 1, None, None] + \
        g01 * s1[:, 0, None, None] * s0[:, 1, None, None] + \
        g11 * s0[:, 0, None, None] * s0[:, 1, None, None]

    return g


def integer_shift_2d(
        grid: torch.tensor,
        shift: torch.tensor,
        y_shift: torch.tensor = None
) -> torch.tensor:
    """
    Shifts a batch of 2D images
    :param grid: Batch of images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D images
    """
    if y_shift is not None:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        shift_ = torch.empty([shift.shape[0], 2])
        shift_[:, 0] = shift
        shift_[:, 1] = y_shift
        shift = shift_
    assert len(shift.shape) == 2 and shift.shape[1] == 2

    shift = shift.long().detach().cpu().numpy()
    g = torch.empty_like(grid)
    for i in range(len(grid)):
        g[i] = torch.roll(grid[i], tuple(shift[i]), (-1, -2))

    return g


def gaussian_blur(grid, sigma):
    ks = round(sigma * 3)
    ks = max(ks, 3)
    ks += 1 - ks % 2  # Make odd
    ts = np.linspace(-ks / 2, ks / 2, ks)
    gauss = np.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    sharp = grid
    blur = np.zeros_like(sharp)

    for i, k in enumerate(kernel):
        j = i - len(kernel) // 2
        blur[max(j, 0):len(blur) + min(j, 0)] += sharp[max(-j, 0):len(sharp) - max(j, 0)] * k

    if grid.ndim > 1:  # 2D
        sharp = blur
        blur = np.zeros_like(sharp)

        for i, k in enumerate(kernel):
            j = i - len(kernel) // 2
            blur[:, max(j, 0):len(blur) + min(j, 0)] += sharp[:, max(-j, 0):len(sharp) - max(j, 0)] * k

    if grid.ndim > 2:  # 3D
        sharp = blur
        blur = np.zeros_like(sharp)

        for i, k in enumerate(kernel):
            j = i - len(kernel) // 2
            blur[:, :, max(j, 0):len(blur) + min(j, 0)] += sharp[:, :, max(-j, 0):len(sharp) - max(j, 0)] * k

    return blur


def make_gaussian_kernel(sigma):
    ks = round(sigma * 3)
    ks = max(ks, 3)
    ks += 1 - ks % 2  # Make odd
    ts = torch.linspace(-ks / 2, ks / 2, ks)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def fast_gaussian_filter(grid, kernel_sigma=None, kernel=None):
    if kernel is not None:
        k = kernel
    elif kernel_sigma is not None:
        k = make_gaussian_kernel(kernel_sigma).to(grid.device)
    else:
        raise RuntimeError("Either provide sigma or kernel.")
    if len(grid.shape) == 5:  # 3D
        grid = torch.nn.functional.conv3d(grid, k[None, None, :, None, None], padding='same')
        grid = torch.nn.functional.conv3d(grid, k[None, None, None, :, None], padding='same')
        grid = torch.nn.functional.conv3d(grid, k[None, None, None, None, :], padding='same')
    elif len(grid.shape) == 4:  # 2D
        grid = torch.nn.functional.conv2d(grid, k[None, None, :, None], padding='same')
        grid = torch.nn.functional.conv2d(grid, k[None, None, None, :], padding='same')
    else:
        raise NotImplementedError("Only 2D and 3D grids are supported.")

    return grid


def local_correlation(grid1, grid2, kernel_size):
    std = torch.std(grid1) + 1e-28
    grid1 = grid1.unsqueeze(0).unsqueeze(0) / std
    grid2 = grid2.unsqueeze(0).unsqueeze(0) / std

    kernel = make_gaussian_kernel(kernel_size).to(grid1.device)

    def f(a): return fast_gaussian_filter(a, kernel=kernel)

    grid1_mean = grid1 - f(grid1)
    grid2_mean = grid2 - f(grid2)
    norm = torch.sqrt(f(grid1_mean.square()) * f(grid2_mean.square())) + 1e-12
    corr = f(grid1_mean * grid2_mean) / norm

    return corr.squeeze(0).squeeze(0)


def random_blob_on_grid(size, positive, negative, sigma, device="cpu"):
    grid = torch.zeros([size] * 3).to(device)
    if positive > 0:
        coord = torch.clip(torch.randn(3, positive) * size / 8. + size // 2, 0, size - 1).to(device).long()
        grid[coord[0], coord[1], coord[2]] += 1
    if negative > 0:
        coord = torch.clip(torch.randn(3, negative) * size / 8. + size // 2, 0, size - 1).to(device).long()
        grid[coord[0], coord[1], coord[2]] -= 1

    grid = fast_gaussian_filter(grid.unsqueeze(0).unsqueeze(0), kernel_size=sigma).squeeze(0).squeeze(0)
    return grid


def circular_mask(bz, radial_fraction=1.):
    ls = torch.linspace(-1, 1, bz)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, indexing="ij"), -1).square(), -1)
    return r2 < radial_fraction ** 2


def spherical_mask(bz, radial_fraction=1.):
    ls = torch.linspace(-1, 1, bz)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, ls, indexing="ij"), -1).square(), -1)
    return r2 < radial_fraction ** 2


if __name__ == "__main__":
    device = "cuda:0"
    count = 1000
    size = 50
    spread = size / 15

    while True:
        with torch.no_grad():
            g = random_blob_on_grid(size, count, count // 2, spread, device=device)

        plt.imshow(g[size // 2].detach().cpu().numpy())
        plt.show()
