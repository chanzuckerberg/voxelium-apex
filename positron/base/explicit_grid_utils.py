#!/usr/bin/env python3

"""
Module for the sparse linear layer
"""
import copy
import time
from typing import TypeVar, Union
import numpy as np
import torch

Tensor = TypeVar('torch.tensor')


def size_to_maxr(size):
    return size // 2 + 1


def maxr_to_size(maxr):
    return (maxr - 1) * 2


def make_grid2d(size: int = None, max_r: int = None, compact: bool = True):
    """
    Makes image grid coordinates and indices.
    Used by sparse project to output projection images out of 3D grids.
    Must provide either max_r or size. If both are given max_r will be ignored.
    For even box size: img_size = max_r * 2 - 2 <=> max_r = floor(img_size / 2) + 1
    For odd box size: img_size = max_r * 2 - 1 <=> max_r = floor(img_size / 2) + 1
    :param size: Size of the grid containing a max_r circle (not including max_r)
    :param max_r: Max radius of circle contained by image grid.
    :param compact: Remove coordinates >= max_r (outside mask)
    """
    if max_r is None:
        if size is None:
            raise RuntimeError("Either max_r or size must be given.")
        if size % 2 == 0:
            size += 1
        max_r = size_to_maxr(size)
    if size is None:
        if max_r is None:
            raise RuntimeError("Either max_r or size must be given.")
        size = maxr_to_size(max_r)

    size_2 = size // 2

    # Make xy-plane grid coordinates
    ls = torch.linspace(-size_2, size_2, size)
    lsx = torch.linspace(0, size_2, size_2 + 1)
    coord = torch.stack(torch.meshgrid(ls, lsx, indexing='ij'), 2)

    # We need to work with explicit indices, flatten coordinate grid
    radius = torch.sqrt(torch.sum(torch.square(coord), -1))

    # Mask out beyond Nyqvist in 2D grid
    mask = radius < max_r
    mask = mask.flatten()

    if compact:
        coord = coord.view(-1, 2)[mask].contiguous()
    else:
        coord = coord.view(-1, 2).contiguous()

    # import matplotlib
    # import matplotlib.pylab as plt
    # matplotlib.use('TkAgg')
    # plt.plot(coord.data[:, 0].numpy(), coord.data[:, 1].numpy(), '.', alpha=0.3)
    # plt.show()

    coord.require_grad = False

    return coord, mask


def make_grid3d(size: int = None, max_r: int = None):
    """
    Makes volume grid coordinates and indices.
    Used by sparse project to output projection images out of 3D grids.
    Must provide either max_r or size. If both are given max_r will be ignored.
    Note: img_size = max_r * 2 + 1 <=> max_r = floor(img_size / 2)
    :param size: Size of the grid containing a max_r circle
    :param max_r: Max radius of circle contained by image grid.
    """
    if max_r is None:
        if size is None:
            raise RuntimeError("Either max_r or size must be given.")
        if size % 2 == 0:
            size += 1
        max_r = size_to_maxr(size)
    if size is None:
        if max_r is None:
            raise RuntimeError("Either max_r or size must be given.")
        size = maxr_to_size(max_r)

    size_2 = size // 2

    # Make xy-plane grid coordinates
    ls = torch.linspace(-size_2, size_2, size)
    lsx = torch.linspace(0, size_2, size_2 + 1)
    coord = torch.stack(torch.meshgrid(ls, ls, lsx, indexing='ij'), 3)

    # Mask out beyond Nyqvist in 2D grid
    radius = torch.sqrt(torch.sum(torch.square(coord), -1))
    mask = radius < max_r

    return coord, mask


def radial_index_expansion_3d(grid):
    assert grid.shape[0] == grid.shape[1] == grid.shape[2] * 2 - 1
    bz = grid.shape[0]
    bz2 = bz // 2
    mask1 = grid == -1

    ls = np.linspace(-bz2, bz2, bz)
    lsx = np.linspace(0, bz2, bz2 + 1)
    z, y, x = np.meshgrid(ls, ls, lsx, indexing="ij")
    c = np.zeros((int(np.sum(mask1)), 3))
    c[:, 0] = x[mask1]
    c[:, 1] = y[mask1]
    c[:, 2] = z[mask1]

    norm = np.sqrt(np.sum(np.square(c), axis=1))
    c_ = np.round(c / norm[:, None]).astype(int)

    c = c.astype(int)
    c[:, 1:] += bz2
    c_ = c - c_

    g = grid[c_[:, 2], c_[:, 1], c_[:, 0]]
    mask2 = g >= 0

    grid[c[mask2, 2], c[mask2, 1], c[mask2, 0]] = g[mask2]


def radial_index_expansion_2d(grid):
    assert grid.shape[0] == grid.shape[1] * 2 - 1
    bz = grid.shape[0]
    bz2 = bz // 2
    mask1 = grid == -1

    ls = np.linspace(-bz2, bz2, bz)
    lsx = np.linspace(0, bz2, bz2 + 1)
    y, x = np.meshgrid(ls, lsx, indexing="ij")
    c = np.zeros((int(np.sum(mask1)), 3))
    c[:, 0] = x[mask1]
    c[:, 1] = y[mask1]

    norm = np.sqrt(np.sum(np.square(c), axis=1))
    c_ = np.round(c / norm[:, None]).astype(int)

    c = c.astype(int)
    c[:, 1:] += bz2
    c_ = c - c_

    g = grid[c_[:, 1], c_[:, 0]]
    mask2 = g >= 0

    grid[c[mask2, 1], c[mask2, 0]] = g[mask2]
