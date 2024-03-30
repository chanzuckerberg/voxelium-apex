#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import warnings
from typing import List, TypeVar, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from positron.base import fourier_shift_2d
from positron.sha3d.cache import Cache

import matplotlib.pyplot as plt


def mean_over_groups(data, groups=None):
    if groups is None:
        return data
    size = int(torch.max(groups).item()) + 1
    means = torch.zeros([size] + list(data.shape[1:]), dtype=data.dtype).to(data.device)
    for i in range(size):
        means[i] = data[groups == i].mean(0)
    return means


@torch.no_grad()
def get_map_features(x, y, c, wx, wy, eps=1e-12, groups=None):
    xT = torch.conj(x)
    c2 = c.square()

    def solve(lhs, rhs):
        lhs = torch.real(lhs)
        lhs = 0.5 * (lhs + lhs.transpose(1, 2))  # Symmetrize
        rhs = torch.real(rhs)
        eye = torch.eye(lhs.shape[-1]).to(lhs.device) * eps
        return torch.linalg.solve(lhs + eye[None], rhs)

    # xT @ x * (c2 * wy + wx) * z = xT @ wy * c * y
    lhs = torch.einsum('bnk, bmk -> bnm', xT, x * (c2 * wy + wx).unsqueeze(1))
    lhs = mean_over_groups(lhs, groups)
    rhs = torch.einsum('bnk, bk -> bn', xT, wy * c * y)
    rhs = mean_over_groups(rhs, groups)
    return solve(lhs, rhs)


def spectra_to_masked_grid(
        spectra, grid_shape, view_as_complex=False, minr=None, maxr=None):
    batched = spectra.dim() == 2
    grid = Cache.spectra_to_grids(spectra, grid_shape, maxr)
    grid = apply_spectral_mask(grid, view_as_complex=view_as_complex, batched=batched, minr=minr, maxr=maxr)
    return grid


def apply_spectral_mask(grid, view_as_complex=False, batched=True, minr=None, maxr=None):
    if view_as_complex:
        grid = torch.view_as_complex(grid)

    spectral_mask = Cache.get_spectral_mask(
        grid.shape[1:] if batched else grid.shape,
        max_r=maxr,
        min_r=minr,
        device=grid.device
    )

    if batched:
        grid = grid[:, spectral_mask]
    else:
        grid = grid[spectral_mask]

    return grid


def get_decomposed_poses(decoder, hv) -> torch.Tensor:
    device = decoder.projector.weight.device
    s_size = decoder.s_size

    batch_size = hv['rot_matrices'].shape[0]
    rot_matrices = hv['rot_matrices'].unsqueeze(1).expand(batch_size, s_size, 3, 3)

    s_prob = torch.eye(s_size, device=device)
    s_prob = s_prob.unsqueeze(0).expand(batch_size, s_size, s_size)

    x = decoder(
        s=s_prob.reshape(batch_size * s_size, s_size),
        # max_r=max_r, # TODO max_r should be used here
        rot_matrices=rot_matrices.reshape(batch_size * s_size, 3, 3).contiguous()
    )

    shifts = hv["shifts_resid"].unsqueeze(1).expand(batch_size, s_size, 2)
    x = fourier_shift_2d(x, shifts.reshape(batch_size * s_size, 2))

    return x


class FeatureExtractor:
    def __init__(
            self,
            decoder,
            voxel_size: float,
            bandpass: List[Tuple[int, int]],
            image_max_r: int = None,
            beta: float = 0.1,
            eps: float = 1e-6
    ):
        self.decoder = decoder
        self.voxel_size = voxel_size
        self.image_max_r = image_max_r
        self.bandpass = bandpass

        self.beta = beta
        self.eps = eps
        self.features_mean = None
        self.features_std = None

        self.s0 = 1

    @torch.no_grad()
    def __call__(self, hv, y, wy, wx, accumulate_stats=True, s0=None, groups=None):
        c = hv['ctfs_']
        batch_size = c.shape[0]
        grid_shape = list(c.shape[1:])
        s_size = self.decoder.s_size

        x = get_decomposed_poses(self.decoder, hv)

        wx = Cache.spectra_to_grids(wx, shape=grid_shape)
        wy = Cache.spectra_to_grids(wy, shape=grid_shape)

        map_features = []

        for minr, maxr in self.bandpass:
            x_ = apply_spectral_mask(x, minr=minr, maxr=maxr, view_as_complex=True)
            x_ = x_.view([batch_size, s_size, x_.shape[-1]])

            y_ = apply_spectral_mask(y, minr=minr, maxr=maxr, view_as_complex=True)
            c_ = apply_spectral_mask(c, minr=minr, maxr=maxr)

            wx_ = apply_spectral_mask(wx, batched=False, minr=minr, maxr=maxr)[None]
            wy_ = apply_spectral_mask(wy, batched=False, minr=minr, maxr=maxr)[None]

            f = get_map_features(x=x_, y=y_, c=c_, wy=wy_, wx=wx_, groups=groups)

            map_features.append(f)

        features = torch.cat(map_features, 1)
        return features

