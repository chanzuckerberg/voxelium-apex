#!/usr/bin/env python3

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from voxelium import size_to_maxr, dft, fast_gaussian_filter, idft
from positron.sha3d.cache import Cache


@torch.no_grad()
def get_base_in_real_space(projector, base_index=None, spectral_rescale=None):
    grid_ft = projector.get_base(base_index)
    if spectral_rescale is not None:
        grid_ft *= spectral_rescale
    return idft(grid_ft, dim=3, real_in=True, centered=True)


@torch.no_grad()
def set_base_from_real_space(projector, grid, base_index=None, spectral_rescale=None):
    grid_ft = dft(grid, dim=3, real_in=True, center=True)
    if spectral_rescale is not None:
        grid_ft *= spectral_rescale
    projector.set_base(grid_ft, base_index)


def apply_fast_gaussian_filter(grid, kernel):
    return fast_gaussian_filter(grid[None, None], kernel=kernel)[0, 0]


def apply_maxpool_fuse(x, mask, kernel_size=9):
    padding = kernel_size // 2
    max_x = F.max_pool3d(x.abs()[None, None], kernel_size=kernel_size, stride=1, padding=padding)[0, 0]
    max_x = max_x * mask
    x = x.clip(min=-max_x, max=max_x)
    return x


def apply_smooth_fuse(x, mask):
    kernel = torch.ones(3).to(x.device)
    kernel /= kernel.sum()
    x[mask == 0] = 0
    x[mask < 0.5] *= 0.9
    x_ = apply_fast_gaussian_filter(x, kernel.to(x.device))
    x = x * mask + x_ * (1 - mask)
    return x


@torch.no_grad()
def apply_solvent_mask(projector, masks, spectral_rescale=None, initial=False):
    size = projector.input_size

    if not isinstance(masks, list):
        masks = [masks] * size

    rescale_backward, rescale_forward = None, None
    if spectral_rescale is not None:
        spectral_rescale = spectral_rescale.clip(min=1e-12)
        rescale_backward = Cache.spectra_to_grids(
            spectra=spectral_rescale,
            shape=(projector.size - 1, projector.size - 1, projector.size_x),
            max_r=size_to_maxr(projector.size)
        )
        rescale_forward = Cache.spectra_to_grids(
            spectra=1. / spectral_rescale,
            shape=(projector.size - 1, projector.size - 1, projector.size_x),
            max_r=size_to_maxr(projector.size)
        )

    for base_index in range(size):
        mask = masks[base_index]
        if mask is None:
            continue
        x = get_base_in_real_space(projector, base_index=base_index, spectral_rescale=rescale_backward)

        if initial:
            x *= mask
        else:
            x = apply_smooth_fuse(x, mask)

        set_base_from_real_space(projector, x, base_index=base_index, spectral_rescale=rescale_forward)


class MaskApplicator:
    def __init__(self, projector, solvent_mask=None, roi_mask=None):
        self.projector = projector
        self.masks = None
        do_solvent = solvent_mask is not None
        do_roi = roi_mask is not None

        if do_solvent:
            solvent_mask = solvent_mask.clip(0, 1)
        if do_roi:
            roi_mask = roi_mask.clip(0, 1)

        size = projector.input_size
        if do_solvent and do_roi:
            self.masks = [solvent_mask] + [roi_mask] * (size - 1)
        elif do_solvent:
            self.masks = [solvent_mask] * size
        elif do_roi:
            self.masks = [None] + [roi_mask] * (size - 1)

        if self.masks is not None:
            apply_solvent_mask(projector, self.masks, initial=True)

    @torch.no_grad()
    def __call__(self, spectral_rescale=None):
        if self.masks is not None:
            apply_solvent_mask(
                projector=self.projector,
                masks=self.masks,
                spectral_rescale=spectral_rescale
            )

