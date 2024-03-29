#!/usr/bin/env python3

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from positron.base.explicit_grid_utils import size_to_maxr
from positron.sha3d.cache import Cache

from positron.base import dft, make_gaussian_kernel, fast_gaussian_filter, idft


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


@torch.no_grad()
def apply_solvent_mask(projector, masks, bevel=1., spectral_rescale=None):
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

    bevel = np.clip(float(bevel), 0, 1)
    kernel = torch.tensor([bevel, 1., bevel]).float().to(masks[0].device)
    kernel /= kernel.sum()

    for base_index in range(size):
        mask = masks[base_index]
        if mask is None:
            continue
        x = get_base_in_real_space(projector, base_index=base_index, spectral_rescale=rescale_backward)

        x_filter = x * mask.pow(1./5.)  # Delay full mask edge dropout with 5 iterations
        x_filter = apply_fast_gaussian_filter(x_filter, kernel)
        x[mask < 1] = x_filter[mask < 1]

        set_base_from_real_space(projector, x, base_index=base_index, spectral_rescale=rescale_forward)


class MaskApplicator:
    def __init__(self, projector, solvent_mask=None, roi_mask=None, bevel=1.):
        self.projector = projector
        self.masks = None
        self.bevel = bevel
        do_solvent = solvent_mask is not None
        do_roi = roi_mask is not None

        size = projector.input_size
        if do_solvent and do_roi:
            self.masks = [solvent_mask] + [roi_mask] * (size - 1)
        elif do_solvent:
            self.masks = [solvent_mask] * size
        elif do_roi:
            self.masks = [None] + [roi_mask] * (size - 1)

        if self.masks is not None:
            apply_solvent_mask(projector, self.masks)

    @torch.no_grad()
    def __call__(self, spectral_rescale=None):
        if self.masks is not None:
            apply_solvent_mask(
                projector=self.projector,
                masks=self.masks,
                bevel=self.bevel,
                spectral_rescale=spectral_rescale
            )

