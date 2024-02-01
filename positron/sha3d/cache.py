#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import sys
from typing import List, TypeVar, Union, Tuple, Any

import numpy as np
import torch

from positron.base.explicit_grid_utils import size_to_maxr

Tensor = TypeVar('torch.tensor')

from positron.base import smooth_square_mask, smooth_circular_mask, get_spectral_indices, dt_symmetrize, \
    spectra_to_grid, grid_spectral_average


class Cache:
    square_masks = {}
    circular_masks = {}
    spectral_indices = {}
    spectral_masks = {}
    encoder_input_masks = {}
    
    @staticmethod
    def _get_square_mask(image_size: int, thickness: float) -> Tensor:
        return torch.Tensor(
            smooth_square_mask(
                image_size=image_size,
                square_side=image_size - thickness * 2,
                thickness=thickness
            )
        )

    @staticmethod
    def get_square_mask(image_size: int, thickness: float, device: Any = 'cpu') -> Tensor:
        tag = str(image_size) + "_" + str(thickness) + "_" + str(device)
        if tag not in Cache.square_masks:
            Cache.square_masks[tag] = Cache._get_square_mask(image_size, thickness).to(device)
        return Cache.square_masks[tag]

    @staticmethod
    def apply_square_mask(input: Tensor, thickness: float) -> Tensor:
        return input * Cache.get_square_mask(input.shape[-1], thickness, input.device)[None, ...]

    @staticmethod
    def _get_circular_mask(image_size: int, radius: float, thickness: float) -> Tensor:
        return smooth_circular_mask(
            grid_size=image_size,
            radius=radius,
            thickness=thickness
        )

    @staticmethod
    def get_circular_mask(image_size: int, radius: float, thickness: float, device: Any = 'cpu') -> Tensor:
        tag = str(image_size) + "_" + str(radius) + "_" + str(thickness) + "_" + str(device)
        if tag not in Cache.circular_masks:
            Cache.circular_masks[tag] = Cache._get_circular_mask(image_size, radius, thickness).to(device)
        return Cache.circular_masks[tag]

    @staticmethod
    def apply_circular_mask(input: Tensor, radius: float, thickness: float) -> Tensor:
        return input * Cache.get_circular_mask(input.shape[-1], radius, thickness, input.device)[None, ...]

    @staticmethod
    def _get_spectral_indices(
            shape: Union[Tuple[int, int], Tuple[int, int, int]], numpy: bool = False, max_r: int = None
    ) -> Union[Tensor, np.ndarray]:
        out = get_spectral_indices(shape)
        if max_r is not None:
            out[out >= max_r] = max_r - 1
        if numpy:
            out = out.cpu().numpy().astype(int)
        return out

    @staticmethod
    def get_spectral_indices(
            shape: Union[Tuple[int, int], Tuple[int, int, int]],
            numpy: bool = False,
            device: Any = 'cpu',
            max_r: int = None
    ) -> Union[Tensor, np.ndarray]:
        tag = str(shape) + "_" + str(max_r)
        tag += "_np" if numpy else "_" + str(device)
        if tag not in Cache.spectral_indices:
            Cache.spectral_indices[tag] = Cache._get_spectral_indices(shape, numpy, max_r)
            if not numpy:
                Cache.spectral_indices[tag] = Cache.spectral_indices[tag].to(device)
        return Cache.spectral_indices[tag]


    @staticmethod
    def _get_spectral_mask(
            shape: Union[Tuple[int, int], Tuple[int, int, int]], max_r: int, min_r: int = None, numpy: bool = False
    ) -> Union[Tensor, np.ndarray]:
        out = get_spectral_indices(shape)
        max_r = min(size_to_maxr(shape[0]), max_r)

        if min_r is None:
            out = out < max_r
        else:
            out = (min_r < out) & (out < max_r)

        if numpy:
            out = out.cpu().numpy()
        return out

    @staticmethod
    def get_spectral_mask(
            shape: Union[Tuple[int, int], Tuple[int, int, int]],
            numpy: bool = False,
            device: Any = 'cpu',
            min_r: int = None,
            max_r: int = None
    ) -> Union[Tensor, np.ndarray]:
        max_r = size_to_maxr(shape[0]) if max_r is None else max_r
        tag = str(shape) + "_" + str(min_r) + "_" + str(max_r)
        tag += "_np" if numpy else "_" + str(device)
        if tag not in Cache.spectral_masks:
            Cache.spectral_masks[tag] = Cache._get_spectral_mask(shape, max_r, min_r=min_r, numpy=numpy)
            if not numpy:
                Cache.spectral_masks[tag] = Cache.spectral_masks[tag].to(device)
        return Cache.spectral_masks[tag]

    @staticmethod
    def spectra_to_grids(
            spectra: Tensor,
            shape: Union[Tuple[int, int], Tuple[int, int, int]],
            max_r: int = None
    ) -> Tensor:
        s_idx = Cache.get_spectral_indices(
            shape,
            max_r=max_r if max_r is not None else spectra.shape[-1],
            device=spectra.device
        )
        return spectra_to_grid(spectra=spectra, indices=s_idx).float()

    @staticmethod
    def grids_to_spectra(
            grid: Tensor,
            max_r: int = None
    ) -> Tensor:
        shape = grid.shape if len(grid.shape) == 2 else grid.shape[1:]
        s_idx = Cache.get_spectral_indices(
            shape,
            max_r=max_r,
            device=grid.device
        )
        return grid_spectral_average(grid=grid, indices=s_idx).float()
