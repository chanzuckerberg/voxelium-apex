#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import torch
from typing import List, TypeVar, Union, Tuple, Any

import numpy as np
import voxelium as vxm

Tensor = TypeVar('torch.tensor')


# Assume get_freq is defined as follows:
def get_freq(
    shape: Tuple[int, ...],
    pixel_size: Union[float, Tuple[float, ...]] = 1.0,
    rfft: bool = False,
    center: bool = True,
    device: str = "cpu"
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Compute frequency coordinates (in cycles per unit length) for an N-dimensional grid using PyTorch.
    Supports non-cubic (or non-square) shapes. If `rfft` is True, the function computes the frequency
    axis for the last dimension using torch.fft.rfftfreq (assuming Hermitian symmetry on that axis),
    while all other dimensions use torch.fft.fftfreq (with an optional FFT shift if center is True).

    Parameters:
      shape (Tuple[int, ...]): A tuple specifying the size along each dimension in the spatial domain.
      pixel_size (float or Tuple[float, ...]): Physical pixel spacing. If given as a float,
                    that spacing is used for all dimensions; if a tuple, it must match the shape.
      rfft (bool): If True, compute the frequency axis for the last dimension using rfftfreq.
      center (bool): If True, full frequency axes (computed via fftfreq) are shifted so that zero frequency is centered.
                     For the rfft axis, the natural ordering (nonnegative frequencies) is preserved.
      device (str): The PyTorch device for the output tensors.

    Returns:
      For 1D input, returns a single PyTorch tensor representing the frequency axis.
      For an N-D input (N > 1), returns a tuple of PyTorch tensors representing the frequency grids
      for each dimension with "ij" indexing.
    """
    ndim = len(shape)
    
    # Broadcast pixel_size if necessary.
    if not isinstance(pixel_size, (tuple, list)):
        pixel_size = (float(pixel_size),) * ndim
    elif len(pixel_size) != ndim:
        raise ValueError("pixel_size must be a single number or a tuple with length equal to the number of dimensions.")
    
    freq_axes = []
    # When rfft=True, only the last dimension uses rfftfreq.
    for i, (n, d_i) in enumerate(zip(shape, pixel_size)):
        if rfft and (i == ndim - 1):
            f = torch.fft.rfftfreq(n, d=d_i)
        else:
            f = torch.fft.fftfreq(n, d=d_i)
            if center:
                f = torch.fft.fftshift(f)
        freq_axes.append(f.to(device))
    
    # Create the N-dimensional frequency grid with "ij" indexing.
    grids = torch.meshgrid(*freq_axes, indexing="ij")
    
    if ndim == 1:
        return grids[0]
    else:
        return tuple(grids)


def get_spectral_indices_fixed(
    shape: Tuple[int, ...],
    center: bool = True,
    maxr: Union[int, None] = None,
    rfft: bool = False,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Computes spectral (radial) indices for an N-dimensional frequency grid.
    Uses get_freq to generate the per-axis frequency grids.
    The radial index (the floored Euclidean distance) is computed at each grid point.
    Only the DC component will be be zero.
    The indices are scaled by the minimum shape dimension.
    

    Parameters:
      shape (Tuple[int, ...]): The shape of the spatial-domain tensor.
      center (bool): Whether full FFT axes are centered.
      maxr (int or None): Optionally clip the index values to this maximum.
      rfft (bool): If True, the last axis is computed with torch.fft.rfftfreq (assuming Hermitian symmetry).
      device (str): The device for torch tensors.
      
    Returns:
      indices (torch.Tensor): An integer tensor where each value is the floored radial frequency index.
    """
    # Obtain the frequency grids (in cycles per unit length) using get_freq.
    grids = get_freq(shape, rfft=rfft, center=center, device=device)
    
    # Ensure we have a tuple of tensors (even for the 1D case).
    if not isinstance(grids, (tuple, list)):
        grids = (grids,)
    
    # Scale each frequency grid to convert from cycles per unit length to discrete bin-like values.
    # This scaling factor is the product of the number of points along that axis and the sample spacing.
    min_scale = min(shape)
    scaled_grids = []
    for g in grids:
        scaled_grids.append(g * min_scale)
    
    # Compute the squared Euclidean distance at each grid point.
    r2 = sum(g**2 for g in scaled_grids)
    
    # Take the square root, floor the result, and cast to an integer type.
    indices = r2.sqrt().round().long()

    # Only DC should be zero
    mask = (r2 > 0) & (indices == 0)
    indices[mask] = 1
    
    # Optionally clip indices to a maximum value.
    if maxr is not None:
        indices = torch.clamp(indices, max=maxr)
    
    return indices

def get_spectral_indices___(shape):
    shape = list(shape).copy()
    rfft = shape[0] // 2 + 1 == shape[-1]
    shape[-1] = shape[0]
    out = get_spectral_indices_fixed(shape, rfft=rfft, center=True)
    # out = get_spectral_indices_old(shape)
    return out

class Cache:
    square_masks = {}
    circular_masks = {}
    spectral_indices = {}
    spectral_masks = {}
    encoder_input_masks = {}
    
    @staticmethod
    def _get_square_mask(image_size: int, thickness: float) -> Tensor:
        return torch.Tensor(
            vxm.smooth_square_mask(
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
        return vxm.smooth_circular_mask(
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
        out = get_spectral_indices___(shape)

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
        out = get_spectral_indices___(shape)

        max_r = min(vxm.size_to_maxr(shape[0]), max_r)

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
        max_r = vxm.size_to_maxr(shape[0]) if max_r is None else max_r
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
        return vxm.spectra_to_grid(spectra=spectra, indices=s_idx).float()

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
        return vxm.grid_spectral_average(grid=grid, indices=s_idx).float()
