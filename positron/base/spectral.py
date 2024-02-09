#!/usr/bin/env python

"""
Module for calculations related to grid manipulations.
This is temporary, functions should be organized in separate files.
"""

import numpy as np
import torch
from typing import Tuple, Union, TypeVar, List

Tensor = TypeVar('torch.tensor')


def get_complex_float_type(type):
    if type == np.float16:
        return np.complex32
    elif type == np.float32:
        return np.complex64
    elif type == np.float64:
        return np.complex128
    elif type == torch.float16:
        return torch.complex32
    elif type == torch.float32:
        return torch.complex64
    elif type == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("Unknown float type")


def _dt_set_axes(shape, dim):
    if dim is None:
        return tuple((np.arange(len(shape)).astype(int)))

    if len(shape) > dim:
        return tuple((np.arange(dim).astype(int)) + 1)
    else:
        return tuple((np.arange(dim).astype(int)))


class rfftn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None):
        if dim is not None:
            ctx.save_for_backward(torch.Tensor([dim]).long())
        return torch.fft.rfftn(x, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = None
        if len(ctx.saved_tensors) > 0:
            dim = tuple(ctx.saved_tensors[0][0].long().cpu().tolist())
        grad_output = torch.conj(grad_output)
        grad = torch.fft.irfftn(grad_output, dim=dim, norm="forward")
        return grad, None


class irfftn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None):
        if dim is not None:
            ctx.save_for_backward(torch.Tensor([dim]).long())
        return torch.fft.irfftn(x, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = None
        if len(ctx.saved_tensors) > 0:
            dim = tuple(ctx.saved_tensors[0][0].long().cpu().tolist())
        grad = torch.fft.rfftn(grad_output, dim=dim, norm="forward")
        grad = torch.conj(grad)
        return grad, None


def dft(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True,
        real_in: bool = False
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Fourier transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :param real_in: Input is real. Only returns hermitian half
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = _dt_set_axes(grid.shape, dim)

    if real_in:
        grid_ft = rfftn.apply(torch.fft.fftshift(grid, dim=axes), axes) if use_torch \
            else np.fft.rfftn(np.fft.fftshift(grid, axes=axes), axes=axes)
    else:
        grid_ft = torch.fft.fftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
            else np.fft.fftn(np.fft.fftshift(grid, axes=axes), axes=axes)

    if center:
        grid_ft = torch.fft.fftshift(grid_ft, dim=axes[:-1] if real_in else axes) if use_torch \
            else np.fft.fftshift(grid_ft, axes=axes[:-1] if real_in else axes)

    return grid_ft


def idft(
        grid_ft: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True,
        real_in: bool = False
) -> Union[Tensor, np.ndarray]:
    """
    Inverse Discreet Fourier transform
    :param grid_ft: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param centered: If the zeroth frequency should be centered
    :param real_in: Input is real. Only returns hermitian half
    :return: Inverse transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid_ft)
    axes = _dt_set_axes(grid_ft.shape, dim)

    if centered:
        grid_ft = torch.fft.ifftshift(grid_ft, dim=axes[:-1] if real_in else axes) if use_torch \
            else np.fft.ifftshift(grid_ft, axes=axes[:-1] if real_in else axes)

    if real_in:
        grid = torch.fft.ifftshift(irfftn.apply(grid_ft, axes), dim=axes) if use_torch \
            else np.fft.ifftshift(np.fft.irfftn(grid_ft, axes=axes), axes=axes)
    else:
        grid = torch.fft.ifftshift(torch.fft.ifftn(grid_ft, dim=axes), dim=axes) if use_torch \
            else np.fft.ifftshift(np.fft.ifftn(grid_ft, axes=axes), axes=axes)

    return grid


def dht(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Hartley transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = None if dim is None else _dt_set_axes(grid.shape, dim)

    grid_ht = torch.fft.fftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
        else np.fft.fftn(np.fft.fftshift(grid, axes=axes), axes=axes)

    if center:
        grid_ht = torch.fft.fftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.fftshift(grid_ht, axes=axes)

    return grid_ht.real - grid_ht.imag


def idht(
        grid_ht: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Inverse Discreet Hartley transform
    :param grid_ht: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param centered: If the zeroth frequency should be centered
    :return: Inverse transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid_ht)
    axes = None if dim is None else _dt_set_axes(grid_ht.shape, dim)

    if centered:
        grid_ht = torch.fft.ifftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.ifftshift(grid_ht, axes=axes)

    f = torch.fft.fftshift(torch.fft.fftn(grid_ht, dim=axes), dim=axes) if use_torch \
        else np.fft.fftshift(np.fft.fftn(grid_ht, axes=axes), axes=axes)

    # Adjust for FFT normalization
    if axes is None:
        f /= np.product(f.shape)
    else:
        f /= np.product(np.array(f.shape)[list(axes)])

    return f.real - f.imag


def htToFt(
        grid_ht: Union[Tensor, np.ndarray],
        dim: int = None
):
    """
    Converts a batch of Hartley transforms to Fourier transforms
    :param grid_ht: Batch of Hartley transforms
    :param dim: Data dimension
    :return: The batch of Fourier transforms
    """
    axes = tuple(np.arange(len(grid_ht.shape))) if dim is None else _dt_set_axes(grid_ht.shape, dim)
    dtype = get_complex_float_type(grid_ht.dtype)

    if torch.is_tensor(grid_ht):
        grid_ft = torch.empty(grid_ht.shape, dtype=dtype).to(grid_ht.device)
        grid_ht_ = torch.flip(grid_ht, axes)
        if grid_ht.shape[-1] % 2 == 0:
            grid_ht_ = torch.roll(grid_ht_, [1] * len(axes), axes)
        grid_ft.real = (grid_ht + grid_ht_) / 2
        grid_ft.imag = (grid_ht - grid_ht_) / 2
    else:
        grid_ft = np.empty(grid_ht.shape, dtype=dtype)
        grid_ht_ = np.flip(grid_ht, axes)
        if grid_ht.shape[-1] % 2 == 0:
            grid_ht_ = np.roll(grid_ht_, 1, axes)
        grid_ft.real = (grid_ht + grid_ht_) / 2
        grid_ft.imag = (grid_ht - grid_ht_) / 2

    return grid_ft


def dt_symmetrize(dt: Tensor, dim: int = None) -> Tensor:
    s = dt.shape
    if dim is None:
        dim = 3 if len(s) >= 3 else 2

    if s[-2] % 2 != 0:
        raise RuntimeError("Box size must be even.")

    if dim == 2:
        if s[-1] == s[-2]:
            if len(s) == 2:
                sym_ht = torch.empty((s[0] + 1, s[1] + 1), dtype=dt.dtype).to(dt.device)
            else:
                sym_ht = torch.empty((s[0], s[1] + 1, s[2] + 1), dtype=dt.dtype).to(dt.device)
            sym_ht[..., 0:-1, 0:-1] = dt
            sym_ht[..., -1, :-1] = dt[..., 0, :]
            sym_ht[..., :, -1] = sym_ht[..., :, 0]
        elif s[-1] == s[-2] // 2 + 1:
            if len(s) == 2:
                sym_ht = torch.empty((s[0] + 1, s[1]), dtype=dt.dtype).to(dt.device)
            else:
                sym_ht = torch.empty((s[0], s[1] + 1, s[2]), dtype=dt.dtype).to(dt.device)
            sym_ht[..., 0:-1, :] = dt
            sym_ht[..., -1, :] = dt[..., 0, :]

    elif dim == 3:
        if s[-1] == s[-2]:
            if len(s) == 3:
                sym_ht = torch.empty((s[0] + 1, s[1] + 1, s[2] + 1), dtype=dt.dtype).to(dt.device)
            else:
                sym_ht = torch.empty((s[0], s[1] + 1, s[2] + 1, s[3] + 1), dtype=dt.dtype).to(dt.device)
            sym_ht[..., 0:-1, 0:-1, 0:-1] = dt
            sym_ht[...,   -1,  :-1,  :-1] = sym_ht[..., 0, :-1, :-1]
            sym_ht[...,  :,    :-1,   -1] = sym_ht[..., :, :-1,   0]
            sym_ht[...,  :,    -1,   :  ] = sym_ht[..., :,   0, :  ]
        elif s[-1] == s[-2] // 2 + 1:
            if len(s) == 3:
                sym_ht = torch.empty((s[0] + 1, s[1] + 1, s[2]), dtype=dt.dtype).to(dt.device)
            else:
                sym_ht = torch.empty((s[0], s[1] + 1, s[2] + 1, s[3]), dtype=dt.dtype).to(dt.device)
            sym_ht[..., 0:-1, 0:-1, :] = dt
            sym_ht[..., -1,    :-1, :] = sym_ht[..., 0, :-1, :]
            sym_ht[..., :,      -1, :] = sym_ht[..., :,   0, :]
    else:
        raise RuntimeError("Dimensionality not supported")
    return sym_ht


def dt_desymmetrize(dt: Tensor, dim: int = None) -> Tensor:
    s = dt.shape
    if dim is None:
        dim = 3 if len(s) >= 3 else 2
    if dim == 2:
        if s[-2] == s[-1] * 2 - 1:
            out = dt[..., :-1, :]
            out[..., 0, :] = (dt[..., 0, :] + dt[..., -1, :]) / 2.
        else:
            out = dt[..., :-1, :-1]
            out[..., 0, :] = (dt[..., 0, :-1] + dt[..., -1, :-1]) / 2.
            out[..., :, 0] = (dt[..., :-1, 0] + dt[..., :-1, -1]) / 2.
    elif dim == 3:
        if s[-2] == s[-1] * 2 - 1:
            out = dt[..., :-1, :-1, :]
            out[..., 0, :, :] = (dt[..., 0, :-1, :] + dt[..., -1, :-1, :]) / 2.
            out[..., :, 0, :] = (dt[..., :-1, 0, :] + dt[..., :-1, -1, :]) / 2.
        else:
            out = dt[..., :-1, :-1, :-1]
            out[..., 0, :, :] = (dt[..., 0, :-1, :-1] + dt[..., -1, :-1, :-1]) / 2.
            out[..., :, 0, :] = (dt[..., :-1, 0, :-1] + dt[..., :-1, -1, :-1]) / 2.
            out[..., :, :, 0] = (dt[..., :-1, :-1, 0] + dt[..., :-1, :-1, -1]) / 2.
    else:
        raise RuntimeError("Dimensionality not supported")

    return out


def rdht(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Hartley transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = tuple(np.arange(len(grid.shape))) if dim is None else _dt_set_axes(grid.shape, dim)

    if use_torch:
        grid_ft = torch.fft.rfftn(torch.fft.fftshift(grid, dim=axes), dim=axes)

        grid_ht = torch.empty_like(grid)
        grid_ht[..., :grid_ft.shape[-1]] = grid_ft.real - grid_ft.imag
        hh = torch.flip(grid_ft.real[..., 1:-1] + grid_ft.imag[..., 1:-1], axes)

        hh = torch.roll(hh, [1] * (len(axes) - 1), axes[:-1])
        grid_ht[..., grid_ft.shape[-1]:] = hh
    else:
        grid_ft = np.fft.rfftn(np.fft.fftshift(grid, axes=axes), axes=axes)

        grid_ht = np.empty(grid.shape)
        grid_ht[..., :grid_ft.shape[-1]] = grid_ft.real - grid_ft.imag
        hh = np.flip(grid_ft.real[..., 1:-1] + grid_ft.imag[..., 1:-1], axes)

        hh = np.roll(hh, 1, axis=axes[:-1])
        grid_ht[..., grid_ft.shape[-1]:] = hh

    if center:
        grid_ht = torch.fft.fftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.fftshift(grid_ht, axes=axes)

    return grid_ht


def ird3ht(
        grid_ht: Tensor
) -> Tensor:
    """
    Inverse Discreet Hartley transform, carried out by doing a conversion
    to a Fourier transform and using rfft. Assumes centered!!!
    :param grid_ht: Pytorch tensor with batch of 3D grids to be transformed
    :return: Inverse transformed Pytorch tensor
    """
    s = grid_ht.shape[-1]
    assert len(grid_ht.shape) == 4
    assert s == grid_ht.shape[-3] and s == grid_ht.shape[-2]
    assert s % 2 == 1

    axes = (1, 2, 3)
    grid_ft = htToFt(grid_ht, dim=3)
    grid_ft = grid_ft[..., s // 2:]
    grid_ft = torch.fft.ifftshift(grid_ft, dim=axes)
    grid = torch.fft.ifftshift(torch.fft.irfftn(grid_ft, dim=axes), dim=axes)

    return grid


def fourier_shift_2d(
        grid_ft: Union[Tensor, np.ndarray],
        shift: Union[Tensor, np.ndarray],
        y_shift: Union[Tensor, np.ndarray] = None
) -> Union[Tensor, np.ndarray]:
    """
    Shifts a batch of 2D Fourier transformed images
    :param grid_ft: Batch of Fourier transformed images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D Fourier transformed images
    """
    complex_channels = len(grid_ft.shape) == 4 and grid_ft.shape[-1] == 2
    assert len(grid_ft.shape) == 3 or complex_channels
    assert shift.shape[0] == grid_ft.shape[0]
    s = grid_ft.shape[1]
    symmetrized = s % 2 == 1
    if symmetrized:
        s -= 1

    if y_shift is None:
        assert len(shift.shape) == 2 and shift.shape[1] == 2
        x_shift = shift[..., 0]
        y_shift = shift[..., 1]
    else:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        x_shift = shift
        y_shift = y_shift

    x_shift = x_shift / float(s)
    y_shift = y_shift / float(s)

    if symmetrized:
        ls = torch.linspace(-s // 2, s // 2, s + 1)
    else:
        ls = torch.linspace(-s // 2, s // 2 - 1, s)
    lsx = torch.linspace(0, s // 2, s // 2 + 1)
    y, x = torch.meshgrid(ls, lsx, indexing='ij')
    x = x.to(grid_ft.device)
    y = y.to(grid_ft.device)
    dot_prod = 2 * np.pi * (x[None, :, :] * x_shift[:, None, None] + y[None, :, :] * y_shift[:, None, None])
    a = torch.cos(dot_prod)
    b = torch.sin(dot_prod)

    if complex_channels:
        ar = a * grid_ft[..., 0]
        bi = b * grid_ft[..., 1]
        ab_ri = (a + b) * (grid_ft[..., 0] + grid_ft[..., 1])
        r = ar - bi
        i = ab_ri - ar - bi
        return torch.cat([r.unsqueeze(-1), i.unsqueeze(-1)], -1)
    else:
        ar = a * grid_ft.real
        bi = b * grid_ft.imag
        ab_ri = (a + b) * (grid_ft.real + grid_ft.imag)

        return ar - bi + 1j * (ab_ri - ar - bi)


def grid_spectral_sum(grid, indices):
    if len(grid.shape) == len(indices.shape) and np.all(grid.shape == indices.shape):  # Has no batch dimension
        spectrum = torch.zeros(int(torch.max(indices)) + 1).to(grid.device)
        spectrum.scatter_add_(0, indices.long().flatten(), grid.flatten())
    elif len(grid.shape) == len(indices.shape) + 1 and np.all(grid.shape[1:] == indices.shape):  # Has batch dimension
        spectrum = torch.zeros([grid.shape[0], int(torch.max(indices)) + 1]).to(grid.device)
        indices = indices.long().unsqueeze(0).expand([grid.shape[0]] + list(indices.shape))
        spectrum.scatter_add_(1, indices.flatten(1), grid.flatten(1))
    else:
        raise RuntimeError("Shape of grid must match spectral_indices, except along the batch dimension.")
    return spectrum


def grid_spectral_average(grid, indices):
    indices = indices.long()
    spectrum = grid_spectral_sum(grid, indices)
    norm = grid_spectral_sum(torch.ones_like(indices).float(), indices)
    if len(spectrum.shape) == 2:  # Batch dimension
        return spectrum / norm[None, :]
    else:
        return spectrum / norm


def spectra_to_grid(spectra, indices):
    if len(spectra.shape) == 1:  # Has no batch dimension
        grid = torch.gather(spectra, 0, indices.flatten().long())
    elif len(spectra.shape) == 2:  # Has batch dimension
        indices = indices.unsqueeze(0).expand([spectra.shape[0]] + list(indices.shape))
        grid = torch.gather(spectra.flatten(1), 1, indices.flatten(1).long())
    else:
        raise RuntimeError("Spectra must be at most two-dimensional (one batch dimension).")
    return grid.view(indices.shape)


def spectral_correlation(grid1, grid2, indices, normalize=False, norm_eps=1e-12):
    if np.any(grid1.shape != grid2.shape):
        print('The grids have to be the same shape')
    correlation = torch.real(grid1 * torch.conj(grid2))

    if normalize:
        correlation = grid_spectral_sum(correlation, indices)
        norm1 = grid_spectral_sum(grid1.abs().square(), indices)
        norm2 = grid_spectral_sum(grid2.abs().square(), indices)
        return correlation / ((norm1 * norm2).sqrt() + norm_eps)
    else:
        return grid_spectral_average(correlation, indices)


def get_spectral_indices(shape: Union[Tuple[int, int], Tuple[int, int, int]], centered=True, device='cpu'):
    h_sym = shape[-2] == shape[-1]  # Hermitian symmetric half included
    dim_2 = len(shape) == 2

    if shape[0] % 2 == 0:
        ls = torch.linspace(-(shape[0] // 2), shape[0] // 2 - 1, shape[0], device=device)
    else:
        ls = torch.linspace(-(shape[0] // 2), shape[0] // 2, shape[0], device=device)
    x_ls = ls if h_sym else torch.linspace(0, shape[-1] - 1, shape[-1], device=device)

    if dim_2:
        assert shape[1] == shape[0] or \
               (shape[1] - 1) * 2 == shape[0] or \
               (shape[1] - 1) * 2 + 1 == shape[0]
        r2 = torch.sum(torch.stack(torch.meshgrid(ls, x_ls, indexing="ij"), -1).square(), -1)
        indices = torch.floor(r2.sqrt()).long()
    else:
        assert shape[2] == shape[1] == shape[0] or \
               (shape[2] - 1) * 2 == shape[1] == shape[0] or \
               (shape[2] - 1) * 2 + 1 == shape[1] == shape[0]
        r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, x_ls, indexing="ij"), -1).square(), -1)
        indices = torch.floor(r2.sqrt()).long()

    if not centered:
        if h_sym:
            indices = torch.fft.ifftshift(indices)
        else:
            if dim_2:
                indices = torch.fft.ifftshift(indices, dim=0)
            else:
                indices = torch.fft.ifftshift(indices, dim=(0, 1))
    return indices


def spectral_resolution(ft_size, voxel_size):
    """
    Get list of inverted resolutions (1/Angstroms) for each spectral index in a Fourier transform.
    """
    res = torch.zeros(ft_size)
    res[1:] = torch.arange(1, ft_size) / (2 * voxel_size * ft_size)
    return res


def spectral_index_from_resolution(resolution: float, image_size: int, pixel_size: float):
    """
    Get spectral index from resolution in Angstroms
    """
    return round(image_size * pixel_size / resolution)


def resolution_from_spectral_index(index: int, image_size: int, pixel_size: float):
    """
    Get the resolution in Angstroms.
    """
    return pixel_size * image_size / float(index)


def resolution_from_fsc(fsc, res, threshold=0.5):
    """
    Get the resolution (res) at the FSC (fsc) threshold.
    """
    assert len(fsc) == len(res)
    if torch.is_tensor(fsc):
        i = torch.argmax(fsc < threshold)
    else:
        i = np.argmax(fsc < threshold)
    if i > 0:
        return res[i - 1]
    else:
        return res[0]


def get_fsc_fourier(grid1_df, grid2_df):
    indices = get_spectral_indices(grid1_df.shape, centered=True)
    fsc = spectral_correlation(grid1_df, grid2_df, indices, normalize=True)
    return fsc[:grid1_df.shape[-1]]


def get_fsc_real(grid1, grid2):
    grid1_df = dft(grid1, dim=3, center=True, real_in=True)
    grid2_df = dft(grid2, dim=3, center=True, real_in=True)
    return get_fsc_fourier(grid1_df, grid2_df)


def get_power_fourier(grid_df):
    indices = get_spectral_indices(grid_df.shape, centered=True)
    grid_df = grid_df.abs().square()
    return grid_spectral_average(grid_df, indices)


def get_power_real(grid):
    grid_df = dft(grid, dim=3, center=True, real_in=True)
    return get_power_fourier(grid_df)


def rescale_fourier(grid, out_sz):
    if out_sz % 2 != 0:
        raise Exception("Bad output size")
    if out_sz == grid.shape[0]:
        return grid

    use_torch = torch.is_tensor(grid)

    if len(grid.shape) == 2:
        if grid.shape[0] != (grid.shape[1] - 1) * 2:
            raise Exception("Input must be cubic")

        if use_torch:
            g = torch.zeros((out_sz, out_sz // 2 + 1), device=grid.device, dtype=grid.dtype)
        else:
            g = torch.zeros((out_sz, out_sz // 2 + 1), dtype=grid.dtype)
        i = np.array(grid.shape) // 2
        o = np.array(g.shape) // 2

        if o[0] < i[0]:
            g = grid[i[0] - o[0]: i[0] + o[0], :g.shape[1]]
        elif o[0] > i[0]:
            g[o[0] - i[0]: o[0] + i[0], :grid.shape[1]] = grid
    elif len(grid.shape) == 3:
        if grid.shape[0] != grid.shape[1] or \
                grid.shape[1] != (grid.shape[2] - 1) * 2:
            raise Exception("Input must be cubic")
        if use_torch:
            g = torch.zeros((out_sz, out_sz, out_sz // 2 + 1), device=grid.device, dtype=grid.dtype)
        else:
            g = torch.zeros((out_sz, out_sz, out_sz // 2 + 1), dtype=grid.dtype)
        i = np.array(grid.shape) // 2
        o = np.array(g.shape) // 2

        if o[0] < i[0]:
            g = grid[i[0] - o[0]: i[0] + o[0], i[1] - o[1]: i[1] + o[1], :g.shape[2]]
        elif o[0] > i[0]:
            g[o[0] - i[0]: o[0] + i[0], o[1] - i[1]: o[1] + i[1], :grid.shape[2]] = grid
    else:
        raise RuntimeError("Only 2D and 3D tensors supported.")

    return g


def rescale_real(grid, out_sz):
    grid_ft = dft(grid, center=True, real_in=True)
    grid_ft = rescale_fourier(grid_ft, out_sz)
    return idft(grid_ft, centered=True, real_in=True)


def rescale_from_voxel_size(shape: Union[list, tuple], voxel_size: float, target_voxel_size: float) -> int:
    (iz, iy, ix) = shape

    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * voxel_size / target_voxel_size))
    if out_sz % 2 != 0:
        vs1 = voxel_size * in_sz / (out_sz + 1)
        vs2 = voxel_size * in_sz / (out_sz - 1)
        if np.abs(vs1 - target_voxel_size) < np.abs(vs2 - target_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    return voxel_size * in_sz / out_sz
