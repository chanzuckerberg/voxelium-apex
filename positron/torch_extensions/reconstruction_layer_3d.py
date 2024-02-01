#!/usr/bin/env python3

"""
Python API for the 3D reconstruction layer
"""
import sys
import time
from typing import TypeVar

import numpy as np
import torch

from positron.base import grid_iterator, dt_desymmetrize, dt_symmetrize
from positron.base.explicit_grid_utils import radial_index_expansion_3d, size_to_maxr

try:
    import positron_sparse3d
except ImportError:
    print("Could not find Positron extension 'sparse3d'.")
    sys.exit(1)


class ReconstructionLayer3D(torch.nn.Module):
    def __init__(
            self,
            size,
            input_size,
            dtype=torch.float32,
            do_bias=True,
            index_margin=3,
            init_basis=True
    ):
        super().__init__()

        if size % 2 == 0:
            size += 1

        self.size = size
        self.size_x = size // 2 + 1
        self.input_size = input_size
        self.index_margin = index_margin
        self.maxr = size_to_maxr(size)
        self.do_bias = do_bias
        self.dtype = dtype

        self.weight_count = None
        self.grid3d_mask = None
        self.grid3d_index = None
        self.inverse_grid3d_indices = None
        self.weight = None
        self.bias = None

        if init_basis:
            self.init_basis()

    def init_from(self, other, clone=False):
        self.weight_count = other.weight_count
        self.grid3d_mask = other.grid3d_mask
        self.grid3d_index = other.grid3d_index
        self.inverse_grid3d_indices = other.inverse_grid3d_indices

        weight = other.weight.data.clone() if clone else other.weight.data
        self.weight = torch.nn.Parameter(data=weight, requires_grad=True)
        if self.do_bias:
            bias = other.bias.data.clone() if clone else other.bias.data
            self.bias = torch.nn.Parameter(data=bias, requires_grad=True)

    def clone(self):
        clone = ReconstructionLayer3D(
            size=self.size,
            input_size=self.input_size,
            dtype=self.dtype,
            do_bias=self.do_bias,
            index_margin=self.index_margin,
            init_basis=False
        )
        clone.init_from(other=self, clone=True)
        return clone

    def init_basis(self):
        bz = self.size
        bz_2 = bz // 2
        bz_x = bz_2 + 1
        grid_mask = np.zeros((bz, bz, bz_x), dtype=bool)
        grid_indices = np.zeros((bz, bz, bz_x), dtype=int) - 1
        inverse_grid_indices = np.zeros((bz * bz * bz_x), dtype=int)
        max_r2 = size_to_maxr(self.size) ** 2
        i = 0
        # j = 0
        # import zCurve
        # while True:
        #     x, y, z = zCurve.deinterlace(j, dims=3)
        #     if x >= 2*bz_x and y >= 2*bz and z >= 2*bz:
        #         break
        #     j += 1
        #     if (z - bz_2) ** 2 + (y - bz_2) ** 2 + x ** 2 < max_r2:
        #         grid_mask[z, y, x] = True
        #         grid_indices[z, y, x] = i
        #         inverse_grid_indices[i] = z * bz * bz_x + y * bz_x + x
        #         i += 1
        for z, y, x in grid_iterator(bz, bz, bz_x):
            if (z - bz_2) ** 2 + (y - bz_2) ** 2 + x ** 2 < max_r2:
                grid_mask[z, y, x] = True
                grid_indices[z, y, x] = i
                inverse_grid_indices[i] = z * bz * bz_x + y * bz_x + x
                i += 1

        self.weight_count = i
        inverse_grid_indices = inverse_grid_indices[:i]

        # Add margin for pixel spread into voxels
        m = self.index_margin
        bz += m * 2
        bz_2 += m
        grid_indices_margin = np.zeros((bz, bz, bz_2 + 1), dtype=int) - 1
        grid_indices_margin[m:-m, m:-m, :-m] = grid_indices
        grid_indices = grid_indices_margin

        for i in range(5):
            radial_index_expansion_3d(grid_indices)

        self.grid3d_mask = torch.nn.Parameter(
            torch.tensor(grid_mask, dtype=torch.bool), requires_grad=False)

        self.grid3d_index = torch.nn.Parameter(
            torch.tensor(grid_indices, dtype=torch.long), requires_grad=False)

        self.inverse_grid3d_indices = torch.nn.Parameter(
            torch.tensor(inverse_grid_indices, dtype=torch.long), requires_grad=False)

        data_tensor = torch.zeros((self.weight_count, self.input_size, 2), dtype=self.dtype)
        self.weight = torch.nn.Parameter(data=data_tensor, requires_grad=True)

        if self.do_bias:
            data_tensor = torch.zeros((self.weight_count, 2), dtype=self.dtype)
            self.bias = torch.nn.Parameter(data=data_tensor, requires_grad=True)

    def forward(self, input, max_r=None, grid2d_coord=None, rot_matrices=None, backprop_eps=True):
        max_r = self.maxr if max_r is None else min(max_r, self.maxr)

        if rot_matrices is not None and grid2d_coord is not None:
            return TrilinearProjection.apply(
                input,  # input
                self.weight,  # weight
                self.bias,  # bias
                self.grid3d_index,  # grid3d_index
                rot_matrices,  # rot_matrices
                grid2d_coord,  # grid2d_coord
                max_r,  # max_r
                backprop_eps,  # backprop_eps
                False  # testing
            )
        else:
            return VolumeExtraction.apply(
                input,  # input
                self.weight,  # weight
                self.bias,  # bias
                self.grid3d_index,  # grid3d_index
                max_r  # max_r
            )

    @torch.no_grad()
    def set_base(self, grid: torch.tensor, index: int = None, symmetrize: bool = True):
        """
        Set implicit indexed 3D DFT grid into base with provided index.
        If index is None, assumes bias.
        """
        if symmetrize:
            grid = dt_symmetrize(grid)

        if index is None:
            self.bias[:] = torch.view_as_real(grid.flatten()[self.inverse_grid3d_indices])
        else:
            self.weight[:, index] = torch.view_as_real(grid.flatten()[self.inverse_grid3d_indices])

    @torch.no_grad()
    def get_base(self, index=None, desymmetrize: bool = True):
        """
        Get implicit indexed 3D DFT grid of base with provided index.
        If index is None, assumes bias.
        """
        m = self.index_margin
        indices = self.grid3d_index[m:-m, m:-m, :-m]

        if index is None:
            grid = self.bias[indices]
        else:
            grid = self.weight[indices, index]
        grid_ = torch.view_as_complex(grid)
        grid = torch.zeros_like(grid_)
        grid[self.grid3d_mask] = grid_[self.grid3d_mask]
        if desymmetrize:
            grid = dt_desymmetrize(grid)

        return grid


class TrilinearProjection(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, input, weight, bias, grid3d_index,
            rot_matrices, grid2d_coord, max_r,
            backprop_eps=0, testing=False
    ):
        assert grid3d_index.shape[0] == grid3d_index.shape[1] == grid3d_index.shape[2] * 2 - 1

        output = positron_sparse3d.trilinear_projection_forward(
            input=input,
            weight=weight,
            bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
            rot_matrix=rot_matrices,
            grid2d_coord=grid2d_coord,
            grid3d_index=grid3d_index,
            max_r=max_r
        )

        ctx.save_for_backward(
            input,
            weight,
            bias,
            grid3d_index,
            rot_matrices,
            grid2d_coord,
            torch.Tensor([max_r]),
            torch.Tensor([backprop_eps]),
            torch.Tensor([testing])
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grid3d_index, rot_matrices, grid2d_coord, \
            max_r, backprop_eps, testing \
            = ctx.saved_tensors

        grad_input, grad_weight, grad_bias, backprop_weight, grad_rot_matrix = \
            positron_sparse3d.trilinear_projection_backward(
                input=input,
                grid2d_grad=grad_output.contiguous(),
                weight=weight,
                bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
                grid3d_index=grid3d_index,
                rot_matrix=rot_matrices,
                grid2d_coord=grid2d_coord,
                max_r=max_r[0],
                return_backprop_weight=backprop_eps[0] > 0
            )

        if bias is None:
            grad_bias = None

        if backprop_eps[0] > 0:
            backprop_weight.add_(backprop_eps[0])
            grad_weight.div_(backprop_weight[:, None, None])
            if grad_bias is not None:
                grad_bias.div_(backprop_weight[:, None])

        return grad_input, grad_weight, grad_bias, None, grad_rot_matrix, None, None, None, None, None


class VolumeExtraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, grid3d_index, max_r=None):

        assert grid3d_index.shape[0] == grid3d_index.shape[1] == grid3d_index.shape[2] * 2 - 1

        output = positron_sparse3d.volume_extraction_forward(
            input=input,
            weight=weight,
            bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
            grid3d_index=grid3d_index,
            max_r=max_r
        )
        ctx.save_for_backward(input, weight, bias, grid3d_index)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grid3d_index = ctx.saved_tensors

        grad_input, grad_weight, grad_bias = \
            positron_sparse3d.volume_extraction_backward(
                input=input,
                weight=weight,
                bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
                grad_output=grad_output,
                grid3d_index=grid3d_index
            )

        if bias is None:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None
