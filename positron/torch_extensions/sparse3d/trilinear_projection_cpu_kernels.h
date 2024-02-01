#pragma once
#ifndef SPARSE_3D_TRILINEAR_PROJECTION_CPU_KERNELS_H
#define SPARSE_3D_TRILINEAR_PROJECTION_CPU_KERNELS_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "base/base.h"

void trilinear_projection_forward_cpu(
    const torch::Tensor grid2d_coord,
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor rot_matrix,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const int init_offset,
    const bool do_bias
);

void trilinear_projection_backward_cpu(
    const torch::Tensor grid2d_coord,
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor rot_matrix,
    const torch::Tensor input,
    const torch::Tensor grad_output,
    torch::Tensor grad_weight,
    torch::Tensor grad_bias,
    torch::Tensor grad_input,
    torch::Tensor grad_rot_matrix,
    torch::Tensor backprop_weight,
    const int max_r2,
    const int init_offset,
    const bool do_bias,
    const bool do_grad_rot_matrix,
    const bool return_backprop_weight
);

#endif // SPARSE_3D_TRILINEAR_PROJECTION_CPU_KERNELS_H