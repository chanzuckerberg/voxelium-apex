#pragma once
#ifndef SPARSE_3D_TRILINEAR_PROJECTION_H
#define SPARSE_3D_TRILINEAR_PROJECTION_H

#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "torch_extensions/sparse3d/trilinear_projection_cpu_kernels.h"
#include "torch_extensions/sparse3d/trilinear_projection_cuda_kernels.h"

torch::Tensor trilinear_projection_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rot_matrix,
    torch::Tensor grid2d_coord,
    torch::Tensor grid3d_index,
    const int max_r
);

std::vector<torch::Tensor> trilinear_projection_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rot_matrix,
    torch::Tensor grad_output,
    torch::Tensor grid2d_coord,
    torch::Tensor grid3d_index,
    bool return_backprop_weight,
    const int max_r
);

#endif // SPARSE_3D_TRILINEAR_PROJECTION_H