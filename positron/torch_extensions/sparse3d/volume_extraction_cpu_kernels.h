#pragma once
#ifndef SPARSE_3D_VOLUME_EXTRACTION_CPU_KERNELS_H
#define SPARSE_3D_VOLUME_EXTRACTION_CPU_KERNELS_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "base/base.h"

void volume_extraction_forward_cpu(
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const bool do_bias
);

void volume_extraction_backward_cpu(
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor input,
    const torch::Tensor grad_output,
    torch::Tensor grad_weight,
    torch::Tensor grad_bias,
    torch::Tensor grad_input,
    const int max_r2,
    const bool do_bias,
    const bool do_input_grad
);

#endif // SPARSE_3D_VOLUME_EXTRACTION_CPU_KERNELS_H