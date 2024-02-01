#pragma once
#ifndef SPARSE_3D_VOLUME_EXTRACTION_H
#define SPARSE_3D_VOLUME_EXTRACTION_H

#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "torch_extensions/sparse3d/volume_extraction_cpu_kernels.h"
#include "torch_extensions/sparse3d/volume_extraction_cuda_kernels.h"

torch::Tensor volume_extraction_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grid3d_index,
    const int max_r
);

std::vector<torch::Tensor> volume_extraction_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grad_output,
    torch::Tensor grid3d_index
);

#endif // SPARSE_3D_VOLUME_EXTRACTION_H