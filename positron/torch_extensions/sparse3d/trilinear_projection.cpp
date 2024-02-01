#include "torch_extensions/sparse3d/trilinear_projection.h"

torch::Tensor trilinear_projection_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rot_matrix,
    torch::Tensor grid2d_coord,
    torch::Tensor grid3d_index,
    const int max_r
)
{
    const int batch_size = input.size(0);
    const bool do_bias = bias.size(0) == weight.size(0);

    CHECK_SIZE_DIM1(grid2d_coord, 2)
    CHECK_SIZE_DIM0(rot_matrix, batch_size)
    CHECK_SIZE_DIM1(rot_matrix, 3)
    CHECK_SIZE_DIM2(rot_matrix, 3)

    auto output = torch::zeros(
        {batch_size, grid2d_coord.size(0), 2},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(true)
    );

    if (input.device().type() == torch::kCPU)
    {
        trilinear_projection_forward_cpu(
            /*grid2d_coord*/ grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*rot_matrix*/ rot_matrix,
            /*input*/ input,
            /*output*/ output,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias
        );
    }
    else if (input.device().type() == torch::kCUDA)
    {
        trilinear_projection_forward_cuda(
            /*grid2d_coord*/ grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*rot_matrix*/ rot_matrix,
            /*input*/ input,
            /*output*/ output,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias
        );
    }
    else
        throw std::logic_error("Support for device not implemented");

    return output;
}


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
)
{
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int points_count = grid2d_coord.size(0);
    const int basis_size = weight.size(0);
    const bool do_bias = bias.size(0) == basis_size;

    CHECK_SIZE_DIM0(grid2d_coord, points_count)
    CHECK_SIZE_DIM1(grid2d_coord, 2)
    CHECK_SIZE_DIM0(rot_matrix, batch_size)
    CHECK_SIZE_DIM1(rot_matrix, 3)
    CHECK_SIZE_DIM2(rot_matrix, 3)

    CHECK_DTYPE(weight, input.dtype())
    CHECK_DTYPE(bias, input.dtype())
    CHECK_DTYPE(rot_matrix, input.dtype())
    CHECK_DTYPE(grad_output, input.dtype())
    CHECK_DTYPE(grid2d_coord, input.dtype())

    torch::Tensor grad_weight, grad_bias, backprop_weight;

    grad_weight = torch::zeros_like(weight);
    grad_bias = torch::zeros_like(bias);

    if (return_backprop_weight)
        backprop_weight = torch::zeros(
            basis_size,
            torch::TensorOptions()
                .dtype(input.dtype())
                .layout(torch::kStrided)
                .device(input.device())
                .requires_grad(false)
        );
    else
        backprop_weight = torch::empty(0,
            torch::TensorOptions()
                .dtype(input.dtype())
                .layout(torch::kStrided)
                .device(input.device())
                .requires_grad(false)
        );

    auto grad_input = torch::zeros(
        {batch_size, input_size},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(false)
    );

    auto grad_rot_matrix = torch::zeros(
        {batch_size, 3, 3},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(false)
    );

    if (input.device().type() == torch::kCPU)
    {
        trilinear_projection_backward_cpu(
            /*grid2d_coord*/grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*rot_matrix*/ rot_matrix,
            /*input*/ input,
            /*grad_output*/ grad_output,
            /*grad_weight*/ grad_weight,
            /*grad_bias*/ grad_bias,
            /*grad_input*/ grad_input,
            /*grad_rot_matrix*/ grad_rot_matrix,
            /*backprop_weight*/ backprop_weight,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias,
            /*do_rot_matrix_grad*/ rot_matrix.requires_grad(),
            /*return_backprop_weight*/ return_backprop_weight
        );
    }
    else if (input.device().type() == torch::kCUDA)
    {
        trilinear_projection_backward_cuda(
            /*grid2d_coord*/grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*rot_matrix*/ rot_matrix,
            /*input*/ input,
            /*grad_output*/ grad_output,
            /*grad_weight*/ grad_weight,
            /*grad_bias*/ grad_bias,
            /*grad_input*/ grad_input,
            /*grad_rot_matrix*/ grad_rot_matrix,
            /*backprop_weight*/ backprop_weight,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias,
            /*do_rot_matrix_grad*/ rot_matrix.requires_grad(),
            /*return_backprop_weight*/ return_backprop_weight
        );
    }
    else
        throw std::logic_error("Support for device not implemented");

    return {grad_input, grad_weight, grad_bias, backprop_weight, grad_rot_matrix};
}
