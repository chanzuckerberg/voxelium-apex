#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "torch_extensions/sparse3d/trilinear_projection.h"
#include "torch_extensions/sparse3d/volume_extraction.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "trilinear_projection_forward",
        &trilinear_projection_forward,
        "Trilinear projector forward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("rot_matrix"),
        py::arg("grid2d_coord"),
        py::arg("grid3d_index"),
        py::arg("max_r")
    );

    m.def(
        "trilinear_projection_backward",
        &trilinear_projection_backward,
        "Trilinear projector backward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("rot_matrix"),
        py::arg("grid2d_grad"),
        py::arg("grid2d_coord"),
        py::arg("grid3d_index"),
        py::arg("return_backprop_weight"),
        py::arg("max_r")
    );

    m.def(
        "volume_extraction_forward",
        &volume_extraction_forward,
        "Volume extraction forward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("grid3d_index"),
        py::arg("max_r")
    );

    m.def(
        "volume_extraction_backward",
        &volume_extraction_backward,
        "Volume extraction backward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("grad_output"),
        py::arg("grid3d_index")
    );
}
