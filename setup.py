#!/usr/bin/env python

"""
Setup module for Positron
"""

import os
import sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import positron


def print_debug_msg():
    print("-------------------------------------- ")
    print("------------- DEBUG MODE ------------- ")
    print("-------------------------------------- ")

nvcc_architectures = ["61", "70", "75", "80", "86", "87", "89", "90"]

debug = False
_DEBUG_LEVEL = os.environ.get('POSITRON_DEBUG', '0')
if len(os.environ.get('POSITRON_DEBUG', '')) > 0:
    debug = True

build_extensions = True
if len(os.environ.get('POSITRON_SKIP_EXT', '')) > 0:
    build_extensions = False

sys.path.insert(0, f'{os.path.dirname(__file__)}/positron')

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)),  "positron")

include_dirs = [project_root]

cxx_extra_compile_args = []
nvcc_extra_compile_args = []

for arch in nvcc_architectures:
    nvcc_extra_compile_args += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]

if debug:
    print_debug_msg()
    cxx_extra_compile_args += ["-g", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
    nvcc_extra_compile_args += ["-G", "-lineinfo"]
else:
    cxx_extra_compile_args += ["-DNDEBUG", "-O3"]
nvcc_extra_compile_args += cxx_extra_compile_args


if build_extensions:
    ext_modules = [
        CUDAExtension(
            name='positron_sparse3d',
            sources=[
                'positron/torch_extensions/sparse3d/pybind.cpp',
                'positron/torch_extensions/sparse3d/trilinear_projection.cpp',
                'positron/torch_extensions/sparse3d/trilinear_projection_cpu_kernels.cpp',
                'positron/torch_extensions/sparse3d/trilinear_projection_cuda_kernels.cu',
                'positron/torch_extensions/sparse3d/volume_extraction.cpp',
                'positron/torch_extensions/sparse3d/volume_extraction_cpu_kernels.cpp',
                'positron/torch_extensions/sparse3d/volume_extraction_cuda_kernels.cu',
            ],
            include_dirs=include_dirs,
            extra_compile_args={'cxx': cxx_extra_compile_args, 'nvcc': nvcc_extra_compile_args},
        )
    ]
else:
    ext_modules = None

setup(
    name='Positron',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "positron = positron.__main__:main",
        ],
    },
    version=positron.__version__
)

if debug:
    print_debug_msg()
