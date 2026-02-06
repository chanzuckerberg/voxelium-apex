#!/usr/bin/env python3
"""
setup.py

This file exists only to conditionally build optional torch extensions.
All package metadata (name, dependencies, entry points, etc.) should live in pyproject.toml.

To skip building torch extensions (e.g. local visualization-only installs):
    APEX_SKIP_EXT=TRUE pip install .

Torch extensions are built by default.
"""

import os
from setuptools import setup


def _truthy_env(var_name: str) -> bool:
    v = os.environ.get(var_name, "")
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


SKIP_EXT = _truthy_env("APEX_SKIP_EXT")

ext_modules = []
cmdclass = {}

if SKIP_EXT:
    print("APEX_SKIP_EXT is set: skipping torch extension build.")
else:
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        # from torch.utils.cpp_extension import CUDAExtension
    except Exception as e:
        raise RuntimeError(
            "Torch is required to build the optional extensions. "
            "Either install torch first, or set APEX_SKIP_EXT=TRUE to skip building extensions.\n"
            f"Original error: {e}"
        )

    # ---------------------------------------------------------------------
    # TODO: Define your actual extension(s) here.
    # See previous message for examples.
    # ---------------------------------------------------------------------

    ext_modules = []

    if ext_modules:
        cmdclass = {"build_ext": BuildExtension}


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
