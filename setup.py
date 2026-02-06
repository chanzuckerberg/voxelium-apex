#!/usr/bin/env python

"""
Setup module for Positron
"""

import os
import sys

from setuptools import setup

import positron

setup(
    name='Voxelium-Apex',
    entry_points={
        "console_scripts": [
            "positron = positron.__main__:main",
        ],
    },
    version=positron.__version__
)

