#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import sys
import os

import argparse
import glob
import pickle
import time

import numpy as np

import matplotlib
import matplotlib.pylab as plt
import torch

from positron.base import load_mrc, spherical_mask, spectral_resolution, get_fsc_real, get_power_real, \
    spectral_index_from_resolution, idft, local_correlation, save_mrc


def load_mrc_(filename):
    grid, voxel_size, global_origin = load_mrc(filename)
    grid = torch.from_numpy(grid.copy())
    return grid, voxel_size, global_origin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Output the local correlation between two grids")
    parser.add_argument("input1", type=str)
    parser.add_argument("input2", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("-k", "--kernel-size", type=int, default=3, help="Width of averaging windows.")
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU to use")
    parser.add_argument("-m", "--spherical_mask", action="store_true", help="Apply a spherical mask first.")
    args = parser.parse_args()

    torch.no_grad()

    grid1, voxel_size, global_origin = load_mrc_(args.input1)
    grid2, _, _ = load_mrc_(args.input2)

    if args.spherical_mask:
        mask = spherical_mask(grid1.shape[-1], 0.9)
        grid1 *= mask
        grid1 *= mask

    if args.gpu is not None:
        grid1 = grid1.to(f"cuda:{args.gpu}")
        grid2 = grid2.to(f"cuda:{args.gpu}")

    corr = local_correlation(grid1, grid2, kernel_size=args.kernel_size).cpu().numpy()

    save_mrc(corr, args.output, voxel_size=voxel_size, origin=global_origin)
