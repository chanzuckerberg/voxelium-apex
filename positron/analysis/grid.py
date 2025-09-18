#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import sys
import os

import argparse
import glob
import pickle
import time

import scipy.stats as stats
import numpy as np

import matplotlib.pylab as plt

import torch
import torch.nn.functional as F

from positron.base import load_mrc, save_mrc


def load_mrc_(filename):
    grid, voxel_size, global_origin = load_mrc(filename)
    grid = torch.from_numpy(grid.copy())
    return grid, voxel_size, global_origin


def main(args):

    if not os.path.exists(args.input):
        print(f"File '{args.input}' not found")
        return

    grid, voxel_size, global_origin = load_mrc_(args.input)

    if args.stats:
        print(f"Statistics for {args.input}")
        print(f"Pixel size: {voxel_size}")
        print(f"Origin: {global_origin}")
        print(f"Mean: {grid.mean()}")
        print(f"Std: {grid.std()}")
        print(f"Min: {grid.min()}")
        print(f"Max: {grid.max()}")
        print(f"Median: {np.median(grid.detach().cpu().numpy())}")

    modifier = False
    if args.add is not None:
        modifier = True
        print(f"Adding {args.add}")
        add, _, _ = load_mrc_(args.add)
        grid = grid + add
    if args.sub is not None:
        modifier = True
        print(f"Subtracting {args.sub}")
        sub, _, _ = load_mrc_(args.sub)
        grid = grid - sub
    if args.mul is not None:
        modifier = True
        print(f"Multiplying by {args.mul}")
        mul, _, _ = load_mrc_(args.mul)
        grid = grid * mul
    if args.div is not None:
        modifier = True
        print(f"Dividing by {args.div}")
        divide, _, _ = load_mrc_(args.div)
        grid = grid / divide
    if args.addc is not None:
        modifier = True
        print(f"Adding constant {args.addc}")
        grid = grid + args.addc
    if args.subc is not None:
        modifier = True
        print(f"Subtracting constant {args.subc}")
        grid = grid - args.subc
    if args.mulc is not None:
        modifier = True
        print(f"Multiplying by constant {args.mulc}")
        grid = grid * args.mulc
    if args.divc is not None:
        if args.divc == 0:
            raise ValueError("Cannot divide by zero")
        modifier = True
        print(f"Dividing by constant {args.divc}")
        grid = grid / args.divc
    
    if modifier:
        if args.output is None:
            output = None
            for i in range(999):
                output = args.input.replace(".mrc", f"_mod{i}.mrc")
                if not os.path.exists(output):
                    break
            if output is None:
                raise ValueError("No output file found")
            args.output = output

        print(f"Saving to {args.output}")
        save_mrc(
            grid.detach().cpu().numpy().astype(np.float32), 
            args.output,
            voxel_size=voxel_size,
            origin=global_origin
        )


def append_args(parser):
    parser.add_argument("-i", "--input", type=str, help="Input file", required=True)


    parser.add_argument(
        "--stats", "--statistics",
        help="Print statistics for input",
        action="store_true"
    )

    parser.add_argument(
        "-a", "--add", "--addition",
        type=str,
        help="Map to add to input",
        default=None
    )
    parser.add_argument(
        "-s", "--sub", "--subtract",
        type=str,
        help="Map to subtract from input",
        default=None
    )
    parser.add_argument(
        "-m", "--mul", "--multiply",
        type=str,
        help="Map to multiply with",
        default=None
    )
    parser.add_argument(
        "-d", "--div", "--divide",
        type=str,
        help="Map to divide input by",
        default=None
    )


    parser.add_argument(
        "--addc", "--add_constant",
        type=float,
        help="Add constant to input",
        default=None
    )
    parser.add_argument(
        "--subc", "--sub_constant", "--subtract_constant",
        type=float,
        help="Subtract constant from input",
        default=None
    )
    parser.add_argument(
        "--mulc", "--mul_constant", "--multiply_constant",
        type=float,
        help="Multiply input by constant",
        default=None
    )
    parser.add_argument(
        "--divc", "--div_constant", "--divide_constant",
        type=float,
        help="Divide input by constant",
        default=None
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file",
        default=None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Used to plot or get a CSV of spectral statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)
