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

from voxelium import load_mrc, spherical_mask, spectral_resolution, get_fsc_real, get_power_real, \
    spectral_index_from_resolution, idft


def load_mrc_(filename):
    grid, voxel_size, global_origin = load_mrc(filename)
    grid = torch.from_numpy(grid.copy())
    return grid, voxel_size, global_origin


def main(args):
    if not args.fsc and not args.powers:
        print("No option selected to calculate.")

    solvent_mask = None
    if args.mask is not None:
        solvent_mask, _, _ = load_mrc_(args.mask)

    mrc_filenames = args.mrcs
    voxel_size = args.voxel_size

    if args.fsc:
        reference = None
        mask = None
        fscs = []

        print("Processing:")
        for i in range(len(mrc_filenames)):
            print(f" {mrc_filenames[i]}")

            grid, voxel_size_, _ = load_mrc_(mrc_filenames[i])
            if voxel_size is None:
                voxel_size = voxel_size_ if voxel_size_ > 0 else 1.

            if reference is None:
                if solvent_mask is None:
                    mask = spherical_mask(grid.shape[0]) if args.spherical_mask else 1
                else:
                    mask = solvent_mask
                reference = grid * mask
            else:
                fsc = get_fsc_real(reference, grid * mask)
                fscs.append(fsc)

        resolution = spectral_resolution(fscs[0].shape[0], voxel_size)
        if args.max_resolution is not None:
            max_index = spectral_index_from_resolution(args.max_resolution, reference.shape[0], voxel_size)
            resolution = resolution[:max_index]
            for i in range(len(fscs)):
                fscs[i] = fscs[i][:max_index]

        if args.csv is not None:
            out = np.zeros((len(resolution), len(fscs) + 1))
            out[:, 0] = resolution
            header = "resolution"
            for i in range(len(fscs)):
                out[:, i + 1] = fscs[i]
                header += f",{mrc_filenames[i + 1]}"
            np.savetxt(args.csv, out, delimiter=', ', header=header, fmt='%1.5f')
            print(f"Outputting to CSV file: {args.csv}")
        else:
            if len(fscs) > 0:
                fig, ax = plt.subplots(1, figsize=(8, 5))
                for i in range(len(fscs)):
                    pobj, = ax.plot(resolution, fscs[i])
                    pobj.set_label(mrc_filenames[i + 1])

                if len(fscs) > 1:
                    ax.legend()
                else:
                    ax.set_title(f"{mrc_filenames[0]} and {mrc_filenames[1]}")
                plt.xlabel('Resolution (1/A)')
                plt.ylabel('FSC [a.u.]')
                if args.title is not None:
                    plt.title(args.title)
                plt.show()
            else:
                print("Nothing to plot!")

    if args.powers:
        grid = None
        mask = None
        powers = []

        print("Processing:")
        for i in range(len(mrc_filenames)):
            print(f" {mrc_filenames[i]}")

            grid, voxel_size_, _ = load_mrc_(mrc_filenames[i])
            if voxel_size is None:
                voxel_size = voxel_size_ if voxel_size_ > 0 else 1.
            if mask is None:
                if solvent_mask is None:
                    mask = spherical_mask(grid.shape[0]) if args.spherical_mask else 1
                else:
                    mask = solvent_mask
            powers.append(get_power_real(grid * mask).log())

        resolution = spectral_resolution(powers[0].shape[0], voxel_size)
        if args.max_resolution is not None:
            max_index = spectral_index_from_resolution(args.max_resolution, grid.shape[0], voxel_size)
            resolution[i] = resolution[:max_index]
            for i in range(len(powers)):
                powers[i] = powers[i][:max_index]

        if args.csv is not None:
            out = np.zeros((len(resolution), len(powers)))
            header = ""
            for i in range(len(powers)):
                out[:, i] = fscs[i]
                header += f",{mrc_filenames[i]}"
            np.savetxt(args.csv, out, delimiter=', ', header=header, fmt='%1.5f')
            print(f"Outputting to CSV file: {args.csv}")
        else:
            if len(powers) > 0:
                fig, ax = plt.subplots(1, figsize=(8, 5))
                for i in range(len(powers)):
                    pobj, = ax.plot(resolution, powers[i])
                    pobj.set_label(mrc_filenames[i])
                if len(powers) > 1:
                    ax.legend()
                else:
                    ax.set_title(f"{mrc_filenames[0]}")
                plt.xlabel('Resolution (1/A)')
                plt.ylabel('log(Power) [a.u.]')
                if args.title is not None:
                    plt.title(args.title)
                plt.show()
            else:
                print("Nothing to plot!")


def append_args(parser):
    parser.add_argument(
        "mrcs",
        nargs="*",
        type=str,
        help="list of input mrc-files"
    )
    parser.add_argument("-f", "--fsc", action='store_true',
                        help="Calculate Fourier shell correlations, frist file is refernce.")
    parser.add_argument("-p", "--powers", action='store_true',
                        help="Calculate spectral powers")
    parser.add_argument(
        "-c,", "--csv",
        type=str,
        help="Rather than plot, dump to this CSV file path",
        default=None
    )
    parser.add_argument(
        "-m,", "--mask",
        type=str,
        help="Solvent mask",
        default=None
    )
    parser.add_argument(
        "-v,", "--voxel-size",
        type=float,
        help="Overwrite voxel size",
        default=None
    )
    parser.add_argument(
        "-r,", "--max-resolution",
        type=float,
        help="Maximum resolution (Angstrom) to calculate",
        default=None
    )
    parser.add_argument(
        "-t,", "--title",
        type=str,
        help="Title to set fort he plot",
        default=None
    )
    parser.add_argument("--spherical-mask", action='store_true', help="Apply a real-space spherical mask")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Used to plot or get a CSV of spectral statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)
