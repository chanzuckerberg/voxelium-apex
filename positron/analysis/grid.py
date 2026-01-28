#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from positron.base import load_mrc, save_mrc, fast_gaussian_filter


def load_mrc_(filename):
    grid, voxel_size, global_origin = load_mrc(filename)
    # ensure float32 torch tensor
    grid = torch.from_numpy(grid.copy()).to(torch.float32)
    return grid, float(voxel_size), np.array(global_origin, dtype=np.float32)


# ---------- Filters & helpers ----------

def _fftfreq_grid(shape, device):
    """Return fx, fy, fz 3D frequency grids (cycles per voxel)."""
    nz, ny, nx = shape
    fz = torch.fft.fftfreq(nz, d=1.0, device=device)
    fy = torch.fft.fftfreq(ny, d=1.0, device=device)
    fx = torch.fft.fftfreq(nx, d=1.0, device=device)
    return torch.meshgrid(fz, fy, fx, indexing="ij")

def apply_ideal_filter(grid, voxel_size, lowpass_A=None, highpass_A=None):
    """
    Ideal low/high-pass in Fourier space.
    - lowpass_A: keep features worse than (>=) this resolution (Å); removes > cutoff frequency
    - highpass_A: remove features worse than this resolution (i.e., remove low-freq background)
    Internally uses cycles/voxel: f_cutoff_vox = voxel_size / resolution_A
    """
    if lowpass_A is None and highpass_A is None:
        return grid

    device = grid.device
    nz, ny, nx = grid.shape
    Fz, Fy, Fx = _fftfreq_grid((nz, ny, nx), device)
    r = torch.sqrt(Fx * Fx + Fy * Fy + Fz * Fz)  # radial frequency (cycles/voxel)

    mask = torch.ones_like(grid, dtype=torch.float32)

    if lowpass_A is not None:
        if lowpass_A <= 0:
            raise ValueError("lowpass must be > 0 Å")
        f_lp = voxel_size / float(lowpass_A)  # cycles/voxel
        mask = mask * (r <= f_lp)

    if highpass_A is not None:
        if highpass_A <= 0:
            raise ValueError("highpass must be > 0 Å")
        f_hp = voxel_size / float(highpass_A)  # cycles/voxel
        mask = mask * (r >= f_hp)

    fft_vol = torch.fft.fftn(grid)
    filtered = torch.fft.ifftn(fft_vol * mask)
    # result should be real for real input, keep numerical imag noise out
    return filtered.real.to(torch.float32)


def main(args):

    if not os.path.exists(args.input):
        print(f"File '{args.input}' not found")
        return

    grid, voxel_size, global_origin = load_mrc_(args.input)  # grid: (Z,Y,X) float32

    if args.stats:
        s = grid.shape
        print(f"Statistics for {args.input}")
        print(f"Shape (voxel): {s[0]} x {s[1]} x {s[2]}")
        print(f"Shape (Å): {s[0] * round(voxel_size, 2)} x {s[1] * round(voxel_size, 2)} x {s[2] * round(voxel_size, 2)}")
        print(f"Pixel size (Å/voxel): {voxel_size}")
        print(f"Origin: {global_origin.tolist()}")
        print(f"Mean: {float(grid.mean())}")
        print(f"Std: {float(grid.std())}")
        print(f"Min: {float(grid.min())}")
        print(f"Max: {float(grid.max())}")
        print(f"Median: {float(torch.median(grid))}")

    modifier = False

    # ---------- Cropping (Z,Y,X) with Python slicing semantics ----------
    # ---------- Cropping ----------
    if args.crop is not None and args.crop_to_shape is not None:
        raise ValueError("Use either --crop or --crop_to_shape, not both.")

    if args.crop_to_shape is not None:
        # Center-crop to the requested shape (Z, Y, X), clamped to the volume size
        tz, ty, tx = args.crop_to_shape
        if tz <= 0 or ty <= 0 or tx <= 0:
            raise ValueError("--crop_to_shape values must be positive integers (Z Y X)")

        nz, ny, nx = grid.shape
        tz = min(tz, nz); ty = min(ty, ny); tx = min(tx, nx)

        z0 = (nz - tz) // 2; z1 = z0 + tz
        y0 = (ny - ty) // 2; y1 = y0 + ty
        x0 = (nx - tx) // 2; x1 = x0 + tx

        print(f"Center-cropping to Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] "
            f"(target shape {tz}×{ty}×{tx}, input {nz}×{ny}×{nx})")
        grid = grid[z0:z1, y0:y1, x0:x1].contiguous()

        print(f"Output shape (voxel): {grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]}")
        ang = round(float(voxel_size), 2)
        print(f"Output shape (Å): {grid.shape[0] * ang} x {grid.shape[1] * ang} x {grid.shape[2] * ang}")
        modifier = True

    elif args.crop is not None:
        if len(args.crop) != 6:
            raise ValueError("--crop expects 6 integers: z0 z1 y0 y1 x0 x1 (end-exclusive)")
        z0, z1, y0, y1, x0, x1 = args.crop
        nz, ny, nx = grid.shape
        z0 = max(0, min(nz, z0)); z1 = max(0, min(nz, z1))
        y0 = max(0, min(ny, y0)); y1 = max(0, min(ny, y1))
        x0 = max(0, min(nx, x0)); x1 = max(0, min(nx, x1))
        if not (z0 < z1 and y0 < y1 and x0 < x1):
            raise ValueError("Invalid crop ranges; ensure start < end for each axis.")
        print(f"Cropping to Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}]")
        grid = grid[z0:z1, y0:y1, x0:x1].contiguous()
        print(f"Output shape (voxel): {grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]}")
        ang = round(float(voxel_size), 2)
        print(f"Output shape (Å): {grid.shape[0] * ang} x {grid.shape[1] * ang} x {grid.shape[2] * ang}")
        modifier = True

    # ---------- Arithmetic ops ----------
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

    # ---------- Frequency filters (Å cutoffs) ----------
    if args.lowpass is not None or args.highpass is not None:
        lp = args.lowpass
        hp = args.highpass
        if lp is not None:
            print(f"Applying ideal LOW-PASS at {lp:.3g} Å (remove > 1/{lp} Å⁻¹)")
        if hp is not None:
            print(f"Applying ideal HIGH-PASS at {hp:.3g} Å (remove < 1/{hp} Å⁻¹)")
        grid = apply_ideal_filter(grid, voxel_size, lowpass_A=lp, highpass_A=hp)
        modifier = True

    # ---------- Gaussian blur (σ in voxels) ----------
    if args.gaussian is not None:
        print(f"Applying 3D Gaussian blur with σ={args.gaussian} voxels")
        grid = fast_gaussian_filter(grid[None, None], args.gaussian)[0, 0]
        modifier = True

    # ---------- Save if anything changed ----------
    if modifier:
        if args.output is None:
            output = None
            for i in range(999):
                candidate = args.input.replace(".mrc", f"_mod{i}.mrc")
                if not os.path.exists(candidate):
                    output = candidate
                    break
            if output is None:
                raise ValueError("No available output filename")
            args.output = output

        print(f"Saving to {args.output}")
        save_mrc(
            grid.detach().cpu().numpy().astype(np.float32),
            args.output,
            voxel_size=voxel_size,
            origin=global_origin
        )


def append_args(parser):
    parser.add_argument("-i", "--input", type=str, help="Input MRC file", required=True)

    parser.add_argument(
        "--stats", "--statistics",
        help="Print statistics for input",
        action="store_true"
    )

    # Cropping: end-exclusive indices in Z,Y,X order
    parser.add_argument(
        "--crop",
        type=int,
        nargs=6,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="Crop volume using end-exclusive indices (Z,Y,X)."
    )
    parser.add_argument(
        "--crop_to_shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Crop volume to center on shape (Z,Y,X)."
    )

    # Frequency domain filters using Å cutoffs (common in cryo-EM)
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Ideal low-pass cutoff in Å (keeps frequencies ≤ 1/Å)."
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=None,
        help="Ideal high-pass cutoff in Å (removes frequencies < 1/Å)."
    )

    # Spatial Gaussian blur
    parser.add_argument(
        "--gaussian",
        type=float,
        default=None,
        help="Apply 3D Gaussian blur with sigma in voxels."
    )

    # Map arithmetic with other volumes
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

    # Scalar arithmetic
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
        help="Output MRC file (auto-generated if omitted)",
        default=None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D voxel grid utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()
    main(args)