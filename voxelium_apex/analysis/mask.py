#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import sys
import os

import argparse
import glob
import pickle
import time

import numpy as np

import matplotlib.pylab as plt

import torch
import torch.nn.functional as F

from voxelium import load_mrc, save_mrc


def load_mrc_(filename):
    grid, voxel_size, global_origin = load_mrc(filename)
    grid = torch.from_numpy(grid.copy())
    return grid, voxel_size, global_origin

def create_spherical_kernel(radius, device=None, dtype=torch.float32):
    """
    Creates a binary 3D spherical kernel with given radius.
    Voxels whose Euclidean distance ≤ radius from center are 1, else 0.
    """
    diameter = 2 * radius + 1
    # create grid of coordinates centered at zero
    coords = torch.arange(diameter, device=device) - radius
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    dist2 = x**2 + y**2 + z**2
    kernel = (dist2 <= radius**2).to(dtype)
    return kernel

def dilate3d_by_radius(grid, radius):
    """
    grid: 3D tensor of 0s and 1s, shape (D, H, W)
    radius: int, Euclidean radius for dilation
    returns: 3D tensor same shape, where zeros become 1 if any original 1
             lies within the sphere of radius around them.
    """
    # build the spherical structuring element
    kernel = create_spherical_kernel(radius, device=grid.device, dtype=grid.dtype)
    # shape for conv3d: (out_channels, in_channels, kz, ky, kx)
    kernel = kernel.unsqueeze(0).unsqueeze(0)       # -> [1,1,depth,height,width]
    
    # add batch+channel dims to input
    x = grid.unsqueeze(0).unsqueeze(0)             # -> [1,1,depth,height,width]
    # convolve with padding so output is same size
    conv = F.conv3d(x, kernel, padding=radius)
    # threshold: any positive means at least one 1 in the neighborhood
    out = (conv > 0).to(grid.dtype)
    # drop batch+channel dims
    return out.squeeze(0).squeeze(0)


def create_cosine_kernel(radius, device=None, dtype=torch.float32):
    """
    Create a (2r+1)^3 kernel whose weights fall from 1 at the center
    to 0 at distance r following a cosine:
        w(r') = 0.5 * (1 + cos(pi * r' / r))   for r' <= r
        w(r') = 0                              otherwise
    """
    diameter = 2*radius + 1
    coords = torch.arange(diameter, device=device) - radius
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    dist = torch.sqrt(x**2 + y**2 + z**2)
    kernel = torch.zeros_like(dist, dtype=dtype)
    mask = dist <= radius
    kernel[mask] = 0.5 * (1 + torch.cos(torch.pi * dist[mask] / radius))
    return kernel

def soft_dilate3d(grid, radius):
    """
    grid: 3D tensor of 0s and 1s, shape (D,H,W), dtype float32
    radius: int radius for dilation
    returns: 3D tensor of floats in [0,1], where each voxel is
             the max cosine-weighted influence of any original 1.
    """
    # build kernel
    kernel = create_cosine_kernel(radius, device=grid.device, dtype=grid.dtype)
    D, H, W = grid.shape
    
    # pad the volume so we can shift without wraparound
    pad = (radius,)*6  # (x1,x2, y1,y2, z1,z2)
    padded = F.pad(grid, pad)
    
    # prepare output
    out = torch.zeros_like(grid, dtype=grid.dtype, device=grid.device)
    
    # slide the kernel footprint, taking max(weight * shifted_grid)
    for dz in range(2*radius+1):
        for dy in range(2*radius+1):
            for dx in range(2*radius+1):
                w = kernel[dz, dy, dx].item()
                if w == 0.0:
                    continue
                sub = padded[dz:dz+D, dy:dy+H, dx:dx+W]
                out = torch.max(out, sub * w)
    
    return out

def main(args):
    grid, voxel_size, global_origin = load_mrc_(args.input)

    if args.threshold is None:
        args.threshold = torch.median(grid).item()

    print(f"Thresholding at {args.threshold}")
    grid = (grid > args.threshold).to(torch.float32)
    
    if args.dilate > 0:
        print(f"Dilating by {args.dilate}")
        grid = dilate3d_by_radius(grid, args.dilate)
    if args.soft > 0:
        print(f"Adding a soft edge by {args.soft}")
        grid = soft_dilate3d(grid, args.soft)

    if args.output is None:
        output = None
        for i in range(999):
            output = args.input.replace(".mrc", f"_mask{i}.mrc")
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
    print(f"Done")


def append_args(parser):
    parser.add_argument("-i", "--input", type=str, help="Input file", required=True)
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        help="Threshold input map at this value",
        default=None
    )
    parser.add_argument(
        "-d", "--dilate",
        type=int,
        help="Dilate input map by this radius",
        default=0
    )
    parser.add_argument(
        "-s", "--soft",
        type=int,
        help="Add a soft edge by this radius",
        default=0
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
