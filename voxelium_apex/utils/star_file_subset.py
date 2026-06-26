#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import argparse, starfile
from pathlib import Path
import numpy as np

def export_subset(expanded_particles, source_particles, subset, output):
    """Export a subset of rows from a STAR-file."""

    # Load the particles from the STAR-file and the subset from the CSV file.
    print(f'Exporting subset ({subset}) of particles ({source_particles}) to: {output}')

    # Read the particles from the STAR-file and the subset from the CSV file.
    subset = np.loadtxt(subset, dtype=int)
    source_df = starfile.read(source_particles)
    general = source_df['general']
    optics = source_df['optics']
    particles = source_df['particles']
    
    # If an expanded particles STAR-file is provided, use it to subset the source particles.
    expand_df = starfile.read(expanded_particles)
    keep_particles = expand_df['particles'].iloc[subset].reset_index(drop=True)
    keep_particles = keep_particles['rlnTomoParticleName'].unique()
    particles = particles[particles['rlnTomoParticleName'].isin(keep_particles)]

    # Write the output STAR-file.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({'general': general, 'optics': optics, 'particles': particles}, output)


def load_mrc_(filename):
    from voxelium import load_mrc
    import torch
    grid, voxel_size, global_origin = load_mrc(filename)
    grid = torch.from_numpy(grid.copy())
    return grid, voxel_size, global_origin


def main(args):
    star = starfile.read(args.input)
    if args.table not in star:
        raise RuntimeError(f"Could not fine the selected table ({args.table}) in STAR-file.")

    indices = np.loadtxt(args.index, dtype=int)

    df = star[args.table]
    if args.exclude:
        new_df = df.drop(indices)
    else:
        new_df = df.iloc[indices]

    print(f"Number of rows in original STAR-file table: {len(df)}")
    print(f"Number of rows in new STAR-file table: {len(new_df)}")

    out_path = args.output
    if out_path is None:
        if args.index[-4:] == ".csv":
            out_path = f"{args.index[:-4]}.star"
        else:
            out_path = f"{args.index}.star"

    star[args.table] = new_df
    starfile.write(star, out_path, overwrite=args.overwrite)
    print(f"Output written to: {out_path}")


def append_args(parser):
    parser.add_argument("--input", "-i", "--in", type=str,
                        help="Input STAR-file", required=True)
    parser.add_argument("--index", "-n", "--idx", type=str,
                        help="File containing row indices", required=True)
    parser.add_argument("--output", "-o", "--out", type=str,
                        help="Output STAR-file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Output STAR-file")
    parser.add_argument("--table", type=str, default="particles",
                        help="Name of the table to select from")
    parser.add_argument("--exclude", action="store_true",
                        help="Instead of including the indices in --index, exclude them")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Tool for subset extraction from STAR-file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)