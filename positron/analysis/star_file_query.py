#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import argparse
import numpy as np
import starfile
import torch

from voxelium import load_mrc


def main(args):
    star = starfile.read(args.input)
    if args.table not in star:
        raise RuntimeError(f"Could not fine the selected table ({args.table}) in STAR-file.")

    df = star[args.table]
    new_df = df.query(args.query)

    print(f"Number of rows in original STAR-file table: {len(df)}")
    print(f"Number of rows in new STAR-file table: {len(new_df)}")

    out_path = args.output
    if out_path is None:
        out_path = f"{args.input[:-5]}_subset.star"

    star[args.table] = new_df
    starfile.write(star, out_path, overwrite=args.overwrite)
    print(f"Output written to: {out_path}")


def append_args(parser):
    parser.add_argument("--input", "-i", "--in", type=str,
                        help="Input STAR-file", required=True)
    parser.add_argument("--query", "-q", type=str,
                        help="Query to apply to selection", required=True)
    parser.add_argument("--output", "-o", "--out", type=str,
                        help="Output STAR-file")
    parser.add_argument("--overwrite", "-w", "--ow", action="store_true",
                        help="Output STAR-file")
    parser.add_argument("--table", "-t", type=str, default="particles",
                        help="Name of the table to select from")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Tool for subset extraction from STAR-file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)