#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np

from positron.base import save_mrc
from positron.sha3d.summary import Summary
from positron.sha3d.train_utils import setup_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts the weights of a decoder layer and prints them as MRC files.")
    parser.add_argument('path', help='input summary file or logdir', type=str)
    parser.add_argument('--output', '-o', '--o', help='root path for output files', type=str, default=None)
    parser.add_argument('--std', '-s', action='store_true', help="Output standard deviation volume")
    parser.add_argument('--mean', '-m', action='store_true', help="Output mean S volume")
    parser.add_argument('--gpu', type=str, default=None, help='gpu to use')
    args = parser.parse_args()

    device, _ = setup_device(args)

    torch.no_grad()
    sum = Summary.load_from_path(args.path, device=device)
    latent = sum.metadata['z']
    structure_factors = sum.metadata['s']

    root_path = args.output if args.output is not None else args.path
    s_size = sum.basis.shape[0]

    sign = np.ones(s_size)
    if 's' in sum.metadata:
        s = sum.metadata['s']
        if torch.is_tensor(s):
            s = s.cpu().detach().numpy()
        sign = np.mean(s, axis=0)

    print(f"Writing out {s_size} basis...")
    for i in range(s_size):
        x = sum.basis[i].cpu().detach().numpy() * sign[i]
        path = os.path.join(root_path, f"basis_{i:03}.mrc")
        print(f" {path}")
        save_mrc(x, path)

    s = sum.metadata['s']

    if args.mean or args.std:
        s_mean = s.mean(0)
        x_mean = sum(s_mean)
        if args.mean:
            print(f"Writing mean S volume...")
            path = os.path.join(root_path, f"basis_mean.mrc")
            print(f" {path}")
            save_mrc(x_mean.cpu().detach().numpy(), path)

    if args.std:
        print(f"Writing standard deviation volume...")
        x_std = torch.zeros_like(x_mean)
        N = min(s.size(0), 1000)
        idx = torch.randperm(s.size(0))[:N]
        for i in range(N):
            x_std += (sum(s[idx[i]]) - x_mean) ** 2
        x_std = torch.sqrt(x_std / (N + 1e-12))
        path = os.path.join(root_path, f"basis_std.mrc")
        print(f" {path}")
        save_mrc(x_std.cpu().detach().numpy(), path)

