#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np

from voxelium import save_mrc, load_mrc, get_bounding_box, gaussian_blur
from voxelium_apex.sha3d.summary import Summary
from voxelium_apex.sha3d.train_utils import setup_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts the weights of a decoder layer and prints them as MRC files.")
    parser.add_argument('path', help='input summary file or logdir', type=str)
    parser.add_argument('--output', '-o', '--o', help='root path for output files', type=str, default=None)
    parser.add_argument('--std', '-s', action='store_true', help="Output standard deviation volume")
    parser.add_argument('--mean', '-m', action='store_true', help="Output mean S volume")
    parser.add_argument('--mrc', '-r', action='store_true', help="Output basis in MRC files")
    parser.add_argument('--csv', '-c', action='store_true', help="Output CSV file with metadata")
    parser.add_argument('--msgp', '-p', action='store_true', help="Output MessagePack file")
    parser.add_argument('--gpu', type=str, default=None, help='GPU to use')
    parser.add_argument('--mask', type=str, default=None, help='Mask to use in compression')
    parser.add_argument('--z-bounds', type=str, default=None, help='Bonding box of z [left right bottom top]')
    parser.add_argument('--z-bins', type=str, default=500, help='Z image size')
    parser.add_argument('--z-smooth', type=str, default=4, help='Z image smoothing')
    parser.add_argument('--plot', action='store_true', help="Plot Z image")
    args = parser.parse_args()

    device, _ = setup_device(args)

    torch.no_grad()
    sum = Summary.load_from_path(args.path, device=device)
    x = sum.basis
    z = sum.metadata['z']
    s = sum.metadata['s']

    root_path = args.output if args.output is not None else os.path.join(args.path, "extract")
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
    s_size = s.shape[1]

    if args.mrc:
        print(f"Writing out {s_size} basis...")
        for i in range(s_size):
            x_ = x[i].cpu().detach().numpy().astype(np.float32)
            path = os.path.join(root_path, f"base_{i:03}.mrc")
            print(f" {path}")
            save_mrc(x_, path)

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

    if args.csv:
        z_ = z.detach().cpu().numpy()
        path = os.path.join(root_path, "metadata_z.csv")
        np.savetxt(path, z_, delimiter=',', fmt='%.4e')

        s_ = s.detach().cpu().numpy()
        path = os.path.join(root_path, "metadata_s.csv")
        np.savetxt(path, s_, delimiter=',', fmt='%.4e')

    if args.msgp:
        x_ = x.detach().cpu().numpy()
        s_ = s.detach().cpu().numpy()
        z_ = z.detach().cpu().numpy()

        print("Initial number of particles:", len(z_))

        _, unique_indices = np.unique(z_, axis=0, return_index=True)
        z_ = z_[unique_indices]
        s_ = s_[unique_indices]

        print("Number of particles with unique Z:", len(z_))

        if args.z_bounds is not None:
            tokens = args.z_bounds.split(" ")
            left = float(tokens[0])
            right = float(tokens[1])
            bottom = float(tokens[2])
            top = float(tokens[3])
            mask = ((left < z_[:, 0]) & (z_[:, 0] < right) &
                    (bottom < z_[:, 1]) & (z_[:, 1] < top))
            s_ = s_[mask]
            z_ = z_[mask]

            print("Number of particles after bounds:", len(z_))

        if args.mask is not None:
            mask, _, _ = load_mrc(args.mask)
            mask = torch.from_numpy(mask.copy())
            ini, end = get_bounding_box(mask > 0)
            x_ = x[:, ini[0]:end[0], ini[1]:end[1], ini[2]:end[2]]

        print("Basis box size:", x_.shape[1], x_.shape[2], x_.shape[3])
        print("Number of elements:", x_.numel())
        print("Final number of particles:", len(z_))

        bins = args.z_bins
        z_min = np.min(z_, axis=0)
        z_max = np.max(z_, axis=0)
        c = (z_ - z_min) / (z_max - z_min) * (bins - 1)

        c = np.round(c).astype(int)
        hm = np.zeros([bins, bins], dtype=int)
        np.add.at(hm, (c[:, 1], c[:, 0]), 1)
        hm_blur = gaussian_blur(hm.astype(np.float32), args.z_smooth)

        if args.plot:
            import matplotlib.pylab as plt
            plt.imshow(hm_blur)
            plt.show()

        from scipy.spatial import cKDTree
        _, unique_indices = np.unique(c, axis=0, return_index=True)
        c = c[unique_indices]
        s_ = s_[unique_indices]
        tree = cKDTree(c)

        z2s = np.full([bins, bins], -1, dtype=int)
        for i in range(bins):
            for j in range(bins):
                p = np.array([i, j])
                distance, index = tree.query(p)
                if distance < 5:
                    z2s[i, j] = index

        sum_dict = {
            "sha3d_summary_version": "1.0.0",
            "s": s_.flatten().tolist(),
            "s_shape": list(s_.shape),
            "map": z2s.flatten().tolist(),
            "hm": hm_blur.flatten().tolist(),
            "hm_shape": list(hm.shape),
            "x": x_.flatten().tolist(),
            "x_shape": list(x_.shape),
        }

        import msgpack
        path = os.path.join(root_path, "sum.msgpack")
        with open(path, 'wb') as file:
            packed = msgpack.pack(sum_dict, file, use_single_float=True, use_bin_type=True)

        print(f"Writing MessagePack file:", path)
