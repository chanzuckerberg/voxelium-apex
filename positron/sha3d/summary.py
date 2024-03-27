#!/usr/bin/env python3

"""
Module for generating 3D spectral heterogeneity analysis (SHA) summaries.
"""

import argparse
import os.path
from typing import Dict

import torch
import torch.nn.functional as F

import numpy as np
from positron.base import dt_desymmetrize, idft, rescale_fourier, smooth_spherical_mask, load_mrc

from positron.sha3d.cache import Cache
from positron.base.explicit_grid_utils import size_to_maxr, maxr_to_size
from positron.sha3d.hidden_variable_container import HiddenVariableContainer
from positron.sha3d.train_utils import load_modules_from_logdir, save_modules_to_logdir, load_module


class Summary(torch.nn.Module):
    def __init__(self, metadata, basis):
        super().__init__()
        self.basis = torch.nn.Parameter(basis)
        self.metadata = metadata
        for name in self.metadata:
            if torch.is_tensor(self.metadata[name]):
                self.metadata[name] = torch.nn.Parameter(self.metadata[name])
                self.register_parameter(name, self.metadata[name])

    def forward(self, s):
        if self.basis is None:
            raise RuntimeError("Bases have not been initialized.")
        return Summary.linear_combination(self.basis, s)

    @staticmethod
    def linear_combination(basis, s):
        if s.dim() == 1:  # Not a batch
            return torch.einsum("nzyx, n -> zyx", basis, s)
        else:
            return torch.einsum("nzyx, bn -> bzyx", basis, s)


    @staticmethod
    def extract_basis_from_decoder(decoder, spectral_rescaling, maxr=None):
        p = decoder.projector
        bz = p.size - 1 if maxr is None else maxr_to_size(maxr)
        device = p.weight.device
        s_size = decoder.s_size

        spectral_rescaling_grid = Cache.spectra_to_grids(
            spectra=spectral_rescaling,
            shape=(p.size-1, p.size-1, p.size_x),
            max_r=size_to_maxr(p.size)
        )

        basis = torch.zeros((s_size, bz, bz, bz), device=device)
        for i in range(s_size):
            x_ft = p.get_base(i)
            x_ft *= spectral_rescaling_grid
            if maxr is not None and maxr != size_to_maxr(x_ft.shape[0]):
                x_ft = rescale_fourier(x_ft, maxr_to_size(maxr))
            basis[i] = idft(x_ft, dim=3, real_in=True, centered=True)

        return basis

    @staticmethod
    def compose_from_modules(mnx, hvc: HiddenVariableContainer):
        circular_mask_radius, circular_mask_thickness = mnx.get_circular_mask_params()
        metadata = {
            'z': hvc.get_metadata('z'),
            's': hvc.get_metadata('s'),
            'voxel_size': mnx.voxel_size,
            'image_size': mnx.image_size,
            'circular_mask_radius': circular_mask_radius,
            'circular_mask_thickness': circular_mask_thickness
        }

        device = mnx.get_device()
        _, data_ctf_spectra, _ = hvc.get_data_stats(0)
        data_ctf_spectra = data_ctf_spectra.to(device)
        basis = Summary.extract_basis_from_decoder(mnx.decoder, data_ctf_spectra)

        grid_size = basis.shape[-1]
        mask = smooth_spherical_mask(
            grid_size=grid_size,
            radius=min(circular_mask_radius, grid_size / 2 - circular_mask_thickness),
            thickness=circular_mask_thickness
        ).to(basis.device)

        basis *= mask[None]

        return Summary(
            metadata=metadata,
            basis=basis
        )

    @staticmethod
    def compose_from_logdir(path, state=None, device=torch.device("cpu")):
        mnx, hvc = load_modules_from_logdir(
            path,
            [
                "mnx" if state is None else f"mnx_{state}",
                "hvc" if state is None else f"hvc_{state}"
             ]
        )

        # Has to be imported here due to torch extension dependencies
        from positron.sha3d.model_container import ModelContainer

        mnx = ModelContainer.load_from_state_dict(mnx)
        hvc = HiddenVariableContainer.load_from_state_dict(hvc)

        return Summary.compose_from_modules(mnx, hvc).to(device)

    @staticmethod
    def load_from_logdir(path, state=None, device=torch.device("cpu")):
        try:
            sum, = load_modules_from_logdir(
                path, ["sum" if state is None else f"mnx_{state}"])
            return Summary.load_from_state_dict(sum).to(device)
        except FileNotFoundError:
            pass

        return Summary.compose_from_logdir(path=path, state=state, device=device)

    @staticmethod
    def load_from_path(path, state=None, device=torch.device("cpu")):
        if os.path.isfile(path):
            return Summary.load_from_state_dict(
                load_module(path)).to(device)
        else:
            return Summary.load_from_logdir(
                path=path, state=state, device=device)

    def save_to_logdir(self, path, state_label, gzip=False):
        save_modules_to_logdir(
            path,
            modules={
                "sum" if state_label is None else f"sum_{state_label}":
                    self.get_state_dict()
            },
            gzip=gzip
        )

    @staticmethod
    def to_sparse_basis(dense_basis):
        mask = dense_basis.sum(dim=0) != 0
        box_shape = dense_basis.shape[1:]
        indices = torch.arange(np.prod(box_shape)).view(box_shape)
        indices = indices[mask]
        values = dense_basis[:, mask].contiguous()

        return values, indices

    def get_state_dict(self) -> Dict:
        sparse_values, sparse_indices = Summary.to_sparse_basis(self.basis)

        return {
            "type": "MnxVoxelSummary",
            "version": "0.0.1",

            "sparse_values": sparse_values,
            "sparse_indices": sparse_indices,
            "box_shape": self.basis.shape[1:],
            "metadata": self.metadata
        }

    @staticmethod
    def to_dense_basis(values, indices, box_shape) -> torch.tensor:
        basis = torch.zeros(
            [values.size(0), np.prod(box_shape)], dtype=values.dtype, device=values.device)
        basis[:, indices] = values
        basis = basis.view([values.size(0)] + list(box_shape))
        return basis

    @staticmethod
    def load_from_state_dict(state_dict):
        if "type" not in state_dict or state_dict["type"] != "MnxVoxelSummary":
            raise TypeError("Input is not an 'MnxVoxelContainer' instance.")

        if "version" not in state_dict:
            raise RuntimeError("MnxVoxelSummary instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            metadata = state_dict["metadata"]
            basis = Summary.to_dense_basis(
                values=state_dict["sparse_values"],
                indices=state_dict["sparse_indices"],
                box_shape=state_dict["box_shape"]
            )

            return Summary(
                metadata=metadata,
                basis=basis
            )

    def apply_mask(self, mask):
        if isinstance(mask, str):
            mask, _, _ = load_mrc(mask)
            mask = torch.from_numpy(mask.copy())
            if np.any(mask.shape != self.basis.shape[1:]):
                raise RuntimeError(f"Model dimensions ({self.basis.shape[1:]}) "
                                   f"do not match the mask ({mask.shape})")

        self.basis.data *= mask[None]


def append_args(parser):
    parser.add_argument("logdir", help="Log directory")
    parser.add_argument("--state", "-s", "--s", help="State label to choose", default=None)
    parser.add_argument("--output", "-o", "--o", help="Output file", default=None)
    parser.add_argument("--mask", "-m", "--m", help="Mask file", default=None)
    parser.add_argument("--no_gzip", "-z", "--z", help="Compress the output file", action="store_true")


def main(args):
    torch.no_grad()

    print("Loading modules...")
    summary = Summary.compose_from_logdir(args.logdir, state=args.state)

    if args.mask is not None:
        if os.path.isfile(args.mask):
            print("Applying mask...")
            summary.apply_mask(args.mask)
        else:
            print("WARNING: Could not find mask-file. Will continue, but file size will be larger.")

    print("Saving summary to log directory...")
    summary.save_to_logdir(args.logdir, state_label=args.state, gzip=not args.no_gzip)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Used to make a summary of Mnx voxel analysis modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)

