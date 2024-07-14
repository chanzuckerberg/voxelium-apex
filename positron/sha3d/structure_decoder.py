#!/usr/bin/env python3

"""
Model for the Structure Decoder
"""

import torch
from positron.base.explicit_grid_utils import make_grid2d, make_grid3d, size_to_maxr, maxr_to_size
from ..base import spectra_to_grid


class StructureDecoder(torch.nn.Module):
    def __init__(self, grid3d_size, s_size, shift=True):
        super().__init__()
        self.grid3d_size = grid3d_size
        self.s_size = s_size
        self.max_r = size_to_maxr(self.grid3d_size)

        self.caches = {}

        from positron.torch_extensions.reconstruction_layer_3d import ReconstructionLayer3D
        self.projector = ReconstructionLayer3D(
            size=grid3d_size,
            input_size=s_size,
            do_bias=False
        )
        p = self.projector.weight.data
        p.normal_()
        for i in range(p.size(1)):
            p_square = p[:, i].square()
            p[i].mul_(1 / (p.size(1) * p_square.mean().sqrt() + 1e-12))

    def _load_cache(self, max_r, is_3d):
        hashable = str(max_r) + ("_3d" if is_3d else "_2d")
        if hashable not in self.caches:
            device = self.projector.weight.device
            if is_3d:
                coord, mask = make_grid3d(size=maxr_to_size(max_r) + 1)
                radius = torch.sqrt(torch.sum(torch.square(coord), -1))
                spectral_idx = torch.floor(radius).long()
                spectral_idx = spectral_idx[mask]
                nc_idx = torch.where(spectral_idx == 0)[0][0]

                self.caches[hashable] = [
                    mask.to(device),
                    spectral_idx.to(device),
                    nc_idx
                ]
            else:
                coord, mask = make_grid2d(size=self.grid3d_size, max_r=max_r)
                radius = torch.sqrt(torch.sum(torch.square(coord), -1))
                spectral_idx = torch.floor(radius).long()
                nc_idx = torch.where(spectral_idx == 0)[0][0]

                self.caches[hashable] = [
                    coord.to(device),
                    mask.to(device),
                    spectral_idx.to(device),
                    nc_idx
                ]

        return self.caches[hashable]

    def forward(
            self,
            s,
            max_r=None,
            rot_matrices=None,
            data_spectra=None,
            projector=None
    ):
        is_3d = rot_matrices is None

        if projector is None:
            projector = self.projector

        if max_r is None:
            max_r = self.max_r

        if is_3d:
            mask, spectral_idx, nc_idx = self._load_cache(max_r, True)
            x_ft = projector(input=s.contiguous(), max_r=max_r)

            if data_spectra is not None:
                spectral_scale = spectra_to_grid(data_spectra, spectral_idx)
                x_ft[:, mask] *= spectral_scale[None, ..., None]

        else:
            coord, mask, spectral_idx, nc_idx = self._load_cache(max_r, False)

            x_ft = projector(
                input=s.contiguous(),
                grid2d_coord=coord,  # TODO try using zCurve for this
                rot_matrices=rot_matrices,
                max_r=max_r
            )

            if data_spectra is not None:
                spectral_scale = spectra_to_grid(data_spectra, spectral_idx)
                x_ft *= spectral_scale[None, ..., None]

            size = self.grid3d_size
            x_ft_ = torch.zeros([x_ft.shape[0], size * (size // 2 + 1), 2]).to(x_ft.device)
            x_ft_[:, mask, :] = x_ft
            x_ft = x_ft_.view(-1, size, size // 2 + 1, 2)

        return x_ft
