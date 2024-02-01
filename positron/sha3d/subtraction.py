#!/usr/bin/env python3

"""
"""
import os
import pickle
from collections import OrderedDict

import torch

from typing import List, TypeVar, Dict, Union

import numpy as np
from positron.base import grid_spectral_sum, grid_spectral_average, fourier_shift_2d, dt_desymmetrize, idft, \
    integer_shift_2d, save_mrc

from positron.sha3d.cache import Cache

from positron.base.explicit_grid_utils import size_to_maxr
from .mask_applicator import apply_solvent_mask
from .train_utils import smoothen_spectra, zero_fill_number


class SubtractionHelper:
    def __init__(
            self,
            log_dir,
            rec,
            mask,
            image_max_r,
            buffer_size=500 * 1024**2  # 300 MiB
    ) -> None:
        self.log_dir = log_dir
        self.rec = rec
        self.mask = mask
        self.image_max_r = image_max_r
        self.buffer_size = buffer_size

        self.last_batch_nbytes = 0

        self.projector = None
        self.index_ref = {}

        self.current_file_index = 0
        self.current_indices = None
        self.current_data = None

    @torch.no_grad()
    def initialize(self):
        self.rec.zero_grad()  # To save some space
        self.projector = self.rec.decoder.projector.clone()
        apply_solvent_mask(self.projector, 1 - self.mask)
        print(f"Subtracting and writing data to: {self.log_dir}")
        os.makedirs(self.log_dir)

    def append_data(self, indices, data):
        if self.current_data is not None and self.current_data.nbytes + data.nbytes > self.buffer_size:
            self.flush()

        if self.current_indices is None:
            self.current_indices = indices
            self.current_data = data
        else:
            self.current_indices = np.concatenate([self.current_indices, indices], 0)
            self.current_data = np.concatenate([self.current_data, data], 0)

    def flush(self):
        if self.current_indices is None:
            return

        num = f"{self.current_file_index:03d}"
        self.current_file_index += 1
        current_file_name = f"batch_{num}.mrcs"

        self.index_ref[current_file_name] = self.current_indices
        self.current_indices = None

        path = os.path.join(self.log_dir, current_file_name)
        save_mrc(self.current_data, path, voxel_size=self.rec.voxel_size)
        self.current_data = None

    @torch.no_grad()
    def __call__(self, s, sample, hv):
        device = s.device

        self.rec.eval()
        x_ft = self.rec.decoder(
            s=s, max_r=self.image_max_r, rot_matrices=hv["rot_matrices"], projector=self.projector)

        x_ft = fourier_shift_2d(x_ft, hv["shifts_resid"])

        x_ft = x_ft * hv['ctfs_'][..., None]
        x_ft = torch.view_as_complex(x_ft)
        x_ft *= hv["amp_ctf_"] + 1e-6

        x_ft = dt_desymmetrize(x_ft, dim=2)
        x = idft(x_ft, dim=2, real_in=True)
        x = integer_shift_2d(x, -hv["shifts_int"])

        y = sample['image'].to(device) - x
        y = y.detach().cpu().numpy()

        self.append_data(indices=sample['idx'].numpy(), data=y)

    def finalize(self):
        self.flush()
        path = os.path.join(self.log_dir, "index.pkl")
        with open(path, 'wb') as file:
            pickle.dump(self.index_ref, file)
