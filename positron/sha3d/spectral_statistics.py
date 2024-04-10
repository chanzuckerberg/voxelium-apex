#!/usr/bin/env python3

"""
"""

import torch

from typing import List, TypeVar, Dict, Union

import numpy as np
from positron.base import grid_spectral_average

from positron.sha3d.cache import Cache

from positron.base.explicit_grid_utils import size_to_maxr
from .train_utils import smoothen_spectra

Tensor = TypeVar('torch.tensor')


class SpectralStatistics(torch.nn.Module):
    def __init__(
            self,
            image_size,
            filter_cutoff_idx,
            lam
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.filter_cutoff_idx = filter_cutoff_idx
        self.lam = lam
        self.maxr = size_to_maxr(image_size)

        self.c_train_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_valid_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_train_ctf_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_valid_ctf_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_train_max_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_valid_max_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)

        self.mse = torch.nn.Parameter(torch.ones(self.maxr), requires_grad=False)

        self.x_train_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.x_valid_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.y_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.x_ctf_train_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.x_ctf_valid_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.y_ctf_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)

    @staticmethod
    @torch.no_grad()
    def get_spectrum_from_grid_(grid):
        mask = Cache.get_spectral_mask(
            grid.shape[-2:],
            max_r=size_to_maxr(grid.shape[-2]),
            device=grid.device
        )
        indices = Cache.get_spectral_indices(
            grid.shape[-2:],
            max_r=size_to_maxr(grid.shape[-2]),
            device=grid.device
        )[mask]

        grid = grid[mask] if len(grid.shape) == 2 else grid[:, mask]
        return grid_spectral_average(grid, indices)

    def spectral_ema_(self, exponential_mean, latest, momentum):
        if len(exponential_mean.shape) == 1:
            latest[-1] = latest[-2]
            latest = smoothen_spectra(latest[None], kernel=3)[0]
        else:
            latest[:, -1] = latest[:, -2]
            latest = smoothen_spectra(latest, kernel=3)

        exponential_mean[:] = exponential_mean * momentum + latest * (1 - momentum)

    @torch.no_grad()
    def update(self, x, y, ctf2, train_mask, valid_mask, mse, momentum=0.99):
        x = torch.view_as_complex(x).detach()
        y = torch.view_as_complex(y).detach()

        mse = self.get_spectrum_from_grid_(mse)
        self.spectral_ema_(self.mse.data, mse / (ctf2 + 1e-12), momentum)

        # POWERS ##########################################################################################

        x_train_power = torch.mean(x[train_mask].square().abs(), 0)
        x_valid_power = torch.mean(x[valid_mask].square().abs(), 0)

        y_power = torch.mean(y.square().abs(), 0)

        x_train_power_spec = self.get_spectrum_from_grid_(x_train_power)
        x_valid_power_spec = self.get_spectrum_from_grid_(x_valid_power)

        y_power_spec = self.get_spectrum_from_grid_(y_power)

        self.spectral_ema_(self.x_train_power.data, x_train_power_spec, momentum)
        self.spectral_ema_(self.x_valid_power.data, x_valid_power_spec, momentum)

        self.spectral_ema_(self.x_ctf_train_power.data, x_train_power_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.x_ctf_valid_power.data, x_valid_power_spec / (ctf2 + 1e-12), momentum)

        self.spectral_ema_(self.y_power.data, y_power_spec, momentum)
        self.spectral_ema_(self.y_ctf_power.data, y_power_spec / (ctf2 + 1e-12), momentum)

        # FSC #############################################################################################

        cc = torch.sum(torch.view_as_real(x) * torch.view_as_real(y), -1)
        c_train_spec = self.get_spectrum_from_grid_(torch.mean(cc[train_mask], 0))
        c_valid_spec = self.get_spectrum_from_grid_(torch.mean(cc[valid_mask], 0))

        self.spectral_ema_(self.c_train_spectrum.data, c_train_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.c_valid_spectrum.data, c_valid_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.c_train_ctf_spectrum.data, c_train_spec, momentum)
        self.spectral_ema_(self.c_valid_ctf_spectrum.data, c_valid_spec, momentum)

        c_train_max_spec = self.get_spectrum_from_grid_(cc[train_mask])
        c_valid_max_spec = self.get_spectrum_from_grid_(cc[valid_mask])
        c_train_max_spec = c_train_max_spec[torch.argmax(torch.mean(c_train_max_spec, 1))]
        c_valid_max_spec = c_valid_max_spec[torch.argmax(torch.mean(c_valid_max_spec, 1))]
        c_train_max_spec = torch.clip(c_train_max_spec, 0, 1)
        c_valid_max_spec = torch.clip(c_valid_max_spec, 0, 1)

        self.spectral_ema_(self.c_train_max_spectrum.data, c_train_max_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.c_valid_max_spectrum.data, c_valid_max_spec / (ctf2 + 1e-12), momentum)

    def get_fsc_spectrum(self, eps=1e-12):
        fsc = self.c_valid_spectrum.data / (self.c_train_spectrum.data + eps)
        fsc = torch.clip(smoothen_spectra(fsc[None], kernel=10)[0], 0, 1)
        fsc[:2] = 1
        return fsc

    def get_y_weight(self, eps=1e-4):
        # sigma = self.mse.data.clip(0, 1)
        sigma = torch.clip(1 - self.c_valid_spectrum.data.clip(0, 1), 0)

        weight = 1. / (sigma + eps)
        return weight

    def get_x_noise(self):
        fsc = self.get_fsc_spectrum()
        return torch.clip(self.x_ctf_train_power.data * (1 - fsc), 0, 1)

    def get_x_signal(self):
        fsc = self.get_fsc_spectrum()
        return torch.clip(self.x_ctf_train_power.data * fsc, 0, 1)

    def get_spectral_summary(self):
        sigma2_weight = self.get_y_weight()
        sigma2_weight /= torch.max(sigma2_weight) + 1e-12

        sigma_1 = self.mse.data.clip(0, 1)
        sigma_2 = torch.clip(1 - self.c_valid_spectrum.data.clip(0, 1), 0)

        return {
            'Means': [
                {'y': sigma2_weight, 'label': '1/sigma^2', 'color': 'black'},
                {'y': self.get_fsc_spectrum(), 'label': 'FSC x', 'color': 'black',
                 'linestyle': 'dashed'},
                {'y': torch.clip(self.c_train_spectrum.data, 0, 1), 'label': 'cc train', 'color': 'blue'},
                {'y': torch.clip(self.c_valid_spectrum.data, 0, 1), 'label': 'cc valid', 'color': 'green'},
                {'y': torch.clip(self.mse.data, 0, 1),
                 'label': 'MSE', 'color': 'green', 'linestyle': 'dashed'},
            ],
            'Powers': [
                {'y': self.x_ctf_train_power.data, 'label': 'x ctf train', 'color': 'blue'},
                {'y': self.x_ctf_valid_power.data, 'label': 'x ctf valid', 'color': 'green'},
                {'y': self.y_ctf_power.data, 'label': 'y ctf', 'color': 'grey'},
                {'y': self.x_train_power.data, 'label': 'x train', 'color': 'blue', 'linestyle': 'dashed'},
                {'y': self.x_valid_power.data, 'label': 'x valid', 'color': 'green', 'linestyle': 'dashed'},
                {'y': self.y_power.data, 'label': 'y', 'color': 'grey', 'linestyle': 'dashed'},
            ],
            'Data Noise': [
                {'y': sigma_1, 'label': 'sigma_1', 'color': 'green'},
                {'y': sigma_2, 'label': 'sigma_2', 'color': 'red'},
            ],
        }

