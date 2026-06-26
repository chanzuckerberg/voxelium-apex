#!/usr/bin/env python3

"""
"""

import torch

from typing import List, TypeVar, Dict, Union

import numpy as np
from voxelium import grid_spectral_average, spectrum_to_grid_mean, size_to_maxr


from voxelium_apex.sha3d.infra.cache import Cache

from voxelium_apex.sha3d.train.train_utils import smoothen_spectra

Tensor = TypeVar('torch.tensor')


class SpectralStatistics(torch.nn.Module):
    def __init__(
            self,
            image_size
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.maxr = size_to_maxr(image_size)

        self.c_train_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_valid_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.x_ctf_train_power_spectrum = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)

        # These stats are not used in regularization
        self.mse = torch.nn.Parameter(torch.ones(self.maxr), requires_grad=False)
        self.c_train_ctf = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.c_valid_ctf = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.x_train_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.x_valid_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
        self.y_power = torch.nn.Parameter(torch.zeros(self.maxr), requires_grad=False)
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
        # Because of edge artefacts, the last 2 elements will be distorted
        if len(exponential_mean.shape) == 1:
            latest[-2:] = latest[-4:-2].mean()
        else:
            latest[:, -2:] = latest[:, -4:-2].mean()

        exponential_mean[:] = exponential_mean * momentum + latest * (1 - momentum)

    @torch.no_grad()
    def update(self, x, y, ctf2, train_mask, valid_mask, mse, momentum=0.99):
        if not torch.any(train_mask) or not torch.any(valid_mask):
            raise RuntimeError('Invalid train or valid mask.')

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

        self.spectral_ema_(self.x_ctf_train_power_spectrum.data, x_train_power_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.x_ctf_valid_power.data, x_valid_power_spec / (ctf2 + 1e-12), momentum)

        self.spectral_ema_(self.y_power.data, y_power_spec, momentum)
        self.spectral_ema_(self.y_ctf_power.data, y_power_spec / (ctf2 + 1e-12), momentum)

        # FSC #############################################################################################

        cc = torch.sum(torch.view_as_real(x) * torch.view_as_real(y), -1)
        c_train_spec = self.get_spectrum_from_grid_(torch.mean(cc[train_mask], 0))
        c_valid_spec = self.get_spectrum_from_grid_(torch.mean(cc[valid_mask], 0))

        self.spectral_ema_(self.c_train_spectrum.data, c_train_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.c_valid_spectrum.data, c_valid_spec / (ctf2 + 1e-12), momentum)
        self.spectral_ema_(self.c_train_ctf.data, c_train_spec, momentum)
        self.spectral_ema_(self.c_valid_ctf.data, c_valid_spec, momentum)

    def get_fsc_spectrum(self, eps=1e-6):
        # Smoothen rigorously to avoid a noisy FSC estimate as the two spectra are almost the same value
        c_valid = smoothen_spectra(self.c_valid_spectrum.data, kernel=self.maxr // 5)
        c_train = smoothen_spectra(self.c_train_spectrum.data, kernel=self.maxr // 5)
        fsc = c_valid / (c_train + eps)
        fsc = fsc.clip(0, 1)
        fsc[:2] = 1

        # Make sure it's monotonically decaying
        fsc = smoothen_spectra(fsc, kernel=5)
        fsc, _ = fsc.cummin(dim=0)
        fsc = smoothen_spectra(fsc, kernel=5)

        return fsc

    def get_y_weight(self, eps=1e-4):
        c_valid = self.c_valid_spectrum.data.clip(0, 1)
        c_valid = smoothen_spectra(c_valid, kernel=5).clip(0, 1)
        sigma = 1 - c_valid

        weight = 1. / (sigma + eps)
        return weight

    def get_x_noise(self):
        fsc = self.get_fsc_spectrum()
        power = smoothen_spectra(self.x_ctf_train_power_spectrum.data, kernel=5)
        noise_power = torch.clip(power * (1 - fsc), 0, 1)
        return noise_power

    def get_x_signal(self):
        fsc = self.get_fsc_spectrum()
        power = smoothen_spectra(self.x_ctf_train_power_spectrum.data, kernel=5)
        noise_power = torch.clip(power * fsc, 0, 1)
        return noise_power

    def get_scalar_summary(self):
        return {
            "Spectral means/x_noise": spectrum_to_grid_mean(self.get_x_noise()),
            "Spectral means/x_signal": spectrum_to_grid_mean(self.get_x_signal()),
            "Spectral means/y_weight": spectrum_to_grid_mean(self.get_y_weight()),
            "Spectral means/fsc": spectrum_to_grid_mean(self.get_fsc_spectrum())
        }

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
                {'y': self.x_ctf_train_power_spectrum.data, 'label': 'x ctf train', 'color': 'blue'},
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

    def get_state_dict(self) -> Dict:
        return {
            "type": "SpectralStatistics",
            "version": "0.0.1",

            "image_size": self.image_size,
            "c_train_spectrum": self.c_train_spectrum.data,
            "c_valid_spectrum": self.c_valid_spectrum.data,
            "x_ctf_train_power_spectrum": self.x_ctf_train_power_spectrum.data,
            "mse": self.mse.data,
            "c_train_ctf": self.c_train_ctf.data,
            "c_valid_ctf": self.c_valid_ctf.data,
            "x_train_power": self.x_train_power.data,
            "x_valid_power": self.x_valid_power.data,
            "y_power": self.y_power.data,
            "x_ctf_valid_power": self.x_ctf_valid_power.data,
            "y_ctf_power": self.y_ctf_power.data,
        }

    @staticmethod
    def load_from_state_dict(state_dict):
        if "type" not in state_dict or state_dict["type"] != "SpectralStatistics":
            raise TypeError("Input is not an 'SpectralStatistics' instance.")

        if "version" not in state_dict:
            raise RuntimeError("SpectralStatistics instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            container = SpectralStatistics(
                image_size=state_dict["image_size"]
            )

            container.c_train_spectrum.data = state_dict["c_train_spectrum"]
            container.c_valid_spectrum.data = state_dict["c_valid_spectrum"]
            container.x_ctf_train_power_spectrum.data = state_dict["x_ctf_train_power_spectrum"]
            container.mse.data = state_dict["mse"]
            container.c_train_ctf.data = state_dict["c_train_ctf"]
            container.c_valid_ctf.data = state_dict["c_valid_ctf"]
            container.x_train_power.data = state_dict["x_train_power"]
            container.x_valid_power.data = state_dict["x_valid_power"]
            container.y_power.data = state_dict["y_power"]
            container.x_ctf_valid_power.data = state_dict["x_ctf_valid_power"]
            container.y_ctf_power.data = state_dict["y_ctf_power"]

            return container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")