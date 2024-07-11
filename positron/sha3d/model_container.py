#!/usr/bin/env python3

"""
Model container for reconstruction
"""
import torch
import torch.nn as nn

from typing import List, TypeVar, Dict, Union

from positron.base import get_activation_function_by_name, ResidBlock
from .train_utils import parse_bounds_str

from ..base import ModelContainer, spectral_index_from_resolution
from ..base.explicit_grid_utils import size_to_maxr
from .base_optim import BaseOptimizer
from .structure_decoder import StructureDecoder


class Encoder(torch.nn.Module):
    def __init__(
            self,
            output_dim: int,
            input_dim: int,
            resid_dim: int = 128,
            resid_count: int = 3,
            activation=torch.nn.ELU(),
            normalize_fn=torch.nn.BatchNorm1d,
            init_factor: float = 1.
    ) -> None:
        super().__init__()

        self.initial_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, resid_dim),
            activation
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(resid_count):
            self.hidden_layers.append(ResidBlock(resid_dim, activation, normalize_fn, init_factor))

        self.final_linear = torch.nn.Linear(resid_dim, output_dim)

    def forward(self, x: torch.tensor, noise=0) -> torch.tensor:
        y = self.initial_layer(x)
        if noise > 0:
            y += torch.randn_like(y) * y.std().detach() * noise

        for hidden_layer in self.hidden_layers:
            y = hidden_layer(y)
            if noise > 0:
                y += torch.randn_like(y) * y.std().detach() * noise

        y = self.final_linear(y)

        return y


class ModelContainer(nn.Module):
    def __init__(
            self,
            z_size,
            s_size,
            feature_bandpass_arg,
            mse_bandpass_arg,
            image_size,
            voxel_size,
            circular_mask_radius_ang,
            circular_mask_thickness_ang,
            norm_network=False,
            z_encoder_dims=None,
            s_encoder_dims=None,
            train_epoch=0,
            train_step=0,
            lr=1e-3,
            wd=1e-2,
            features_mean=None,
            features_std=None,
            do_roi=False,
            s0_ema=None
    ) -> None:
        super().__init__()

        self.z_size = z_size
        self.s_size = s_size
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.circular_mask_radius_ang = circular_mask_radius_ang
        self.circular_mask_thickness_ang = circular_mask_thickness_ang

        self.train_epoch = train_epoch
        self.train_step = train_step

        self.lr = lr
        self.wd = wd

        self.do_roi = do_roi
        self.s0_ema = 0 if s0_ema is None else s0_ema

        self.max_r = size_to_maxr(image_size)
        self.grid3d_size = image_size + 1 - image_size % 2

        self.mse_bandpass_arg = mse_bandpass_arg
        self.mse_bandpass = None
        if mse_bandpass_arg is None:
            self.mse_bandpass = (1, self.max_r)
        else:
            highpass_ang, lowpass_ang = parse_bounds_str(mse_bandpass_arg)[0]
            if lowpass_ang <= voxel_size * 2:
                maxr = self.max_r
            else:
                maxr = spectral_index_from_resolution(lowpass_ang, image_size, voxel_size)
            minr = spectral_index_from_resolution(highpass_ang, image_size, voxel_size)
            minr = min(minr, maxr - 1)
            self.mse_bandpass = (minr, maxr)

        self.feature_bandpass_arg = feature_bandpass_arg
        if feature_bandpass_arg is None:
            self.feature_bandpass = [
                (1, self.max_r),
                (1, self.max_r // 2),
                (1, self.max_r // 3),
            ]
        else:
            self.feature_bandpass = []
            bandpass_ang = parse_bounds_str(feature_bandpass_arg)
            for bp in bandpass_ang:
                if bp is None:
                    if (0, self.max_r) not in self.feature_bandpass:
                        self.feature_bandpass.append((0, self.max_r))
                else:
                    highpass_ang, lowpass_ang = bp
                    if lowpass_ang <= voxel_size * 2:
                        maxr = self.max_r
                    else:
                        maxr = spectral_index_from_resolution(lowpass_ang, image_size, voxel_size)
                    minr = spectral_index_from_resolution(highpass_ang, image_size, voxel_size)
                    minr = min(minr, maxr - 1)
                    if (minr, maxr) not in self.feature_bandpass:
                        self.feature_bandpass.append((minr, maxr))

        feature_size = len(self.feature_bandpass) * (s_size - 1 if do_roi else s_size)

        self.z_encoder = Encoder(
            input_dim=feature_size,
            output_dim=z_size * 2,
            resid_dim=128,
            resid_count=4,
        )

        self.s_encoder = Encoder(
            input_dim=z_size,
            output_dim=s_size,
            resid_dim=128,
            resid_count=4,
        )

        self.norm_network = None
        if norm_network:
            self.norm_network = Encoder(
                input_dim=feature_size,
                output_dim=1,
                resid_dim=64,
                resid_count=4,
            )

        self.feature_size = feature_size
        self.z_encoder_dims = z_encoder_dims
        self.s_encoder_dims = s_encoder_dims

        # Structure Decoder
        self.decoder = StructureDecoder(
            grid3d_size=self.grid3d_size,
            s_size=s_size
        )

        self.decoder_opt = None
        self.lion_opt = None
        self.adam_opt = None

        self.features_mean = features_mean
        self.features_std = features_std

    def get_circular_mask_params(self):
        circular_mask_radius = self.circular_mask_radius_ang / self.voxel_size
        circular_mask_thickness = self.circular_mask_thickness_ang / self.voxel_size
        return circular_mask_radius, circular_mask_thickness

    def reparameterize(self, mu, logvar):
        logvar = logvar.clip(max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def normalize_features(self, features, eps=1e-6):
        if self.training:
            if self.features_mean is None:
                self.features_mean = features.mean(0, keepdim=True)
                self.features_std = features.std(0, keepdim=True)
            else:
                b = 0.1
                self.features_mean = self.features_mean * b + features.mean(0, keepdim=True) * (1 - b)
                self.features_std = self.features_std * b + features.std(0, keepdim=True) * (1 - b)
            return (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + eps)
        else:
            return (features - self.features_mean) / (self.features_std + eps)

    def z_encode(self, features, noise=0):
        nn = self.z_encoder(features, noise=noise)
        z = nn[:, :nn.size(1) // 2]
        log_var = nn[:, nn.size(1) // 2:]
        return z, log_var

    def s_encode(self, z, features=None, noise=0):
        s = self.s_encoder(z, noise=noise)
        if self.do_roi:
            if self.training:
                s0_mean = s[:, 0].mean()
                s[:, 0] = s0_mean
                self.s0_ema = self.s0_ema * 0.9 + s0_mean.detach().cpu().item() * 0.1
            else:
                s[:, 0] = self.s0_ema

        if self.norm_network is not None and features is not None:
            s = s * (torch.sigmoid(self.norm_network(features)) * 0.2 + 0.9)

        return s

    def init_optimizers(self):
        _, spectral_idx, _ = self.decoder._load_cache(self.decoder.max_r, True)
        params = [
            {'params': self.decoder.projector.weight, 'spectral_idx': spectral_idx, 'max_r': self.max_r}
        ]
        self.decoder_opt = BaseOptimizer(params)

        params = [
            {"params": self.z_encoder.parameters(), "lr": self.lr, "weight_decay": self.wd},
            {"params": self.s_encoder.parameters(), "lr": self.lr, "weight_decay": self.wd},
        ]
        if self.norm_network is not None:
            params.append( {"params": self.norm_network.parameters(), "lr": self.lr, "weight_decay": self.wd})
        self.adam_opt = torch.optim.AdamW(params)

    def clip_grad(self, clip):
        torch.nn.utils.clip_grad_norm_(self.z_encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.s_encoder.parameters(), clip)
        if self.norm_network is not None:
            torch.nn.utils.clip_grad_norm_(self.norm_network.parameters(), clip)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adam_opt.zero_grad(set_to_none)
        self.decoder_opt.zero_grad(set_to_none)

    def get_device(self):
        return next(self.parameters()).device

    def get_mse_weight_spectrum(self):
        w = torch.full([self.max_r], 1e-2)
        w[self.mse_bandpass[0]:self.mse_bandpass[1]] = 1.
        return w

    def get_state_dict(self) -> Dict:
        return {
            "type": "MnxVoxelContainer",
            "version": "0.0.1",

            "z_size": self.z_size,
            "s_size": self.s_size,
            "feature_bandpass_arg": self.feature_bandpass_arg,
            "mse_bandpass_arg": self.mse_bandpass_arg,

            "image_size": self.image_size,
            "voxel_size": self.voxel_size,
            "circular_mask_radius_ang": self.circular_mask_radius_ang,
            "circular_mask_thickness_ang": self.circular_mask_thickness_ang,

            "train_epoch": self.train_epoch,
            "train_step": self.train_step,

            "z_encoder_dims": self.z_encoder_dims,
            "s_encoder_dims": self.s_encoder_dims,

            "features_mean": self.features_mean,
            "features_std": self.features_std,

            "z_encoder": self.z_encoder.state_dict(),
            "s_encoder": self.s_encoder.state_dict(),
            "norm_network": None if self.norm_network is None else self.norm_network.state_dict(),

            "adam_opt": self.adam_opt.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_opt": self.decoder_opt.state_dict(),

            "do_roi": self.do_roi,
            "s0_ema": self.s0_ema
        }

    @staticmethod
    def load_from_state_dict(state_dict, skip_optimizers=False):
        if "type" not in state_dict or state_dict["type"] != "MnxVoxelContainer":
            raise TypeError("Input is not an 'MnxVoxelContainer' instance.")

        if "version" not in state_dict:
            raise RuntimeError("MnxVoxelContainer instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            container = ModelContainer(
                z_size=state_dict["z_size"],
                s_size=state_dict["s_size"],
                image_size=state_dict["image_size"],
                voxel_size=state_dict["voxel_size"],
                circular_mask_radius_ang=state_dict["circular_mask_radius_ang"],
                circular_mask_thickness_ang=state_dict["circular_mask_thickness_ang"],
                norm_network=state_dict["norm_network"] if state_dict["norm_network"] is not None else None,
                z_encoder_dims=state_dict["z_encoder_dims"],
                s_encoder_dims=state_dict["s_encoder_dims"],
                train_epoch=state_dict["train_epoch"],
                train_step=state_dict["train_step"],
                feature_bandpass_arg=state_dict["feature_bandpass_arg"],
                mse_bandpass_arg=state_dict["mse_bandpass_arg"],
                features_mean=state_dict["features_mean"],
                features_std=state_dict["features_std"],
                do_roi=state_dict["do_roi"],
                s0_ema=state_dict["s0_ema"]
            )

            container.z_encoder.load_state_dict(state_dict["z_encoder"])
            container.s_encoder.load_state_dict(state_dict["s_encoder"])
            container.decoder.load_state_dict(state_dict["decoder"])

            if container.norm_network is not None:
                container.norm_network.load_state_dict(state_dict["norm_network"])

            if not skip_optimizers:
                container.init_optimizers()
                container.adam_opt.load_state_dict(state_dict["adam_opt"])
                container.decoder_opt.load_state_dict(state_dict["decoder_opt"])

            return container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
