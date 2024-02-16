#!/usr/bin/env python3

"""
Model container for reconstruction
"""
import torch
import torch.nn as nn

from typing import List, TypeVar, Dict, Union

from positron.base import get_activation_function_by_name
from .train_utils import parse_bounds_str

from ..base import ModelContainer, spectral_index_from_resolution
from ..base.explicit_grid_utils import size_to_maxr
from .base_optim import BaseOptimizer
from .structure_decoder import StructureDecoder


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn, batch_norm=False) -> None:
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.activation_fn = get_activation_function_by_name(activation_fn)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.batch_norm = batch_norm
        self.input_batchnorm = nn.BatchNorm1d(hidden_dims[0]) if batch_norm else None

        self.hidden_layers = nn.ModuleList()
        self.hidden_batchnorms = nn.ModuleList()
        pre_dim = hidden_dims[0]
        for dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(pre_dim, dim))
            self.hidden_batchnorms.append(nn.BatchNorm1d(dim) if batch_norm else None)
            pre_dim = dim

        self.output_layer = nn.Linear(pre_dim, output_dim)

    def forward(self, x: torch.tensor, noise=0) -> torch.tensor:
        y = x
        # y = self.input_bn(y)
        y = self.input_layer(y)
        if self.batch_norm:
            y = self.input_batchnorm(y)
        y = self.activation_fn(y)
        if noise > 0:
            y += torch.randn_like(y) * noise

        for hidden_layer, batchnorm in zip(self.hidden_layers, self.hidden_batchnorms):
            y = hidden_layer(y)
            if self.batch_norm:
                y = batchnorm(y)
            y = self.activation_fn(y)
            if noise > 0:
                y += torch.randn_like(y) * noise

        return self.output_layer(y)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn, batch_norm=False) -> None:
        super().__init__()
        self.activation_fn = get_activation_function_by_name(activation_fn)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.batch_norm = batch_norm
        self.input_batchnorm = nn.BatchNorm1d(hidden_dims[0]) if batch_norm else None

        self.hidden_layers = nn.ModuleList()
        self.hidden_batchnorms = nn.ModuleList()
        pre_dim = hidden_dims[0]
        for dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(pre_dim, dim))
            self.hidden_batchnorms.append(nn.BatchNorm1d(dim) if batch_norm else None)
            pre_dim = dim

        self.output_layer = nn.Linear(pre_dim, output_dim)
        self.final_bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.tensor, noise=0) -> torch.tensor:
        y = self.input_layer(x)
        if self.batch_norm:
            y = self.input_batchnorm(y)
        y = self.activation_fn(y)
        if noise > 0:
            y += torch.randn_like(y) * noise

        for hidden_layer, batchnorm in zip(self.hidden_layers, self.hidden_batchnorms):
            y = hidden_layer(y)
            if self.batch_norm:
                y = batchnorm(y)
            y = self.activation_fn(y)
            if noise > 0:
                y += torch.randn_like(y) * noise

        return self.output_layer(y)


class ModelContainer(nn.Module):
    def __init__(
            self,
            z_size,
            s_size,
            feature_bandpass_arg,
            image_size,
            voxel_size,
            circular_mask_radius_ang,
            circular_mask_thickness_ang,
            z_encoder_dims=None,
            s_encoder_dims=None,
            train_epoch=0,
            train_step=0,
            lr=1e-3,
            wd=1e-2,
            features_mean=None,
            features_std=None
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

        self.max_r = size_to_maxr(image_size)
        self.grid3d_size = image_size + 1 - image_size % 2

        self.feature_bandpass_arg = feature_bandpass_arg
        if feature_bandpass_arg is None:
            self.feature_bandpass = [
                (2, self.max_r),
                (2, self.max_r // 2),
                (2, self.max_r // 3),
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
                    minr = spectral_index_from_resolution(highpass_ang, image_size, voxel_size)
                    if lowpass_ang <= voxel_size * 2:
                        maxr =self.max_r
                    else:
                        maxr = spectral_index_from_resolution(lowpass_ang, image_size, voxel_size)
                    if (minr, maxr) not in self.feature_bandpass:
                        self.feature_bandpass.append((minr, maxr))

        feature_size = len(self.feature_bandpass) * s_size

        self.z_encoder = Encoder(
            input_dim=feature_size,
            hidden_dims=[128, 128, 128] if z_encoder_dims is None else z_encoder_dims,
            output_dim=z_size * 2,
            activation_fn='elu'
        )

        self.s_encoder = Decoder(
            input_dim=z_size,
            hidden_dims=[128, 128, 128, 128] if z_encoder_dims is None else z_encoder_dims,
            output_dim=s_size,
            activation_fn='elu'
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

    def encode(self, x, noise=0):
        out = self.z_encoder(x, noise=noise)
        mu = out[:, :out.size(1) // 2]
        log_var = out[:, out.size(1) // 2:]
        return mu, log_var

    def vae(self, x, noise=0):
        mu, log_var = self.encode(x, noise=noise)
        z = mu if noise == 0 else self.reparameterize(mu, log_var)

        s = self.s_encoder(z, noise=noise)
        s = s / (s.square().sum(1, keepdim=True).sqrt() + 1e-12)

        return z, s, mu, log_var

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
        self.adam_opt = torch.optim.AdamW(params)

    def clip_grad(self, clip):
        torch.nn.utils.clip_grad_norm_(self.z_encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.s_encoder.parameters(), clip)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adam_opt.zero_grad(set_to_none)
        self.decoder_opt.zero_grad(set_to_none)

    def get_device(self):
        return next(self.parameters()).device

    def get_state_dict(self) -> Dict:
        return {
            "type": "MnxVoxelContainer",
            "version": "0.0.1",
            
            "z_size": self.z_size,
            "s_size": self.s_size,
            "feature_bandpass_arg": self.feature_bandpass_arg,

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

            "adam_opt": self.adam_opt.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_opt": self.decoder_opt.state_dict(),
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
                z_encoder_dims=state_dict["z_encoder_dims"],
                s_encoder_dims=state_dict["s_encoder_dims"],
                train_epoch=state_dict["train_epoch"],
                train_step=state_dict["train_step"],
                feature_bandpass_arg=state_dict["feature_bandpass_arg"],
                features_mean=state_dict["features_mean"],
                features_std=state_dict["features_std"],
            )

            container.z_encoder.load_state_dict(state_dict["z_encoder"])
            container.s_encoder.load_state_dict(state_dict["s_encoder"])
            container.decoder.load_state_dict(state_dict["decoder"])

            if not skip_optimizers:
                container.init_optimizers()
                container.adam_opt.load_state_dict(state_dict["adam_opt"])
                container.decoder_opt.load_state_dict(state_dict["decoder_opt"])

            return container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
