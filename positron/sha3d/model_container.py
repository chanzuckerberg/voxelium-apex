#!/usr/bin/env python3

"""
Model container for reconstruction
"""
import torch
import torch.nn as nn

from typing import List, TypeVar, Dict, Union

from positron.base import get_activation_function_by_name
from .retention_classifier import RetentionClassifier
from .utils import parse_bounds_str

from ..base import ModelContainer, spectral_index_from_resolution
from ..base.explicit_grid_utils import size_to_maxr
from .base_optim import BaseOptimizer
from .structure_decoder import StructureDecoder


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn) -> None:
        super().__init__()
        self.activation_fn = get_activation_function_by_name(activation_fn)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        pre_dim = hidden_dims[0]
        for dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(pre_dim, dim))
            pre_dim = dim

        self.output_layer = nn.Linear(pre_dim, output_dim)

    def forward(self, x: torch.tensor, dropout=0) -> torch.tensor:
        y = self.activation_fn(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            if dropout > 0:
                y = nn.functional.dropout(y, dropout)
            y = self.activation_fn(hidden_layer(y))

        if dropout > 0:
            y = nn.functional.dropout(y, dropout)

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
            train_step=0
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

        self.max_r = size_to_maxr(image_size)
        self.grid3d_size = image_size + 1 - image_size % 2

        self.feature_bandpass_arg = feature_bandpass_arg
        bandpass_ang = parse_bounds_str(feature_bandpass_arg)
        self.feature_bandpass = []
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
            hidden_dims=[64, 64, 32] if z_encoder_dims is None else z_encoder_dims,
            output_dim=z_size,
            activation_fn='elu'
        )

        self.s_encoder = Encoder(
            input_dim=z_size,
            hidden_dims=[32, 64, 64, 64] if z_encoder_dims is None else z_encoder_dims,
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

        self.feature_retention = RetentionClassifier(self.feature_size)
        self.z_retention = RetentionClassifier(self.z_size)
        self.s_retention = RetentionClassifier(self.s_size)

        self.decoder_opt = None
        self.lion_opt = None
        self.adam_opt = None
    
    def get_circular_mask_params(self):
        circular_mask_radius = self.circular_mask_radius_ang / self.voxel_size
        circular_mask_thickness = self.circular_mask_thickness_ang / self.voxel_size
        return circular_mask_radius, circular_mask_thickness
    
    def encode(self, x, dropout=0):
        z = self.z_encoder(x, dropout=dropout)

        z_ = z
        # if dropout > 0. and z.size(1) > 2:
        #     z1 = z[:, :2]
        #     z2 = torch.nn.functional.dropout(z[:, 2:], 0.5)
        #     z_ = torch.cat([z1, z2], 1)

        s = self.s_encoder(z_)
        return z, s

    def init_optimizers(self):
        _, spectral_idx, _ = self.decoder._load_cache(self.decoder.max_r, True)
        params = [
            {'params': self.decoder.projector.weight, 'spectral_idx': spectral_idx, 'max_r': self.max_r}
        ]
        self.decoder_opt = BaseOptimizer(params)

        params = [
            {"params": self.z_encoder.parameters(), "lr": 1e-3, "weight_decay": 1e-2},
            {"params": self.s_encoder.parameters(), "lr": 1e-3, "weight_decay": 1e-2},
        ]
        self.adam_opt = torch.optim.AdamW(params)

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

            "z_encoder": self.z_encoder.state_dict(),
            "s_encoder": self.s_encoder.state_dict(),

            "feature_retention": self.feature_retention.get_state_dict(),
            "z_retention": self.z_retention.get_state_dict(),
            "s_retention": self.s_retention.get_state_dict(),

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
                feature_bandpass_arg=state_dict["feature_bandpass_arg"]
            )

            container.z_encoder.load_state_dict(state_dict["z_encoder"])
            container.s_encoder.load_state_dict(state_dict["s_encoder"])
            container.decoder.load_state_dict(state_dict["decoder"])

            container.feature_retention = RetentionClassifier.load_from_state_dict(state_dict["feature_retention"])
            container.z_retention = RetentionClassifier.load_from_state_dict(state_dict["z_retention"])
            container.s_retention = RetentionClassifier.load_from_state_dict(state_dict["s_retention"])

            if not skip_optimizers:
                container.init_optimizers()
                container.adam_opt.load_state_dict(state_dict["adam_opt"])
                container.decoder_opt.load_state_dict(state_dict["decoder_opt"])

            return container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
