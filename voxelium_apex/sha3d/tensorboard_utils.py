#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import os
from typing import List, TypeVar, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from voxelium import dt_desymmetrize, idft, make_imshow_fig, make_scatter_fig, pca_dim_reduction, make_heatmap_fig
from voxelium_apex.sha3d.hidden_variable_container import HiddenVariableContainer

Tensor = TypeVar('torch.tensor')


def tensor_to_np(tensor):
    return tensor.detach().cpu().numpy()


class TensorboardSummary:
    def __init__(self, logdir, pixel_size, spectral_size, step=0):
        self.summary_fn = os.path.join(logdir, "tb")
        self.summary = SummaryWriter(self.summary_fn)
        self.spectral_size = spectral_size
        self.step = step
        self.pixel_size = pixel_size

    def set_step(self, step: int = None):
        self.step = self.step + 1 if step is None else step

    def add_scalar(self, title, scalar):
        if torch.is_tensor(scalar):
            scalar = tensor_to_np(scalar)
        self.summary.add_scalar(title, scalar, self.step)

    def add_scalars(self, scalars: Dict):
        for key in scalars.keys():
            self.add_scalar(key, scalars[key])

    def add_figure(self, title, figure):
        self.summary.add_figure(title, figure, self.step)
        
    def add_histogram(self, name, data):
        self.summary.add_histogram(name, data, self.step)

    def write_stats(self, x_ft, y_ft, data_amp, data_ctf_amp):
        if x_ft.shape[-1] != 2:
            x_ft = torch.view_as_real(x_ft)
        if y_ft.shape[-1] != 2:
            y_ft = torch.view_as_real(y_ft)
        self.summary.add_scalar(f"Stats/X mean", tensor_to_np(x_ft.mean()), self.step)
        self.summary.add_scalar(f"Stats/X std", tensor_to_np(x_ft.std()), self.step)
        self.summary.add_scalar(f"Stats/Y mean", tensor_to_np(y_ft.mean()), self.step)
        self.summary.add_scalar(f"Stats/Y std", tensor_to_np(y_ft.std()), self.step)
        self.summary.add_scalar(f"Stats/data amp mean", tensor_to_np(data_amp.mean()), self.step)
        self.summary.add_scalar(f"Stats/data amp std", tensor_to_np(data_amp.std()), self.step)
        self.summary.add_scalar(f"Stats/data ctf amp mean", tensor_to_np(data_ctf_amp.mean()), self.step)
        self.summary.add_scalar(f"Stats/data ctf amp std", tensor_to_np(data_ctf_amp.std()), self.step)

    def write_images(self, x_ft, y_ft, ctf):
        x_ft = x_ft.detach()
        y_ft = y_ft.detach()
        ctf = ctf.detach()

        y_ft_ = tensor_to_np(torch.abs(torch.view_as_complex(y_ft[0])))
        c_ = torch.abs(ctf[0].detach()).data.cpu().numpy()
        c_std = np.std(c_)
        y_ft_std = np.std(y_ft_)
        if c_std != 0 and y_ft_std != 0:
            c_ /= c_std
            y_ft_ /= y_ft_std
            y_ft_[:c_.shape[0] // 2] = c_[:c_.shape[0] // 2]
        self.summary.add_figure(f"Data/CTF", make_imshow_fig(y_ft_), self.step)

        x_ft_ = tensor_to_np(dt_desymmetrize(torch.view_as_complex(x_ft[0]), dim=2))
        self.summary.add_figure(f"Output/FT", make_imshow_fig(np.abs(x_ft_)), self.step)

        x_ = idft(x_ft_, dim=2, real_in=True)
        self.summary.add_figure(f"Output/Image", make_imshow_fig(x_), self.step)

    def write_hidden_variable(self, hvc: HiddenVariableContainer):
        vars = hvc.vars

        v = torch.stack([vars['pose_alpha'].vars, vars['pose_beta'].vars, vars['pose_gamma'].vars], 1)
        o = torch.stack([vars['pose_alpha'].orig, vars['pose_beta'].orig, vars['pose_gamma'].orig], 1)
        e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
        self.summary.add_scalar(f"Hidden variables/pose error", e, self.step)

        v = torch.stack([vars['shift_x'].vars, vars['shift_y'].vars], 1)
        o = torch.stack([vars['shift_x'].orig, vars['shift_y'].orig], 1)
        e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
        self.summary.add_scalar(f"Hidden variables/shift error", e, self.step)

        if hvc.do_ctf:
            v = torch.stack([vars['ctf_defocus_u'].vars, vars['ctf_defocus_v'].vars], 1)
            o = torch.stack([vars['ctf_defocus_u'].orig, vars['ctf_defocus_v'].orig], 1)
            e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
            self.summary.add_scalar(f"Hidden variables/ctf defocus error", e, self.step)

            v = vars['ctf_angle'].vars
            o = vars['ctf_angle'].orig
            e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
            self.summary.add_scalar(f"Hidden variables/ctf angle error", e, self.step)

        z = hvc.get_metadata('z')
        if z is not None:
            if z.size(1) > 2:
                z = pca_dim_reduction(z, subsample=20000)
                # z = z[:, :2]

            fig = make_scatter_fig(x=z[:, 0], y=z[:, 1])
            self.summary.add_figure(f"Latent scatter", fig, self.step)

            z_ = z[:min(z.size(0), 200_000)]
            fig = make_heatmap_fig(x=z_[:, 0], y=z_[:, 1])
            self.summary.add_figure(f"Latent heatmap", fig, self.step)
