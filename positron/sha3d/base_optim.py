# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch implementation of the Lion optimizer."""
import torch
from torch.optim.optimizer import Optimizer

from positron.base import spectra_to_grid, grid_spectral_average
from positron.sha3d.train_utils import smoothen_spectra


class BaseOptimizer(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, fsc_spectrum=None):
        """Performs a single optimization step.
        Returns:
            None
        """

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                nr_base = p.shape[1]
                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    group['spectral_idx'] = group['spectral_idx'].to(p.device)
                    state['spectral_power'] = torch.ones([nr_base, group['max_r']], device=p.device)

                sidx = group['spectral_idx']
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                update_scale = state['spectral_power'].mean(0)
                update_scale = smoothen_spectra(update_scale[None], kernel=10)[0]

                s = 1 - group['lr'] * update_scale * (1 - fsc_spectrum)
                rescale_grid = spectra_to_grid(s, sidx)
                p.mul_(rescale_grid[:, None, None])

                # Weight update
                s = -group['lr'] * update_scale * fsc_spectrum
                rescale_grid = spectra_to_grid(s, sidx)
                update = torch.sign(exp_avg * beta1 + grad * (1 - beta1)) * rescale_grid[:, None, None]

                p.add_(update)

                scale = 0.
                for i in range(nr_base):
                    p_square = p[:, i].square()
                    scale += p_square.mean()
                    state['spectral_power'][i] = grid_spectral_average(p_square.mean(-1), sidx).sqrt()
                    # Because of edge artefacts, the last 2 elements will be distorted
                    state['spectral_power'][i][-2:] = state['spectral_power'][i][-4:-2].mean()

                scale = 1. / (scale.sqrt() + 1e-12)
                p.mul_(scale)
                exp_avg.mul_(scale)

                # p.mul_(torch.sqrt(1 / (state['spectral_power'].mean(dim=1)[None, :, None] + 1e-12)))

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)



    def get_stats(self):
        spectral_power = self.state[self.param_groups[0]['params'][0]]['spectral_power']
        list = []
        for i in range(spectral_power.shape[0]):
            list.append({"y": spectral_power[i], 'color': 'lightgrey'})

        update_scale = spectral_power.mean(0)
        update_scale = smoothen_spectra(update_scale[None], kernel=10)[0]
        list.append({"y": update_scale, 'color': 'black'})

        return list

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
