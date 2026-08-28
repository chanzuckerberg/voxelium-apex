"""PyTorch implementation of the base optimizer."""

import torch
from torch.optim.optimizer import Optimizer

from voxelium import grid_spectral_average, spectra_to_grid

from voxelium_apex.sha3d.train_utils import smoothen_spectra


class BaseOptimizer(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the optimizer hyperparameters.

        Args:
            params (iterable): Parameters to optimize or dictionaries defining
                parameter groups.
            lr (float, optional): Learning rate. Default: 1e-4.
            betas (Tuple[float, float], optional): Coefficients used for the
                running averages. Default: (0.9, 0.99).
            weight_decay (float, optional): Weight decay coefficient. Default: 0.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.init_devices = True

    @torch.no_grad()
    def step(self, fsc_spectrum=None):
        """Perform a single optimization step.

        Returns:
            None
        """
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                num_bases = parameter.shape[1]
                gradient = parameter.grad
                state = self.state[parameter]

                # Move optimizer state to the parameter device, or initialize it
                # there when it does not yet exist.
                if self.init_devices:
                    device = parameter.device

                    state["exp_avg"] = (
                        state["exp_avg"].to(device)
                        if "exp_avg" in state
                        else torch.zeros_like(parameter)
                    )

                    state["spectral_power"] = (
                        state["spectral_power"].to(device)
                        if "spectral_power" in state
                        else torch.ones(
                            [num_bases, group["max_r"]],
                            device=device,
                        )
                    )

                    group["spectral_idx"] = (
                        group["spectral_idx"].to(device)
                        if "spectral_idx" in group
                        else group["spectral_idx"].to(device)
                    )

                spectral_idx = group["spectral_idx"]
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Estimate the frequency-dependent update scale by averaging
                # spectral power across bases and smoothing the resulting spectrum.
                update_scale = state["spectral_power"].mean(0)
                update_scale = smoothen_spectra(
                    update_scale[None],
                    kernel=10,
                )[0]

                # Rescale the current parameters according to the FSC spectrum.
                spectrum_scale = (
                    1
                    - group["lr"]
                    * update_scale
                    * (1 - fsc_spectrum)
                )
                rescale_grid = spectra_to_grid(
                    spectrum_scale,
                    spectral_idx,
                )
                parameter.mul_(rescale_grid[:, None, None])

                # Apply the signed gradient/momentum update.
                update_spectrum = (
                    -group["lr"]
                    * update_scale
                    * fsc_spectrum
                )
                rescale_grid = spectra_to_grid(
                    update_spectrum,
                    spectral_idx,
                )
                update = torch.sign(
                    exp_avg * beta1
                    + gradient * (1 - beta1)
                ) * rescale_grid[:, None, None]

                parameter.add_(update)

                # Recompute spectral power for each basis and accumulate the
                # parameter magnitude used for global normalization.
                parameter_scale = 0.0

                for basis_idx in range(num_bases):
                    parameter_square = parameter[:, basis_idx].square()
                    parameter_scale += parameter_square.mean()

                    state["spectral_power"][basis_idx] = (
                        grid_spectral_average(
                            parameter_square.mean(-1),
                            spectral_idx,
                        ).sqrt()
                    )

                    # Edge artifacts distort the final two spectral bins.
                    state["spectral_power"][basis_idx][-2:] = (
                        state["spectral_power"][basis_idx][-4:-2].mean()
                    )

                # Normalize both the parameters and momentum by the same factor.
                parameter_scale = 1.0 / (
                    parameter_scale.sqrt() + 1e-12
                )
                parameter.mul_(parameter_scale)
                exp_avg.mul_(parameter_scale)

                # parameter.mul_(
                #     torch.sqrt(
                #         1 / (
                #             state["spectral_power"].mean(dim=1)[None, :, None]
                #             + 1e-12
                #         )
                #     )
                # )

                # Update the momentum running average.
                exp_avg.mul_(beta2).add_(
                    gradient,
                    alpha=1 - beta2,
                )

    def get_stats(self):
        spectral_power = self.state[
            self.param_groups[0]["params"][0]
        ]["spectral_power"]

        stats = []

        for basis_idx in range(spectral_power.shape[0]):
            stats.append(
                {
                    "y": spectral_power[basis_idx],
                    "color": "lightgrey",
                }
            )

        update_scale = spectral_power.mean(0)
        update_scale = smoothen_spectra(
            update_scale[None],
            kernel=10,
        )[0]

        stats.append(
            {
                "y": update_scale,
                "color": "black",
            }
        )

        return stats