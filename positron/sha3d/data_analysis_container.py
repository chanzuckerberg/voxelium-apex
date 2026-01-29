#!/usr/bin/env python3

"""
Container for dataset analysis results
"""

import os
import sys
import warnings

import numpy as np

import torch

from positron.base.single_particle_dataset import SingleParticleDataset
from positron import relion
from positron.sha3d.distributed_processing import DistributedProcessing
from positron.sha3d.hidden_variable_container import HiddenVariableContainer, HiddenVariableModule
from positron.sha3d.train_utils import get_np_dtype, load_modules_from_logdir, save_modules_to_logdir
from positron.sha3d.model_container import ModelContainer


# Default hidden variable learning rates
HV_LEARNING_RATES = {
    'pose_alpha': 1e-3,
    'pose_beta': 1e-3,
    'pose_gamma': 1e-3,
    'shift_x': 1e-1,
    'shift_y': 1e-1,
    'ctf_defocus_u': 1e-2,
    'ctf_defocus_v': 1e-2,
    'ctf_angle': 1e-2
}


class DatasetAnalysisContainer:
    def __init__(
            self,
            reconstruction_container: ModelContainer,
            particle_dataset: SingleParticleDataset,
            hidden_variable_container: HiddenVariableContainer
    ) -> None:
        self.reconstruction_container = reconstruction_container
        self.particle_dataset = particle_dataset
        self.hidden_variable_container = hidden_variable_container

    @staticmethod
    def initialize_from_args(args, device=None):
        if device is None:
            device = DistributedProcessing.get_device()

        # Check if there is an existing checkpoint file
        found_checkpoint = os.path.isfile(os.path.join(args.log_dir, f"mnx.pt"))

        parse_dataset = True
        checkpoint_dac = None

        # Load dataset
        if found_checkpoint:
            checkpoint_dac = DatasetAnalysisContainer.load_from_logdir(args.log_dir, device=device)
            checkpoint_dac.reconstruction_container.to(device)
            print(
                f"Found existing checkpoint files at training step "
                f"{checkpoint_dac.reconstruction_container.train_step} and epoch "
                f"{checkpoint_dac.reconstruction_container.train_epoch}"
            )
            if args.input is None:
                parse_dataset = False
        
        
        if parse_dataset:
            print(f"Parsing input from path: {args.input}")

            relion_dataset = relion.RelionDataset(
                args.input,
                dtype=get_np_dtype(args.dtype),
            )
            dataset = relion_dataset.make_particle_dataset(
                max_res=args.max_data_resolution
            )

            image_size = dataset.get_output_image_size(0)
            pixel_size = dataset.get_output_pixel_size(0)

            if args.cache is not None:
                dataset.set_cache_root(os.path.join(args.cache, "train"))

            optics_groups = dataset.get_optics_group_stats()
            for i, og in enumerate(optics_groups):
                if dataset.get_output_image_size(i) != image_size:
                    raise RuntimeError(
                        f"Optics groups must have the same image size.\n"
                        f"But optics group '{og['id']}' has {og['image_size']} and "
                        f"'{optics_groups[0]['id']}' has {image_size}."
                    )
                if dataset.get_output_pixel_size(i) != pixel_size:
                    warnings.warn(
                        f"Optics groups must have the same pixel size.\n"
                        f"But optics group '{og['id']}' has {og['pixel_size']} A and "
                        f"'{optics_groups[0]['id']}' has {pixel_size} A.",
                        RuntimeWarning
                    )
        else:
            dataset = checkpoint_dac.particle_dataset
            print(f"Using existing dataset found in checkpoit file.")

        # Preload images
        if args.preload:
            print(f"\nPreloading data...")
            dataset.preload_images()

        if not found_checkpoint:
            ###############################################
            # Reconstruction Module
            ###############################################

            # Image pre-processing
            max_diameter_ang = image_size * pixel_size - args.circular_mask_thickness

            if args.particle_diameter is None:
                diameter_ang = image_size * 0.75 * pixel_size - args.circular_mask_thickness
                print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
            else:
                if args.particle_diameter > max_diameter_ang:
                    print(
                        f"WARNING: Specified particle diameter {round(args.particle_diameter)} angstrom is too large\n"
                        f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
                    )
                    diameter_ang = max_diameter_ang
                else:
                    diameter_ang = args.particle_diameter

            print("Setting up model...")

            try:
                z_encoder_dims = list(map(int, args.z_encoder_dims.split(',')))
            except:
                raise RuntimeError(f"Could not convert --z_encoder_dims {args.z_encoder_dims} "
                                   f"to a comma separated list of floats.")
            try:
                s_encoder_dims = list(map(int, args.z_encoder_dims.split(',')))
            except:
                raise RuntimeError(f"Could not convert --s_encoder_dims {args.s_encoder_dims} "
                                   f"to a comma separated list of floats.")

            reconstruction_container = ModelContainer(
                z_size=args.z_size,
                s_size=args.s_size,
                mse_bandpass_arg=args.mse_bandpass,
                feature_bandpass_arg=args.feature_bandpass,
                image_size=image_size,
                voxel_size=pixel_size,
                circular_mask_radius_ang=diameter_ang,
                circular_mask_thickness_ang=args.circular_mask_thickness,
                z_encoder_dims=z_encoder_dims,
                s_encoder_dims=s_encoder_dims,
                do_roi=args.roi_mask is not None
            )

            reconstruction_container.to(device)
            reconstruction_container.init_optimizers()
        else:
            reconstruction_container = checkpoint_dac.reconstruction_container
            reconstruction_container.to(device)

        if parse_dataset:
            ###############################################
            # HIDDEN VARIABLE CONTAINER
            ###############################################
            vars = {}
            optics_groups = []

            # Poses
            euler_angles = dataset.part_rotation
            euler_angles = torch.Tensor(euler_angles).float().detach()
            vars['pose_alpha'] = HiddenVariableModule(euler_angles[:, 0], norm=np.pi, mean=0)
            vars['pose_beta'] = HiddenVariableModule(euler_angles[:, 1], norm=np.pi / 2,
                                                     mean=np.pi / 2)  # Tilt [0, 180]
            vars['pose_gamma'] = HiddenVariableModule(euler_angles[:, 2], norm=np.pi, mean=0)

            # Shifts
            shifts = dataset.part_translation
            shifts = torch.Tensor(shifts).float().detach()
            vars['shift_x'] = HiddenVariableModule(shifts[:, 0], norm=1, mean=0)
            vars['shift_y'] = HiddenVariableModule(shifts[:, 1], norm=1, mean=0)

            og_stats = dataset.get_optics_group_stats()
            for i in range(len(og_stats)):
                optics_groups.append({
                    'id': og_stats[i]['id'],
                    'pixel_size': dataset.get_output_pixel_size(i)
                })

            # CTFs
            do_ctf = dataset.part_defocus is not None and not np.any(np.isnan(dataset.part_defocus))
            if do_ctf:
                ctf_functions = dataset.get_optics_group_ctfs()
                for i in range(len(ctf_functions)):
                    optics_groups[i]['ctf'] = ctf_functions[i]

                defocus = dataset.part_defocus

                ctf_defocus = torch.Tensor(defocus[:, :2]).detach().float()
                defocus_norm = float(torch.std(ctf_defocus))
                defocus_mean = float(torch.mean(ctf_defocus))
                vars['ctf_defocus_u'] = HiddenVariableModule(ctf_defocus[:, 0], norm=defocus_norm, mean=defocus_mean)
                vars['ctf_defocus_v'] = HiddenVariableModule(ctf_defocus[:, 1], norm=defocus_norm, mean=defocus_mean)

                ctf_angle = torch.Tensor(defocus[:, 2]).detach().float()
                if torch.min(ctf_angle) >= 0:
                    vars['ctf_angle'] = HiddenVariableModule(ctf_angle, norm=360., mean=180.)
                else:
                    vars['ctf_angle'] = HiddenVariableModule(ctf_angle, norm=180., mean=0.)

            og_idx = dataset.part_og_idx
            og_idx = torch.Tensor(og_idx)

            hidden_variable_container = HiddenVariableContainer(
                vars=vars,
                op_learning_rates=HV_LEARNING_RATES,
                optics_groups=optics_groups,
                part_og_idx=og_idx,
                image_size=image_size,
                batch_size=args.batch_size
            )
            hidden_variable_container.set_device(device)
            hidden_variable_container.init_optimizers()

            if found_checkpoint:
                hidden_variable_container.mirror_og_stats(checkpoint_dac.hidden_variable_container)

        ###############################################
        # FINALIZE
        ###############################################

        data_analysis_container = DatasetAnalysisContainer(
            reconstruction_container=reconstruction_container,
            particle_dataset=dataset,
            hidden_variable_container=hidden_variable_container
        )

        return data_analysis_container

    def get_dataset_size(self):
        return self.particle_dataset.get_size()

    def save_to_logdir(self, path, state_label=None):
        save_modules_to_logdir(path, {
            "mnx" if state_label is None else f"mnx_{state_label}":
                self.reconstruction_container.get_state_dict(),
            "hvc" if state_label is None else f"hvc_{state_label}":
                self.hidden_variable_container.get_state_dict(),
            "data":
                self.particle_dataset.get_state_dict()
        })

    @staticmethod
    def load_from_logdir(path, device="cpu", state=None):
        rec, hvc, data = load_modules_from_logdir(
            path,
            [
                "mnx" if state is None else f"mnx_{state}",
                "hvc" if state is None else f"hvc_{state}",
                "data"
             ]
        )

        reconstruction_container = ModelContainer.load_from_state_dict(rec)
        reconstruction_container = reconstruction_container.to(device)

        hidden_variable_container = HiddenVariableContainer.load_from_state_dict(hvc)
        hidden_variable_container.set_device(device)

        particle_dataset = SingleParticleDataset()
        particle_dataset.set_state_dict(data)

        return DatasetAnalysisContainer(
            reconstruction_container=reconstruction_container,
            hidden_variable_container=hidden_variable_container,
            particle_dataset=particle_dataset,
        )