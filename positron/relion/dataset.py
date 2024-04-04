#!/usr/bin/env python

"""
Module for loading RELION particle datasets
"""

import os
import warnings
from glob import glob
import numpy as np

from typing import List

from positron.base.ctf import ContrastTransferFunction
from positron.base.single_particle_dataset import SingleParticleDataset
from positron.base.star_file import load_star
from .utils import find_project_root


class RelionDataset:
    def __init__(self, path: str = None, dtype: np.dtype = np.float32):
        self.dtype = dtype
        self.project_root = None
        self.data_star_path = None
        self.preload = None
        self.image_file_paths = []

        # In data star file
        self.part_random_subset = None
        self.part_rotation = None
        self.part_translation = None
        self.part_defocus = None
        self.part_og_idx = None
        self.part_stack_idx = None
        self.part_image_file_path_idx = None
        self.part_norm_correction = None
        self.part_group_id = None
        self.nr_particles = None

        # In data star file
        self.optics_groups = []
        self.optics_groups_ids = []

        if path is not None:
            self.load(path)

    def load(self, path: str) -> None:
        """
        Load data from path
        :param path: relion job directory or data file
        """
        if os.path.isfile(path):
            data_star_path = path
            root_search_path = os.path.dirname(os.path.abspath(path))
        else:
            data_star_path = os.path.abspath(self._find_star_file_in_path(path, "data"))
            root_search_path = os.path.abspath(path)

        self.data_star_path = os.path.abspath(data_star_path)
        data = load_star(self.data_star_path)

        if 'optics' not in data:
            raise RuntimeError("Optics groups table not found in data star file")
        if 'particles' not in data:
            raise RuntimeError("Particles table not found in data star file")

        self._load_optics_group(data['optics'])
        self._load_particles(data['particles'])

        self.project_root = find_project_root(root_search_path, self.image_file_paths[0])

        # Convert image paths to absolute paths
        for i in range(len(self.image_file_paths)):
            self.image_file_paths[i] = os.path.abspath(os.path.join(self.project_root, self.image_file_paths[i]))

        # TODO check cross reference integrity, e.g. all part_group_id exist in noise_group_id

    def make_particle_dataset(
            self,
            max_res: float = None
    ):
        mask = np.full(self.nr_particles, True)

        part_idx = np.arange(self.nr_particles)

        dataset = SingleParticleDataset()
        dataset.initialize(
            image_file_paths=self.image_file_paths,
            part_idx=part_idx[mask],
            part_random_subset=self.part_random_subset[mask],
            part_rotation=self.part_rotation[mask],
            part_translation=self.part_translation[mask],
            part_defocus=self.part_defocus[mask],
            part_og_idx=self.part_og_idx[mask],
            part_stack_idx=self.part_stack_idx[mask],
            part_image_file_path_idx=self.part_image_file_path_idx[mask],
            part_norm_correction=self.part_norm_correction[mask],
            part_group_id=self.part_group_id[mask],
            optics_group_stats=self.optics_groups,
            dtype=self.dtype,
            max_res=max_res
        )

        return dataset

    def _load_optics_group(self, optics: dict) -> None:
        if 'rlnOpticsGroup' not in optics:
            raise RuntimeError(
                "Optics group id (rlnOpticsGroup) is required, "
                "but was not found in optics group table."
            )

        if 'rlnImageSize' not in optics:
            raise RuntimeError(
                "Image size (rlnImageSize) is required, "
                "but was not found in optics group table."
            )

        if 'rlnImagePixelSize' not in optics:
            raise RuntimeError(
                "Image pixel size (rlnImagePixelSize) is required, "
                "but was not found in optics group table."
            )

        nr_optics = len(optics['rlnOpticsGroup'])

        for i in range(nr_optics):
            optics_group_id = int(optics['rlnOpticsGroup'][i])
            image_size = int(optics['rlnImageSize'][i])
            pixel_size = float(optics['rlnImagePixelSize'][i])

            if image_size <= 0 or image_size % 2 != 0:
                raise RuntimeError(
                    f"Invalid value ({image_size}) for image size of optics group {optics_group_id}.\n"
                    f"Image size must be even and larger than 0."
                )
            if pixel_size <= 0:
                raise RuntimeError(
                    f"Invalid value ({pixel_size}) for pixel size of optics group {optics_group_id}."
                )

            voltage = float(optics['rlnVoltage'][i]) \
                if 'rlnVoltage' in optics else None
            spherical_aberration = float(optics['rlnSphericalAberration'][i]) \
                if 'rlnSphericalAberration' in optics else None
            amplitude_contrast = float(optics['rlnAmplitudeContrast'][i]) \
                if 'rlnAmplitudeContrast' in optics else None

            self.optics_groups_ids.append(optics_group_id)
            self.optics_groups.append({
                "id": optics_group_id,
                "image_size": image_size,
                "pixel_size": pixel_size,
                "voltage": voltage,
                "spherical_aberration": spherical_aberration,
                "amplitude_contrast": amplitude_contrast
            })

    def _load_particles(self, particles: dict) -> None:
        if 'rlnImageName' not in particles:
            raise RuntimeError(
                "Image name (rlnImageName) is required, "
                "but was not found in particles table."
            )

        if 'rlnOpticsGroup' not in particles:
            raise RuntimeError(
                "Optics group id (rlnOpticsGroup) is required, "
                "but was not found in particles table."
            )

        nr_particles = len(particles['rlnImageName'])

        part_group_names = np.full(nr_particles, "", dtype=object)
        use_group_names = False

        self.part_og_idx = np.full(nr_particles, -1, dtype=int)
        self.part_norm_correction = np.full(nr_particles, 1., dtype=np.float32)
        self.part_group_id = np.full(nr_particles, -1, dtype=int)
        self.part_defocus = np.full([nr_particles, 3], 0, dtype=np.float32)
        self.part_rotation = np.full([nr_particles, 3], 0, dtype=np.float32)
        self.part_translation = np.full([nr_particles, 2], 0, dtype=np.float32)
        self.part_random_subset = np.full(nr_particles, -1, dtype=int)
        self.part_stack_idx = np.full(nr_particles, -1, dtype=int)
        self.part_image_file_path_idx = np.full(nr_particles, -1, dtype=int)

        self.nr_particles = nr_particles

        for i in range(nr_particles):
            # Optics group ---------------------------------------
            og_id = int(particles['rlnOpticsGroup'][i])
            og_idx = self.optics_groups_ids.index(og_id)
            self.part_og_idx[i] = og_idx
            og = self.optics_groups[og_idx]

            # Norm correction -------------------------------------
            if 'rlnNormCorrection' in particles:
                self.part_norm_correction[i] = float(particles['rlnNormCorrection'][i])

            # Group ----------------------------------------------
            if 'rlnGroupNumber' in particles:
                self.part_group_id[i] = int(particles['rlnGroupNumber'][i])
            else:
                use_group_names = True

            if 'rlnGroupName' in particles:
                part_group_names[i] = particles['rlnGroupName'][i]

            # CTF parameters -------------------------------------
            if 'rlnDefocusU' in particles and \
                    'rlnDefocusV' in particles and \
                    'rlnDefocusAngle' in particles:
                self.part_defocus[i, 0] = float(particles['rlnDefocusU'][i])
                self.part_defocus[i, 1] = float(particles['rlnDefocusV'][i])
                self.part_defocus[i, 2] = float(particles['rlnDefocusAngle'][i])

            # Rotation parameters --------------------------------
            if 'rlnAngleRot' in particles and \
                    'rlnAngleTilt' in particles and \
                    'rlnAnglePsi' in particles:
                self.part_rotation[i, 0] = float(particles['rlnAngleRot'][i]) * np.pi / 180.
                self.part_rotation[i, 1] = float(particles['rlnAngleTilt'][i]) * np.pi / 180.
                self.part_rotation[i, 2] = float(particles['rlnAnglePsi'][i]) * np.pi / 180.
            elif 'rlnAnglePsi' in particles:
                a = np.array([0., 0., float(particles['rlnAnglePsi'][i])])
                a *= np.pi / 180.
                self.part_rotation[i] = a

            # Translation parameters ------------------------------
            if 'rlnOriginXAngst' in particles and 'rlnOriginYAngst' in particles:
                scale = 1. / og['pixel_size']
                self.part_translation[i, 0] = float(particles['rlnOriginXAngst'][i]) * scale
                self.part_translation[i, 1] = float(particles['rlnOriginYAngst'][i]) * scale

            # Image data ------------------------------------------
            img_name = particles['rlnImageName'][i]
            img_tokens = img_name.split("@")
            if len(img_tokens) == 2:
                image_stack_id = int(img_tokens[0]) - 1
                img_path = img_tokens[1]
            elif len(img_tokens) == 1:
                image_stack_id = 0
                img_path = img_tokens[1]
            else:
                raise RuntimeError(f"Invalid image file name (rlnImageName): {img_name}")

            self.part_random_subset[i] = particles["rlnRandomSubset"][i]
            self.part_stack_idx[i] = image_stack_id

            try:  # Assume image file path has been added to list
                img_path_idx = self.image_file_paths.index(img_path)
                self.part_image_file_path_idx[i] = img_path_idx
            except ValueError:  # If image file path not found in existing list
                img_path_idx = len(self.image_file_paths)
                self.part_image_file_path_idx[i] = img_path_idx
                self.image_file_paths.append(img_path)

        if use_group_names:
            _, part_group_id = np.unique(part_group_names, return_inverse=True)
            for i in range(nr_particles):
                if self.part_group_id[i] < 0:
                    self.part_group_id[i] = part_group_id[i]

    @staticmethod
    def _find_star_file_in_path(path: str, type: str = "optimiser") -> str:
        if os.path.isfile(os.path.join(path, f"run_{type}.star")):
            return os.path.join(path, f"run_{type}.star")
        files = glob(os.path.join(path, f"*{type}.star"))
        if len(files) > 0:
            files = list.sort(files)
            return files[-1]

        raise FileNotFoundError(f"Could not find '{type}' star-file in path: {path}")

