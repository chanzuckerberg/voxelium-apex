#!/usr/bin/env python3

"""
Module for managing pytorch dataset of particles.
"""

import os
import shutil
import warnings
from typing import Any, List, Dict

import numpy as np
import mrcfile
from scipy.ndimage import shift
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import tqdm

from positron.base import rescale_real
from positron.base.ctf import ContrastTransferFunction

from collections import namedtuple
from multiprocessing import Process, Value, Array


class SingleParticleDataset(Dataset):
    def __init__(self) -> None:
        self.image_file_paths = None

        self.part_idx = None
        self.part_random_subset = None
        self.part_rotation = None
        self.part_translation = None
        self.part_defocus = None
        self.part_og_idx = None
        self.part_stack_idx = None
        self.part_image_file_path_idx = None
        self.part_norm_correction = None
        self.part_group_id = None
        self.part_group_idx = None
        self.part_tomo_idx = None
        self.part_preloaded_image = None
        self.nr_parts = None
        self.nr_noise_groups = None
        self.dtype = np.float32

        self.max_res = None

        self.has_ctf = None
        self.compute_ctf = True

        # Dictionaries mapping optics group id to data
        self.optics_group_stats = []
        self.optics_group_ctfs = []

        self.cache_root = None
        self.shared_cached_file = None
        self.shared_cached_pos = None
        self.shared_cached_size = None
        self.shared_cached_side = None

        self.data_preloaded = False
        self.data_caching = False

    def initialize(
            self,
            image_file_paths: np.array,
            part_idx: np.array,
            part_random_subset: np.array,
            part_rotation: np.array,
            part_translation: np.array,
            part_defocus: np.array,
            part_og_idx: np.array,
            part_stack_idx: np.array,
            part_image_file_path_idx: np.array,
            part_norm_correction: np.array,
            part_group_id: np.array,
            optics_group_stats: np.array,
            part_tomo_id: np.array = None,
            dtype: np.dtype = np.float32,
            max_res: float = None,
    ) -> None:
        self.image_file_paths = image_file_paths
        self.part_idx = part_idx
        self.part_random_subset = part_random_subset
        self.part_rotation = part_rotation
        self.part_translation = part_translation
        self.part_defocus = part_defocus
        self.part_og_idx = part_og_idx
        self.part_stack_idx = part_stack_idx
        self.part_image_file_path_idx = part_image_file_path_idx
        self.part_norm_correction = part_norm_correction
        self.part_group_id = part_group_id
        self.optics_group_stats = optics_group_stats
        self.dtype = dtype
        self.max_res = max_res

        # convert unique ids to indices
        d = defaultdict(count(0).__next__)
        self.part_group_idx = np.array([d[k] for k in self.part_group_id])
        self.nr_noise_groups = len(list(set(self.part_group_idx)))
        self.nr_parts = len(self.part_idx)

        if part_tomo_id is None:
            self.part_tomo_idx = self.part_group_idx
        else:
            d = defaultdict(count(0).__next__)
            self.part_tomo_idx = np.array([d[k] for k in part_tomo_id])

        if np.all(np.isnan(self.part_defocus)):
            self.has_ctf = False
        else:
            self.has_ctf = True
            self.setup_ctfs()

        self.setup_cache_resources()

    def setup_ctfs(self, h_sym: bool = False, compute_ctf: bool = None):
        if self.part_defocus is None:
            return

        if compute_ctf is not None:
            self.compute_ctf = compute_ctf

        for og in self.optics_group_stats:
            if og["voltage"] is not None or \
                    og["spherical_aberration"] is not None or \
                    og["amplitude_contrast"] is not None:
                ctf = ContrastTransferFunction(
                    og["voltage"],
                    og["spherical_aberration"],
                    og["amplitude_contrast"]
                )
            else:
                ctf = None
                warnings.warn(f"WARNING: CTF parameters missing for optics group ID: {id}", RuntimeWarning)

            self.optics_group_ctfs.append(ctf)

    def get_nr_optics_group(self):
        return len(self.optics_group_stats)

    def get_optics_group_stats(self):
        return self.optics_group_stats

    def get_optics_group_ctfs(self):
        return self.optics_group_ctfs

    def get_nr_noise_groups(self):
        return self.nr_noise_groups

    def get_size(self):
        return self.nr_parts

    def set_cache_root(self, path):
        self.cache_root = path
        if os.path.isdir(self.cache_root):
            shutil.rmtree(self.cache_root)
        if os.path.isfile(self.cache_root):
            raise RuntimeError("Cache path is a file.")
        else:
            os.makedirs(self.cache_root)

        self.setup_cache_resources()

    def setup_cache_resources(self):
        if self.cache_root is None or self.nr_parts is None or self.data_preloaded:
            return
        self.shared_cached_file = Array('i', [-1] * self.nr_parts)
        self.shared_cached_pos = Array('i', [-1] * self.nr_parts)
        self.shared_cached_size = Array('i', [-1] * self.nr_parts)
        self.shared_cached_side = Array('i', [-1] * self.nr_parts)
        self.data_caching = True

    def append_to_cache(self, index, data):
        process_info = torch.utils.data.get_worker_info()

        pid = 0 if process_info is None else process_info.id
        file_path = os.path.join(self.cache_root, "process_" + str(pid) + ".dat")
        if not os.path.isfile(file_path):
            file = open(file_path, 'wb')
        else:
            file = open(file_path, 'ab')

        file.seek(0, os.SEEK_END)  # Seek to end of file
        pos = file.tell()

        self.shared_cached_size[index] = data.size * data.itemsize  # Get data size in bytes
        self.shared_cached_pos[index] = pos
        self.shared_cached_file[index] = pid
        self.shared_cached_side[index] = data.shape[-1]

        file.write(data)
        file.close()

    def load_from_cache(self, index):
        if self.shared_cached_file[index] < 0 or \
                self.shared_cached_pos[index] < 0 or \
                self.shared_cached_size[index] <= 0:
            return None

        pid = self.shared_cached_file[index]
        file_path = os.path.join(self.cache_root, "process_" + str(pid) + ".dat")
        with open(file_path, 'r+b') as file:
            file.seek(self.shared_cached_pos[index], os.SEEK_SET)
            data = file.read(self.shared_cached_size[index])
        data = np.frombuffer(data, dtype=self.dtype)
        side = self.shared_cached_side[index]
        data = data.reshape(side, side)
        return data

    def get_output_image_size(self, optics_group_index):
        image_size = self.optics_group_stats[optics_group_index]["image_size"]
        if self.max_res is None:
            return image_size
        pixel_size = self.optics_group_stats[optics_group_index]["pixel_size"]
        max_res = self.max_res if self.max_res / 2 > pixel_size else pixel_size * 2
        max_r = round(image_size * pixel_size / max_res)
        return max_r * 2

    def get_output_pixel_size(self, optics_group_index):
        pixel_size = self.optics_group_stats[optics_group_index]["pixel_size"]
        if self.max_res is None:
            return pixel_size
        else:
            return max(self.max_res / 2, pixel_size)

    def rescale_image(self, index, image):
        output_image_size = self.get_output_image_size(self.part_og_idx[index])
        return rescale_real(image, output_image_size)

    def preload_images(self, verbose=True):
        self.part_preloaded_image = [None for _ in range(self.nr_parts)]
        part_index_list = np.arange(self.nr_parts)
        unique_file_idx, unique_reverse = np.unique(self.part_image_file_path_idx, return_inverse=True)

        pbar = None
        if verbose:
            pbar = tqdm.tqdm(total=self.nr_parts, smoothing=0.1)

        for i in range(len(unique_file_idx)):
            file_idx = unique_file_idx[i]
            path = self.image_file_paths[file_idx]
            with mrcfile.mmap(path, 'r') as mrc:
                # Mask out particles with no images in this file stack
                this_file_mask = unique_reverse == i
                this_file_stack_indices = self.part_stack_idx[this_file_mask]
                this_file_index_list = part_index_list[this_file_mask]  # Particles indices with images in this file
                
                # Since this_file_stack_indices indexes into the mmap object, we should make sure
                # it is sorted, so we minimize disk accesses
                stack_indices_argsort = np.argsort(this_file_stack_indices)
                for j in range(len(this_file_stack_indices)):
                    k = stack_indices_argsort[j]
                    idx = this_file_index_list[k]
                    # Take slices of images for this data set

                    if len(mrc.data.shape) == 2:
                        self.part_preloaded_image[idx] = mrc.data.astype(self.dtype).copy()
                    elif len(mrc.data.shape) == 3:
                        self.part_preloaded_image[idx] = \
                            mrc.data[this_file_stack_indices[k]].astype(self.dtype).copy()
                    else:
                        raise RuntimeError(f"Unsupported data dimensionality (dim={len(mrc.data.shape)}) "
                                           f"in file {path}.")
                    if verbose:
                        pbar.update()

        if verbose:
            pbar.close()

        self.data_preloaded = True
        self.data_caching = False

    def load_image(self, index):
        image_file_path_idx = self.part_image_file_path_idx[index]
        image_filename = self.image_file_paths[image_file_path_idx]
        image = None
        if self.data_preloaded:
            image = self.part_preloaded_image[index]
        elif self.data_caching:
            image = self.load_from_cache(index)

        if image is None:
            with mrcfile.mmap(image_filename, 'r') as mrc:
                stack_idx = self.part_stack_idx[index]
                if len(mrc.data.shape) == 2:
                    image = mrc.data.astype(self.dtype).copy()
                elif len(mrc.data.shape) == 3:
                    image = mrc.data[stack_idx].astype(self.dtype)
                else:
                    raise RuntimeError(f"Unsupported data dimensionality (dim={len(mrc.data.shape)}).")

        if self.max_res is not None:
            image = self.rescale_image(index, image)

        if self.data_caching and self.shared_cached_size[index] <= 0:
            self.append_to_cache(index, image)

        return image

    def __getitem__(self, index):
        image_np = self.load_image(index)
        image = torch.from_numpy(image_np.astype(np.float32))
        og_idx = self.part_og_idx[index]

        data = {
            "image": image,
            "rotation": torch.from_numpy(self.part_rotation[index]),
            "translation": torch.from_numpy(self.part_translation[index]),
            "idx": self.part_idx[index],
            "optics_group_idx": og_idx,
            "group_idx": self.part_group_idx[index],
            "tomo_idx": self.part_tomo_idx[index]
        }
        
        if self.compute_ctf:
            if not self.has_ctf or self.optics_group_ctfs[og_idx] is None:
                data["ctf"] = torch.ones_like(image)
            else:
                data["ctf"] = torch.from_numpy(
                    self.optics_group_ctfs[og_idx](
                        self.optics_group_stats[og_idx]["image_size"],
                        self.optics_group_stats[og_idx]["pixel_size"],
                        torch.from_numpy([self.part_defocus[index][0]]),
                        torch.from_numpy([self.part_defocus[index][1]]),
                        torch.from_numpy([self.part_defocus[index][2]])
                    )
                ).squeeze(0)
        
        return data

    def __len__(self):
        return self.nr_parts

    def get_state_dict(self) -> Dict:
        return {
            "type": "ParticleDataset",
            "version": "0.0.1",
            "image_file_paths": self.image_file_paths,
            "part_idx": self.part_idx,
            "part_random_subset": self.part_random_subset,
            "part_rotation": self.part_rotation,
            "part_translation": self.part_translation,
            "part_defocus": self.part_defocus,
            "part_og_idx": self.part_og_idx,
            "part_stack_idx": self.part_stack_idx,
            "part_image_file_path_idx": self.part_image_file_path_idx,
            "part_norm_correction": self.part_norm_correction,
            "part_group_id": self.part_group_id,
            "optics_group_stats": self.optics_group_stats,
            "max_res": self.max_res,
        }

    def set_state_dict(self, state_dict):
        if "type" not in state_dict or state_dict["type"] != "ParticleDataset":
            raise TypeError("Input is not an 'ParticleDataset' instance.")

        if "version" not in state_dict:
            raise RuntimeError("ParticleDataset instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            self.initialize(
                image_file_paths=state_dict["image_file_paths"],
                part_idx=state_dict["part_idx"],
                part_random_subset=state_dict["part_random_subset"],
                part_rotation=state_dict["part_rotation"],
                part_translation=state_dict["part_translation"],
                part_defocus=state_dict["part_defocus"],
                part_og_idx=state_dict["part_og_idx"],
                part_stack_idx=state_dict["part_stack_idx"],
                part_image_file_path_idx=state_dict["part_image_file_path_idx"],
                part_norm_correction=state_dict["part_norm_correction"],
                part_group_id=state_dict["part_group_id"],
                optics_group_stats=state_dict["optics_group_stats"],
                max_res=state_dict["max_res"]
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
