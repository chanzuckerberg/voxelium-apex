#!/usr/bin/env python3

"""
Utility module for training
"""

from glob import glob
import os
import shutil
import sys
from io import BytesIO
from typing import List, TypeVar, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from positron.base.star_file import load_star
from positron.relion import find_project_root
from positron.sha3d.cache import Cache

import matplotlib.pyplot as plt

Tensor = TypeVar('torch.tensor')

from positron.base import dt_symmetrize, integer_shift_2d, dft


def cosine_ascend(begin_ascend, end_ascend, x):
    if x < begin_ascend:
        return 0.
    if x > end_ascend:
        return 1.
    a = begin_ascend
    b = end_ascend - begin_ascend
    return .5 + np.cos(np.pi * (x - a) / b + np.pi) / 2.


def cosine_descend(begin_descend, end_descend, x):
    if x < begin_descend:
        return 1.
    if x > end_descend:
        return 0.
    a = begin_descend
    b = end_descend - begin_descend
    return .5 + np.cos(np.pi * (x - a) / b) / 2.


def cosine_interpolate(begin_ascend, end_ascend, x, begin_value=0, end_value=1):
    if begin_value < end_value:
        return begin_value + (end_value - begin_value) * cosine_ascend(begin_ascend, end_ascend, x)
    elif begin_value > end_value:
        return end_value + (begin_value - end_value) * cosine_descend(begin_ascend, end_ascend, x)
    else:
        return begin_value


def get_kld_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def reparameterize(mu, logvar):
    logvar = torch.clip(0.5 * logvar, max=5)
    std = torch.exp(logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def setup_device(args, verbose=False):
    device = None
    gpu_ids = []
    if args.gpu is not None:
        queried_gpu_ids = args.gpu.split(",")
        for i in range(len(queried_gpu_ids)):
            gpu_id = int(queried_gpu_ids[i].strip())
            try:
                gpu_name = torch.cuda.get_device_name(gpu_id)
            except AssertionError:
                if verbose:
                    print(f'WARNING: GPU with the device id "{gpu_id}" not found.', file=sys.stderr)
                continue
            if verbose:
                print(f'Found device "{gpu_name}"')
            gpu_ids.append(gpu_id)

        if len(gpu_ids) > 0:
            device = "cuda:" + str(gpu_ids[0])
            if verbose:
                print("Running on GPU with device id(s)", *gpu_ids)
        else:
            if verbose:
                print(f'WARNING: no GPUs were found with the specified ids.', file=sys.stderr)

    if len(gpu_ids) == 0:
        gpu_ids = None
        if verbose:
            print("Running on CPU")
        device = torch.device("cpu")

    return device, gpu_ids


np_dtype_dict = {"float32": np.float32, "float16": np.float16, "float64": np.float64}


def get_np_dtype(dtype_str: str) -> np.dtype:
    return np_dtype_dict[dtype_str]


def find_star_file_in_path(path: str, type: str = "optimiser") -> str:
    if os.path.isfile(os.path.join(path, f"run_{type}.star")):
        return os.path.join(path, f"run_{type}.star")
    files = glob(os.path.join(path, f"*{type}.star"))
    if len(files) > 0:
        files = list.sort(files)
        return files[-1]

    raise FileNotFoundError(f"Could not find '{type}' star-file in path: {path}")


def dump_particles_to_dir(input_path, output_path):
    """
    Load data from path
    :param path: relion job directory or data file
    """
    if os.path.isfile(input_path):
        data_star_path = input_path
        root_search_path = os.path.dirname(os.path.abspath(input_path))
    else:
        data_star_path = os.path.abspath(find_star_file_in_path(input_path, "data"))
        root_search_path = os.path.abspath(input_path)

    data_star_path = os.path.abspath(data_star_path)
    data = load_star(data_star_path)

    if 'optics' not in data:
        raise RuntimeError("Optics groups table not found in data star file")
    if 'particles' not in data:
        raise RuntimeError("Particles table not found in data star file")

    particles = data['particles']
    nr_particles = len(particles['rlnImageName'])
    image_file_paths = set()

    for i in range(nr_particles):
        img_name = particles['rlnImageName'][i]
        img_tokens = img_name.split("@")
        if len(img_tokens) == 2:
            img_path = img_tokens[1]
        elif len(img_tokens) == 1:
            img_path = img_tokens[1]
        else:
            raise RuntimeError(f"Invalid image file name (rlnImageName): {img_name}")
        image_file_paths.add(img_path)

    image_file_paths = list(image_file_paths)

    project_root = find_project_root(root_search_path, image_file_paths[0])

    # Convert image paths to absolute paths
    for i in range(len(image_file_paths)):
        image_file_paths[i] = os.path.abspath(os.path.join(project_root, image_file_paths[i]))

    new_project_path = os.path.abspath(output_path)
    destination_image_file_paths = [p.replace(project_root, new_project_path) for p in image_file_paths]

    [os.makedirs(os.path.dirname(p), exist_ok=True) for p in destination_image_file_paths]

    for src, dst in zip(image_file_paths, destination_image_file_paths):
        shutil.copy(src, dst)

    new_star_path = data_star_path.replace(os.path.dirname(data_star_path), new_project_path)
    shutil.copy(data_star_path, new_star_path)


def plot_fscs(output_file, **fscs):
    fig, main_ax = plt.subplots()
    for plot_name in fscs:
        fsc_file = load_star(fscs[plot_name])
        fsc_values = [float(x) for x in fsc_file["fsc"]["rlnFourierShellCorrelation"]]
        main_ax.plot(fsc_values, label=plot_name)
    main_ax.legend()
    main_ax.set_xlabel("1/Angstroms (1/Å)")
    main_ax.set_ylabel("FSC")
    plt.plot()
    plt.savefig(output_file)


def smoothen_spectra(spec, kernel=5):
    kernel += 1 - kernel % 2  # Make odd
    p = kernel // 2
    spec_pad = torch.zeros([spec.shape[0], spec.shape[1] + p * 2]).to(spec.device)
    spec_pad[:, p:-p] = spec
    spec_pad[:, :p] = spec[:, 0][:, None]
    spec_pad[:, -p:] = spec[:, -1][:, None]
    filter = torch.full([1, 1, kernel], 1. / kernel).to(spec.device)
    return F.conv1d(spec_pad.unsqueeze(1), filter).squeeze(1)


def smoothen_spectra(spec, kernel=5):
    no_batch = spec.ndim == 1
    if no_batch:
        spec = spec.unsqueeze(0)
    kernel += 1 - kernel % 2  # Make odd
    p = kernel // 2
    spec_pad = torch.zeros([spec.shape[0], spec.shape[1] + p * 2]).to(spec.device)
    spec_pad[:, p:-p] = spec
    spec_pad[:, :p] = spec[:, 0][:, None]
    spec_pad[:, -p:] = spec[:, -1][:, None]
    spec_pad = spec_pad.unsqueeze(1)
    x = torch.linspace(-1, 1, kernel).to(spec.device)
    filter = torch.exp(-3 * x.square())
    filter = filter[None, None] / torch.sum(filter)
    w = F.conv1d(torch.ones_like(spec_pad), filter).squeeze(1)
    spec_pad_filter = F.conv1d(spec_pad, filter).squeeze(1)
    if no_batch:
        spec_pad_filter = spec_pad_filter.squeeze(0)
    return spec_pad_filter


def preprocess_batch_hv(sample, dac):
    rec = dac.reconstruction_container
    hvc = dac.hidden_variable_container

    device = rec.get_device()

    image_size = sample['image'].shape[-1]

    part_idx = sample['idx'].to(device)
    hv = hvc.get_by_index(part_idx)  # Get all hidden variables
    hv["ctfs_"] = dt_symmetrize(hv["ctfs"], dim=2)[:, :, image_size // 2:]
    hv["amp_ctf_"] = dt_symmetrize(hv["amp_ctf"], dim=2)[:, :, image_size // 2:]

    hv["shifts_int"] = torch.round(hv["shifts"]).long().detach()
    hv["shifts_resid"] = hv["shifts"] - hv["shifts_int"]

    return hv


def preprocess_batch_data(sample, dac):
    rec = dac.reconstruction_container
    hvc = dac.hidden_variable_container

    device = rec.get_device()
    part_idx = sample['idx'].to(device)

    hv = preprocess_batch_hv(sample, dac)

    circular_mask_radius, circular_mask_thickness = rec.get_circular_mask_params()
    y = sample['image'].to(device)
    image_size = y.shape[-1]
    y = Cache.apply_square_mask(y, thickness=circular_mask_thickness)

    y_ = integer_shift_2d(y, hv["shifts_int"])
    y_ = Cache.apply_circular_mask(y_, thickness=circular_mask_thickness, radius=circular_mask_radius)
    y_ft = dft(y_, dim=2)
    # TODO Only use one hermitian half
    hvc.accumulate_data_stats(y_ft.detach(), hv["ctfs"].detach(), part_idx)
    y_ft = dt_symmetrize(y_ft, dim=2)[:, :, image_size // 2:]
    y_ft /= hv["amp_ctf_"] + 1e-6
    y_ft = torch.view_as_real(y_ft).detach()

    return y_ft, hv


def load_module(path: str) -> Dict:
    if path[-2:] == "gz":
        # Open the gzip file and decompress it
        import gzip
        with gzip.open(path, 'rb') as f:
            # Convert the compressed data back into a BytesIO buffer for torch.load
            buffer = BytesIO(f.read())

            # Deserialize the buffer's content using torch.load
            return torch.load(buffer, map_location="cpu")
    else:
        return torch.load(path, map_location="cpu")


def load_modules_from_logdir(logdir: str, module_names: List[str]) -> List:
    modules = []
    for module_name in module_names:
        paths = [
            os.path.join(logdir, module_name + ".pt"),
            os.path.join(logdir, module_name + ".backup.pt"),
            os.path.join(logdir, module_name + ".pt.gz"),
            os.path.join(logdir, module_name + ".backup.pt.gz")
        ]
        state_dict = None
        for path in paths:
            if os.path.isfile(path):
                try:
                    state_dict = load_module(path)
                except Exception as e:
                    pass
        if state_dict is None:
            msg = f"Unable to load module {module_name} for either of these paths: \n"
            for p in paths:
                msg += f" {p}\n"
            raise FileNotFoundError(msg)
        modules.append(state_dict)
    return modules


def save_modules_to_logdir(logdir: str, modules: Dict, gzip=False):
    for module_name in modules:
        if gzip:
            main_path = os.path.join(logdir, module_name + ".pt.gz")
            backup_path = os.path.join(logdir, module_name + ".backup.pt.gz")

            # Serialize with torch.save to an in-memory buffer
            buffer = BytesIO()
            torch.save(modules[module_name], buffer)
            buffer.seek(0)  # Important to move the buffer's pointer back to the beginning

            # Compress the buffer's content using gzip
            import gzip as _gzip
            with _gzip.open(backup_path, 'wb') as f:
                f.write(buffer.getvalue())
        else:
            main_path = os.path.join(logdir, module_name + ".pt")
            backup_path = os.path.join(logdir, module_name + ".backup.pt")
            torch.save(modules[module_name], backup_path)
        os.replace(backup_path, main_path)


def scaled_ratio(x, beta=2., a_min=0., a_max=1.):
    x = x.clip(0, 1)
    a_min = min(a_min, a_max)
    a_max = max(a_min, a_max)
    f = (a_max - a_min) * x * torch.exp(-beta * (x - 1) ** 2) + a_min
    return f.clip(a_min, a_max)


def generate_quadratic_sequence(a_min, a_max, n):
    """
    Generates a sequence of 'n' numbers between 'a_min' and 'a_max' with quadratic increments.

    Parameters:
    a_min (float): The starting value of the sequence.
    a_max (float): The end value of the sequence, or close to it.
    n (int): The number of values to generate in the sequence.

    The function calculates increments between consecutive numbers in a way that they increase
    quadratically. The first number in the sequence is 'a_min', and subsequent numbers are
    calculated by adding a quadratically increasing value to the previous number. The final
    number in the sequence may not be exactly 'a_max' due to the nature of the increments.

    Returns:
    list: A list of 'n' numbers starting from 'a_min' with quadratic increments.

    Raises:
    ValueError: If 'n' is less than 2, as at least two numbers are needed to form a sequence.
    """

    if n < 2:
        raise ValueError("n must be at least 2 to generate a sequence.")

    # Calculate the total range to be divided
    total_range = a_max - a_min

    # Calculate the sum of the first (n-1) squares, which is used to divide the range
    sum_of_squares = sum([i ** 2 for i in range(1, n)])

    # Generate the sequence
    sequence = [a_min]
    for i in range(1, n):
        # Calculate the next increment
        increment = (i ** 2 / sum_of_squares) * total_range
        # Add the increment to the last number in the sequence
        next_number = sequence[-1] + increment
        sequence.append(next_number)

    return sequence


def parse_bounds_str(bounds_str, default=None):
    if bounds_str is None or len(bounds_str) == 0:
        return [default]

    bounds = []
    tokens = bounds_str.split(",")
    for token in tokens:
        limits = token.split("-")
        if len(limits) != 2 and default not in bounds:
            bounds.append(default)
            continue
        if limits[0].isdigit() and limits[1].isdigit():
            lim0 = float(limits[0])
            lim1 = float(limits[1])
            if lim0 == lim1 and default not in bounds:
                bounds.append(default)
                continue
            highpass_ang = max(lim0, lim1)  # Largest values is high resolution
            lowpass_ang = min(lim0, lim1)  # Smallest value is low resolution
            bounds.append((highpass_ang, lowpass_ang))
        elif default not in bounds:
            bounds.append(default)

    return bounds


def zero_fill_number(number, max_num):
    # Calculate the number of digits in the maximum number
    max_digits = len(str(max_num))

    # Format the number to have the same number of digits with leading zeros
    formatted_number = f"{number:0{max_digits}d}"

    return formatted_number
