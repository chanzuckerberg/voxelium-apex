#!/usr/bin/env python

"""
Simple Pytorch tools
"""

import argparse
import glob
import importlib.util
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def standardize(np_input):
    mean = np.mean(np_input, axis=(1, 2, 3, 4))
    mean = np.resize(mean, (np_input.shape[0], 1, 1, 1, 1))
    std = np.std(np_input, axis=(1, 2, 3, 4)) + 1e-12
    std = np.resize(std, (np_input.shape[0], 1, 1, 1, 1))
    return mean, std


def torch_standardize(torch_input):
    mean = torch.mean(torch_input, dim=(1, 2, 3, 4))
    mean = torch.reshape(mean, (torch_input.shape[0], 1, 1, 1, 1))
    std = torch.std(torch_input, dim=(1, 2, 3, 4)) + 1e-12
    std = torch.reshape(std, (torch_input.shape[0], 1, 1, 1, 1))
    return mean, std


def normalize(np_input):
    norm = np.sqrt(np.sum(np.square(np_input), axis=(1, 2, 3, 4))) + 1e-12
    norm = np.resize(norm, (np_input.shape[0], 1, 1, 1, 1))
    return norm


def torch_normalize(torch_input):
    norm = torch.sqrt(torch.sum((torch_input) ** 2, dim=(1, 2, 3, 4))) + 1e-12
    norm = torch.reshape(norm, (torch_input.shape[0], 1, 1, 1, 1))
    return norm


def make_imshow_fig(data):
    if len(data.shape) == 3:
        data = data[data.shape[0] // 2]

    if torch.is_tensor(data):
        data = data.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(data)
    plt.axis("off")
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    matplotlib.use(backend)

    return fig


def make_scatter_fig(x, y, c=None):
    if torch.is_tensor(x):
        x = x.detach().data.cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    if c is None:
        c = np.arange(len(x))
    else:
        if torch.is_tensor(c):
            c = c.detach().data.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    alpha = min(10. / np.sqrt(len(x)), 1.)
    ax.scatter(x, y, edgecolors=None, marker='.', c=c, cmap="summer", alpha=alpha)

    mx = np.mean(x)
    sx = np.std(x) * 3
    my = np.mean(y)
    sy = np.std(y) * 3

    ax.set_xlim([mx - sx, mx + sx])
    ax.set_ylim([my - sy, my + sy])

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def make_gaussian_kde_fig(x, y):
    if torch.is_tensor(x):
        x = x.detach().data.cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 7))

    xy = np.vstack([x, y])
    from scipy.stats import gaussian_kde
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z, s=100)

    mx = np.mean(x)
    sx = np.std(x) * 3
    my = np.mean(y)
    sy = np.std(y) * 3

    ax.set_xlim([mx - sx, mx + sx])
    ax.set_ylim([my - sy, my + sy])

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def make_line_fig(x, y, y_log=False):
    if torch.is_tensor(x):
        x = x.detach().data.cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y)

    if y_log:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def make_series_line_fig(data, y_log=False):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 5))
    do_legend = False
    for d in data:
        y = d['y']
        if torch.is_tensor(y):
            y = y.detach().data.cpu().numpy()

        if 'x' in d:
            x = d['x']
            if torch.is_tensor(x):
                x = x.detach().data.cpu().numpy()
        else:
            x = np.arange(len(y))
        ax.plot(x, y,
                label=d['label'] if 'label' in d else None,
                color=d['color'] if 'color' in d else None,
                linestyle=d['linestyle'] if 'linestyle' in d else None
                )
        do_legend = 'label' in d or do_legend

    if do_legend:
        ax.legend()
    ax.grid()

    if y_log:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    matplotlib.use(backend)
    return fig


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def optimizer_set_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    # plt.barh(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.barh(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.yticks(range(0, len(ave_grads), 1), layers)
    plt.xlabel("average gradient")
    plt.ylabel("Layers")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)],
               ['max-gradient', 'mean-gradient'])
    plt.show()


def pca_dim_reduction_(x, n_components: int = 2, subsample: int = None):
    x_ = x
    if subsample is not None and subsample < x.size(0):
        idx = torch.randperm(x.size(0), device=x.device)
        x_ = x[idx[:subsample]]

    # Compute the covariance matrix
    cov_mat = torch.cov(x_.t())

    # Compute the eigenvalues and eigenvectors
    eigen_values, eigen_vectors = torch.linalg.eigh(cov_mat, UPLO='U')

    # Sort the eigenvalues and eigenvectors
    sorted_indices = torch.argsort(eigen_values, descending=True)
    sorted_eigenvectors = eigen_vectors[:, sorted_indices]

    # Select the top k eigenvectors (n_components)
    W = sorted_eigenvectors[:, :n_components]

    # Transform the original dataset
    transformed_data = torch.matmul(x, W)

    return transformed_data


def pca_dim_reduction(x, n_components=2, subsample=None, raise_oom=False):
    try:
        transformed_data = pca_dim_reduction_(x, n_components=n_components, subsample=subsample)
    except RuntimeError as e:
        if 'out of memory' in str(e) and not raise_oom:
            transformed_data = pca_dim_reduction_(x.to("cpu"), n_components=n_components, subsample=subsample)
            transformed_data = transformed_data.to(x.device)

    return transformed_data
