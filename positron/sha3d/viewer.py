#!/usr/bin/env python

"""
Module for visualizing 3D spectral heterogeneity analysis (SHA) training results or summaries.
"""

import sys

import numpy as np
import argparse

import scipy.ndimage

import threading

import matplotlib.pylab as plt
from matplotlib.widgets import Button

import torch

import multiprocessing as mp

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from positron.base import get_spectral_indices, spectra_to_grid, dft, idft

from positron.base.grid import load_mrc, save_mrc
from positron.base.torch_utils import pca_dim_reduction
from positron.sha3d.summary import Summary
from positron.sha3d.utils import setup_device

from matplotlib import backend_bases
backend_bases.NavigationToolbar2.toolitems = (
    ('Home', 'Reset original view', 'home', 'home'),
    ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
    ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
)


def main(args):
    device, _ = setup_device(args)

    print("Loading modules...")

    summary = Summary.load_from_path(args.path, device=device)
    latent = summary.metadata['z']
    structure_factors = summary.metadata['s']
    structure_factors_mean = structure_factors.mean(0).cpu().detach().numpy().astype(np.float32)

    if args.max_points is not None:
        selected_idxs = np.arange(latent.shape[0])
        np.random.shuffle(selected_idxs)
        latent = latent[selected_idxs[:min(args.max_points, latent.shape[0])]]
        structure_factors = structure_factors[selected_idxs[:min(args.max_points, latent.shape[0])]]

    embed = latent
    if embed.size(1) > 2:
        print("Creating 2D representation...")

        embed = (embed - embed.mean(dim=0, keepdim=True)) / (embed.std(dim=0, keepdim=True) + 1e-12)

        # embed = apply_tsne(embed.to(device), y_init=embed[:, :2], verbose=True)

        embed = pca_dim_reduction(embed)
        # embed = embed[:, :2]

        # umap_model = UMAP()
        # embed = umap_model.fit(x=embed)

        embed = embed.cpu().detach().numpy().astype(np.float32)

        # import umap
        # embed = umap.UMAP().fit_transform(embed.astype(np.float32))

        # from sklearn.manifold import TSNE
        # embed = TSNE(n_components=2, perplexity=200).fit_transform(embed.astype(np.float32))

        # import gpumap
        # embed = gpumap.GPUMAP().fit_transform(embed).astype(np.float32)

        # from sklearn.decomposition import PCA
        # embed = PCA(n_components=2).fit_transform(embed.astype(np.float32))
    else:
        embed = latent.cpu().detach().numpy().astype(np.float32)

    print("Visualizing representation...")

    if args.nogui:
        print("No GUI... Exiting!")
        exit(0)

    structure_factors = structure_factors.cpu().detach().numpy().astype(np.float32)

    torch.no_grad()

    from positron.sha3d.renderer import volumeRendererProcessLoop

    # N = 3000
    #
    # x_min = np.min(embed[:, 0])
    # x_max = np.max(embed[:, 0])
    # y_min = np.min(embed[:, 1])
    # y_max = np.max(embed[:, 1])
    #
    # x = (embed[:, 0] - x_min) / (x_max - x_min)
    # y = (embed[:, 1] - y_min) / (y_max - y_min)
    #
    # x = margin + x * (N - 2. * margin)
    # y = margin + y * (N - 2. * margin)

    # heat_map = np.zeros((N, N))
    # heat_map[y.astype(int), x.astype(int)] += 1
    # heat_map_smooth = scipy.ndimage.gaussian_filter(heat_map, 3)

    nn_time = 0
    ft_time = 0

    # SPECTRAL MODULATION -------------------------------------------------------------------------------

    ft_shape = list(summary.basis.shape[1:])
    ft_shape[-1] = ft_shape[-1] // 2 + 1
    spectral_indices = get_spectral_indices(ft_shape, device=device)

    voxel_size = 1
    if "voxel_size" in summary.metadata:
        voxel_size = max(summary.metadata['voxel_size'], 0.01)

    ref_basis_df = torch.zeros([summary.basis.shape[0]] + ft_shape, device="cpu", dtype=torch.complex64)
    for i in range(summary.basis.shape[0]):
        ref_basis_df[i] = dft(summary.basis[i].to("cpu"), real_in=True)

    # LATENT VISUALIZATION -------------------------------------------------------------------------------

    fig_hm, ax_hm = plt.subplots(figsize=(7, 7))  # Heat map

    ax_hm.title.set_text(args.path)

    ax_hm.axis('off')
    plt.tight_layout()

    # MAKE HEAT MAP -------------------------------------------------------------------------------------

    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde

    cm_voltage = [
        (1.000, 1.000, 1.000), (0.768, 0.881, 0.943), (0.522, 0.761, 0.959), (0.435, 0.607, 0.998),
        (0.488, 0.421, 0.953), (0.515, 0.211, 0.817), (0.459, 0.054, 0.570), (0.309, 0.082, 0.301)
    ]
    cm_tropical = [
        (1.000, 1.000, 1.000), (0.267, 0.987, 0.988), (0.154, 0.934, 0.722), (0.429, 0.843, 0.431),
        (0.647, 0.719, 0.203), (0.772, 0.580, 0.031), (0.837, 0.429, 0.067), (0.850, 0.273, 0.195),
        (0.808, 0.111, 0.354), (0.699, 0.022, 0.528), (0.565, 0.054, 0.646)
    ]
    n_bins = 100  # Use 100 bins for smooth transitions

    # Create the colormap
    cmap_name = 'custom_white_purple_blue'
    cm = LinearSegmentedColormap.from_list(cmap_name, np.array(cm_tropical), N=n_bins)

    marker_size = 0.05

    x = embed[:, 0]
    y = embed[:, 1]

    x = (x - x.mean()) / (x.std() + 1e-3)
    y = (y - y.mean()) / (y.std() + 1e-3)

    coord = np.stack([x, y], 1)

    coord_ = np.unique(coord, axis=0)

    x = coord_[:min(10000, len(x) - 1), 0]
    y = coord_[:min(10000, len(y) - 1), 1]
    xy = np.vstack([x, y])

    k = gaussian_kde(xy)
    xi, yi = np.mgrid[-3:3:100j, -3:3:100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # ax_hm.pcolormesh(xi, yi, zi.reshape(xi.shape))
    ax_hm.contourf(xi, yi, zi.reshape(xi.shape), levels=100, alpha=1, cmap=cm)

    z = k(xy)
    z_ = (1 - z / z.max()) * 0.1
    ax_hm.scatter(x, y, c="black", s=10, edgecolors='none', marker='o', alpha=z_)

    ax_hm.set_xlim([-3, 3])
    ax_hm.set_ylim([-3, 3])

    # VOLUME RENDERER QUEUES -----------------------------------------------------------------------

    volume_render_input_queue = mp.Queue()  # Input to the volume renderer
    volume_render_output_queue = mp.Queue()  # Output from the volume renderer

    # BUTTONS & TEXT BOXES -------------------------------------------------------------------------

    def make_text_box(ax, text):
        ax.set_navigate(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.text(
            0.5, 0.5, text,
            verticalalignment='center',
            horizontalalignment='center',
            transform=ax.transAxes
        )

    clear_button_axes = plt.axes([0.01, 0.01, 0.1, 0.05])
    clear_button = Button(clear_button_axes, 'clear\n(Esc)')

    subset_button_axes = plt.axes([0.12, 0.01, 0.1, 0.05])
    subset_button = Button(subset_button_axes, 'subset')

    save_volumes_button_axes = plt.axes([0.23, 0.01, 0.1, 0.05])
    save_volumes_button = Button(save_volumes_button_axes, 'save\nvolumes')

    save_images_button_axes = plt.axes([0.34, 0.01, 0.1, 0.05])
    save_images_button = Button(save_images_button_axes, 'save\nGIF')

    # B-Factor setting ---------

    bfactor = 0
    bfactor_step = args.bfac_step

    bfactor_up_button_axes = plt.axes([0.725, 0.037, 0.02, 0.023])
    bfactor_up_button = Button(bfactor_up_button_axes, '↑')

    bfactor_down_button_axes = plt.axes([0.725, 0.01, 0.02, 0.023])
    bfactor_down_button = Button(bfactor_down_button_axes, '↓')

    bfactor_text_axes = plt.axes([0.75, 0.01, 0.1, 0.05])
    bfactor_text = make_text_box(bfactor_text_axes, f"B-factor\n{bfactor}")

    # Iso-Value setting ---------

    iso_value_up_button_axes = plt.axes([0.865, 0.037, 0.02, 0.023])
    iso_value_up_button = Button(iso_value_up_button_axes, '↑')

    iso_value_down_button_axes = plt.axes([0.865, 0.01, 0.02, 0.023])
    iso_value_down_button = Button(iso_value_down_button_axes, '↓')

    iso_value_text_axes = plt.axes([0.89, 0.01, 0.1, 0.05])
    iso_value_text = make_text_box(iso_value_text_axes, "Iso value\n-")

    # EVENT HANDLERS ------------------------------------------------------------------------------

    def get_volume(sf):
        global nn_time, ft_time
        vol = summary(torch.from_numpy(sf).to(device))
        return vol.detach().cpu().numpy()

    def set_default_volume():
        volume_render_input_queue.put([
            get_volume(structure_factors_mean)
        ])

    circles = []
    circles_coord = []
    selected_ids = []
    volumes = []

    def clear_selection(_):
        if len(volumes) == 0:
            return

        for i in range(len(circles)):
            circles[i].remove()

        circles.clear()
        circles_coord.clear()
        selected_ids.clear()
        volumes.clear()

        set_default_volume()
        fig_hm.canvas.draw()

    selector_obj = None
    selector_line = {'color': '#c596fb', 'linewidth': 4, 'alpha': 0.8}
    subset = 1

    clear_button.on_clicked(clear_selection)

    def clear_subset_selection(_=None):
        global selector_obj
        if selector_obj is not None:
            selector_obj.clear()
            selector_obj = None

    def save_dataset_indices(vertices):
        global subset, coord, selector_obj
        path = Path(vertices)
        mask = path.contains_points(coord)
        indices = np.arange(0, coord.shape[0])[mask]
        path = f"subset_{subset}.csv"
        print(f"Saving indices of {len(indices)} selected particles to {path}")
        np.savetxt(path, indices, delimiter=',', fmt='%d')
        subset += 1
        clear_subset_selection()

    def subset_selection(_=None):
        global selector_obj
        if selector_obj is None:
            selector_obj = PolygonSelector(
                ax_hm, onselect=save_dataset_indices, props=selector_line, useblit=True)
        else:
            clear_subset_selection()

    subset_button.on_clicked(subset_selection)

    def save_selected_volumes(_=None):
        print('Saving selected structures to MRC-files:')
        for i, v in enumerate(volumes):
            path = f"particle_id_{str(selected_ids[i])}.mrc"
            print(f" {path}")
            save_mrc(v, path)

    save_volumes_button.on_clicked(save_selected_volumes)

    def save_volume_images(_=None):
        volume_render_input_queue.put("save_images")

    save_images_button.on_clicked(save_volume_images)

    @torch.no_grad()
    def update_bfactor(bfac):
        global volumes, summary, spectral_indices, ref_basis_df, device

        idx = torch.linspace(0, 2, ft_shape[-1] * 2, device=device)
        res2 = torch.square(idx / voxel_size)
        profile = torch.exp(-bfac / 4. * res2)
        grid = spectra_to_grid(profile, spectral_indices)

        for i in range(summary.basis.shape[0]):
            df = ref_basis_df[i].to(device) * grid
            summary.basis[i] = idft(df, real_in=True)

        if len(volumes) > 0:
            volumes = []
            for i in range(len(selected_ids)):
                idx = selected_ids[i]
                vol = get_volume(structure_factors[idx])
                volumes.append(vol)
            volume_render_input_queue.put(volumes)
        else:
            set_default_volume()

        bfactor_text.set_text(f"B-factor\n{round(bfactor)}")
        fig_hm.canvas.blit(iso_value_text_axes.bbox)
        fig_hm.canvas.draw()

    def raise_bfactor(_=None):
        global bfactor
        bfactor += bfactor_step
        update_bfactor(bfactor)

    bfactor_up_button.on_clicked(raise_bfactor)

    def lower_bfactor(_=None):
        global bfactor
        bfactor -= bfactor_step
        update_bfactor(bfactor)

    bfactor_down_button.on_clicked(lower_bfactor)

    def raise_iso_value(_=None):
        volume_render_input_queue.put("up")

    iso_value_up_button.on_clicked(raise_iso_value)

    def lower_iso_value(_=None):
        volume_render_input_queue.put("down")

    iso_value_down_button.on_clicked(lower_iso_value)

    #  --------------------------------------------------------------------------------

    set_default_volume()  # Warm up

    def onClickHm(event):
        if event.xdata is None or event.ydata is None:
            return

        if selector_obj is not None:
            return

        xy = np.array([event.xdata, event.ydata])

        state_change = False

        if event.button == 1:  # Add volume
            c = np.sum((coord - xy) ** 2, axis=1)
            idx = np.argmin(c)
            dis2 = c[idx]
            xy = [coord[idx, 0], coord[idx, 1]]

            if dis2 < 0.1 and idx not in selected_ids:
                circle = plt.Circle(xy, marker_size, color='black', alpha=1, zorder=2)
                vol = get_volume(structure_factors[idx])
                circles.append(circle)
                circles_coord.append(xy)
                selected_ids.append(idx)
                volumes.append(vol)
                print("Selected point index", idx)
                ax_hm.add_patch(circle)
                state_change = True

        elif event.button == 3:  # Remove point
            if len(circles) > 0:
                c = np.sum((np.array(circles_coord) - xy) ** 2, axis=1)
                selected_idx = np.argmin(c)
                dis2 = c[selected_idx]

                if dis2 < 0.1:
                    circles[selected_idx].remove()
                    del (circles[selected_idx])
                    del (circles_coord[selected_idx])
                    del (selected_ids[selected_idx])
                    del (volumes[selected_idx])
                    state_change = True

        if state_change:
            if len(volumes) == 0:
                set_default_volume()
            else:
                volume_render_input_queue.put(volumes)
            fig_hm.canvas.draw()

    def handle_key_press(key):
        global selector_obj
        if key == 'escape':
            clear_selection(None)
            clear_subset_selection(None)
        elif key == "left":
            lower_bfactor()
        elif key == "right":
            raise_bfactor()

    def onKeyPressEvent(event):
        key = event.key
        handle_key_press(key)
        volume_render_input_queue.put(key)

    click_connect = fig_hm.canvas.mpl_connect('button_press_event', onClickHm)
    key_press = fig_hm.canvas.mpl_connect('key_press_event', onKeyPressEvent)

    # VOLUME RENDERER PROCESS ---------------------------------------------------------------------------

    volume_render_process = mp.Process(
        target=volumeRendererProcessLoop,
        args=(
            volume_render_input_queue,
            volume_render_output_queue
        )
    )
    volume_render_process.start()

    def volume_renderer_event():
        task = volume_render_output_queue.get()

        if task is not None:
            if isinstance(task, str):
                if task == "exit":
                    return

                if len(task) > 10 and task[:10] == "iso_value_":
                    iso_value = round(float(task[10:]), 3)
                    iso_value_text.set_text(f"Iso value\n{iso_value}")
                    fig_hm.canvas.blit(iso_value_text_axes.bbox)
                    fig_hm.canvas.draw()

            handle_key_press(task.lower())

        threading.Timer(0, volume_renderer_event).start()
    threading.Timer(0.1, volume_renderer_event).start()

    # Start process loop --------------------------------------------------------------------------------

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Exiting!")

    volume_render_input_queue.put("exit")
    volume_render_output_queue.put("exit")
    volume_render_process.join()
    volume_render_process.terminate()


def append_args(parser):
    parser.add_argument('path', help='input summary file or logdir', type=str)
    parser.add_argument('--gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--dont_cache_embed', action="store_true")
    parser.add_argument('--ignore_cached_embed', action="store_true")
    parser.add_argument('--nogui', action="store_true")
    parser.add_argument('--scale_invar', action="store_true")
    parser.add_argument('--max_points', type=int, default=None, help='Number of points to consider')
    parser.add_argument('--bfac_step', type=float, default=20, help='B-factor steps')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Used to make a visualize SHA3D training summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)
