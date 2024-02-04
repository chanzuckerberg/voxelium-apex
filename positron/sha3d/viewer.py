#!/usr/bin/env python

"""
Module for visualizing 3D spectral heterogeneity analysis (SHA) training results or summaries.
"""

import sys

import numpy as np
import argparse

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
from positron.sha3d.train_utils import setup_device
from positron.sha3d.renderer import volumeRendererProcessLoop

from matplotlib import backend_bases
backend_bases.NavigationToolbar2.toolitems = (
    ('Home', 'Reset original view', 'home', 'home'),
    ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
    ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
)


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


class Viewer:
    @torch.no_grad()
    def __init__(self, summary, embed, structure_factors, device, title=None, bfac_step=20):
        self.summary = summary
        self.embed = embed
        self.structure_factors = structure_factors
        self.structure_factors_mean = structure_factors.mean(0)
        self.device = device
        
        self.subset_index = 1
        self.subset_selection_ongoing = False
        self.selector_obj = None

        # SPECTRAL MODULATION -------------------------------------------------------------------------------

        self.ft_shape = list(summary.basis.shape[1:])
        self.ft_shape[-1] = self.ft_shape[-1] // 2 + 1
        self.spectral_indices = get_spectral_indices(self.ft_shape, device=self.device)

        self.voxel_size = 1
        if "voxel_size" in summary.metadata:
            self.voxel_size = max(summary.metadata['voxel_size'], 0.01)

        self.ref_basis_df = torch.zeros([summary.basis.shape[0]] + self.ft_shape, device="cpu", dtype=torch.complex64)
        for i in range(summary.basis.shape[0]):
            self.ref_basis_df[i] = dft(summary.basis[i].to("cpu"), real_in=True)

        # LATENT VISUALIZATION -------------------------------------------------------------------------------

        self.fig_hm, self.ax_hm = plt.subplots(figsize=(7, 7))  # Heat map

        if title is not None:
            self.ax_hm.title.set_text(title)

        self.ax_hm.axis('off')
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
        self.marker_size = 0.05

        # Create the colormap
        cmap_name = 'custom_white_purple_blue'
        cm = LinearSegmentedColormap.from_list(cmap_name, np.array(cm_tropical), N=n_bins)

        x = embed[:, 0].cpu().numpy().astype(np.float32)
        y = embed[:, 1].cpu().numpy().astype(np.float32)

        x = (x - x.mean()) / (x.std() + 1e-3)
        y = (y - y.mean()) / (y.std() + 1e-3)

        self.coord = np.stack([x, y], 1)

        coord = np.unique(self.coord, axis=0)
        mask = np.zeros(coord.shape[0], dtype=bool)
        mask[:min(2000, len(x) - 1)] = True
        np.random.shuffle(mask)

        x = coord[mask, 0]
        y = coord[mask, 1]
        xy = np.vstack([x, y])

        k = gaussian_kde(xy)
        xi, yi = np.mgrid[-3:3:100j, -3:3:100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        self.ax_hm.contourf(xi, yi, zi.reshape(xi.shape), levels=100, alpha=1, cmap=cm)

        z = k(xy)
        z_ = (1 - z / z.max()) * 0.1
        self.ax_hm.scatter(x, y, c="black", s=10, edgecolors='none', marker='o', alpha=z_)

        self.ax_hm.set_xlim([-3, 3])
        self.ax_hm.set_ylim([-3, 3])

        # VOLUME RENDERER QUEUES -----------------------------------------------------------------------

        self.volume_render_input_queue = mp.Queue()  # Input to the volume renderer
        self.volume_render_output_queue = mp.Queue()  # Output from the volume renderer

        # BUTTONS & TEXT BOXES -------------------------------------------------------------------------

        clear_button_axes = plt.axes([0.01, 0.01, 0.1, 0.05])
        clear_button = Button(clear_button_axes, 'clear\n(Esc)')

        subset_button_axes = plt.axes([0.12, 0.01, 0.1, 0.05])
        subset_button = Button(subset_button_axes, 'subset')

        save_volumes_button_axes = plt.axes([0.23, 0.01, 0.1, 0.05])
        save_volumes_button = Button(save_volumes_button_axes, 'save\nvolumes')

        save_images_button_axes = plt.axes([0.34, 0.01, 0.1, 0.05])
        save_images_button = Button(save_images_button_axes, 'save\nGIF')

        # B-Factor setting ---------

        self.bfactor = 0
        self.bfactor_step = bfac_step

        bfactor_up_button_axes = plt.axes([0.725, 0.037, 0.02, 0.023])
        bfactor_up_button = Button(bfactor_up_button_axes, '↑')

        bfactor_down_button_axes = plt.axes([0.725, 0.01, 0.02, 0.023])
        bfactor_down_button = Button(bfactor_down_button_axes, '↓')

        bfactor_text_axes = plt.axes([0.75, 0.01, 0.1, 0.05])
        self.bfactor_text = make_text_box(bfactor_text_axes, f"B-factor\n{self.bfactor}")

        # Iso-Value setting ---------

        iso_value_up_button_axes = plt.axes([0.865, 0.037, 0.02, 0.023])
        iso_value_up_button = Button(iso_value_up_button_axes, '↑')

        iso_value_down_button_axes = plt.axes([0.865, 0.01, 0.02, 0.023])
        iso_value_down_button = Button(iso_value_down_button_axes, '↓')

        self.iso_value_text_axes = plt.axes([0.89, 0.01, 0.1, 0.05])
        self.iso_value_text = make_text_box(self.iso_value_text_axes, "Iso value\n-")

        # EVENT HANDLERS ------------------------------------------------------------------------------

        self.circles = []
        self.circles_coord = []
        self.selected_ids = []
        self.volumes = []

        self.selector_line = {'color': '#c596fb', 'linewidth': 4, 'alpha': 0.8}

        clear_button.on_clicked(self.clear_selection)
        subset_button.on_clicked(self.subset_selection)
        save_volumes_button.on_clicked(self.save_selected_volumes)
        save_images_button.on_clicked(self.save_volume_images)
        bfactor_up_button.on_clicked(self.raise_bfactor)
        bfactor_down_button.on_clicked(self.lower_bfactor)
        iso_value_up_button.on_clicked(self.raise_iso_value)
        iso_value_down_button.on_clicked(self.lower_iso_value)

        #  --------------------------------------------------------------------------------

        self.set_default_volume()  # Warm up

        click_connect = self.fig_hm.canvas.mpl_connect('button_press_event', self.onClickHm)
        key_press = self.fig_hm.canvas.mpl_connect('key_press_event', self.onKeyPressEvent)

        # VOLUME RENDERER PROCESS ---------------------------------------------------------------------------

        volume_render_process = mp.Process(
            target=volumeRendererProcessLoop,
            args=(
                self.volume_render_input_queue,
                self.volume_render_output_queue
            )
        )
        volume_render_process.start()

        threading.Timer(0.1, self.volume_renderer_event).start()

        # Start process loop --------------------------------------------------------------------------------

        try:
            plt.show()
        except KeyboardInterrupt:
            print("Exiting!")

        self.volume_render_input_queue.put("exit")
        self.volume_render_output_queue.put("exit")
        volume_render_process.join()
        volume_render_process.terminate()

    def get_volume(self, sf):
        vol = self.summary(sf.to(self.device))
        return vol.detach().cpu().numpy()

    def set_default_volume(self):
        self.volume_render_input_queue.put([
            self.get_volume(self.structure_factors_mean)
        ])

    def clear_selection(self, _=None):
        if len(self.volumes) == 0:
            return

        for i in range(len(self.circles)):
            self.circles[i].remove()

        self.circles.clear()
        self.circles_coord.clear()
        self.selected_ids.clear()
        self.volumes.clear()

        self.set_default_volume()
        self.fig_hm.canvas.draw()

    def clear_subset_selection(self, _=None):
        if self.selector_obj is not None:
            self.selector_obj.clear()
            del self.selector_obj
            self.selector_obj = None
            self.subset_selection_ongoing = False

    def save_dataset_indices(self, vertices):
        path = Path(vertices)
        mask = path.contains_points(self.coord)
        indices = np.arange(0, self.coord.shape[0])[mask]
        path = f"subset_{self.subset_index}.csv"
        print(f"Saving indices of {len(indices)} selected particles to {path}")
        np.savetxt(path, indices, delimiter=',', fmt='%d')
        self.subset_index += 1
        self.clear_subset_selection()

    def subset_selection(self, _=None):
        if self.selector_obj is None and not self.subset_selection_ongoing:
            self.subset_selection_ongoing = True
            self.selector_obj = PolygonSelector(
                self.ax_hm,
                onselect=self.save_dataset_indices,
                props=self.selector_line,
                useblit=True
            )
            self.subset_selection_ongoing = False
        else:
            self.clear_subset_selection()

    def save_selected_volumes(self, _=None):
        print('Saving selected structures to MRC-files:')
        for i, v in enumerate(self.volumes):
            path = f"particle_id_{str(self.selected_ids[i])}.mrc"
            print(f" {path}")
            save_mrc(v, path)

    def save_volume_images(self, _=None):
        self.volume_render_input_queue.put("save_images")

    @torch.no_grad()
    def update_bfactor(self, bfac):
        idx = torch.linspace(0, 2, self.ft_shape[-1] * 2, device=self.device)
        res2 = torch.square(idx / self.voxel_size)
        profile = torch.exp(-bfac / 4. * res2)
        grid = spectra_to_grid(profile, self.spectral_indices)

        for i in range(self.summary.basis.shape[0]):
            df = self.ref_basis_df[i].to(self.device) * grid
            self.summary.basis[i] = idft(df, real_in=True)

        if len(self.volumes) > 0:
            volumes = []
            for i in range(len(self.selected_ids)):
                idx = self.selected_ids[i]
                vol = self.get_volume(self.structure_factors[idx])
                volumes.append(vol)
            self.volume_render_input_queue.put(volumes)
        else:
            self.set_default_volume()

        self.bfactor_text.set_text(f"B-factor\n{round(self.bfactor)}")
        self.fig_hm.canvas.blit(self.iso_value_text_axes.bbox)
        self.fig_hm.canvas.draw()

    def raise_bfactor(self, _=None):
        self.bfactor += self.bfactor_step
        self.update_bfactor(self.bfactor)

    def lower_bfactor(self, _=None):
        self.bfactor -= self.bfactor_step
        self.update_bfactor(self.bfactor)

    def raise_iso_value(self, _=None):
        self.volume_render_input_queue.put("up")

    def lower_iso_value(self, _=None):
        self.volume_render_input_queue.put("down")

    def onClickHm(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.selector_obj is not None:
            return

        xy = np.array([event.xdata, event.ydata])

        state_change = False

        if event.inaxes == self.ax_hm:
            if event.button == 1:  # Add volume
                d = np.sum((self.coord - xy) ** 2, axis=1)
                idx = np.argmin(d)
                dis2 = d[idx]
                xy = [self.coord[idx, 0], self.coord[idx, 1]]

                if dis2 < 0.1 and idx not in self.selected_ids:
                    circle = plt.Circle(xy, self.marker_size, color='black', alpha=0.5, zorder=2)
                    vol = self.get_volume(self.structure_factors[idx])
                    self.circles.append(circle)
                    self.circles_coord.append(xy)
                    self.selected_ids.append(idx)
                    self.volumes.append(vol)
                    print("Selected point index", idx)
                    self.ax_hm.add_patch(circle)
                    state_change = True

            elif event.button == 3:  # Remove point
                if len(self.circles) > 0:
                    c = np.sum((np.array(self.circles_coord) - xy) ** 2, axis=1)
                    selected_idx = np.argmin(c)
                    dis2 = c[selected_idx]

                    if dis2 < 0.1:
                        self.circles[selected_idx].remove()
                        del (self.circles[selected_idx])
                        del (self.circles_coord[selected_idx])
                        del (self.selected_ids[selected_idx])
                        del (self.volumes[selected_idx])
                        state_change = True

        if state_change:
            if len(self.volumes) == 0:
                self.set_default_volume()
            else:
                self.volume_render_input_queue.put(self.volumes)
            self.fig_hm.canvas.draw()

    def handle_key_press(self, key):
        if key == 'escape':
            self.clear_selection()
            self.clear_subset_selection()
        elif key == "left":
            self.lower_bfactor()
        elif key == "right":
            self.raise_bfactor()

    def onKeyPressEvent(self, event):
        key = event.key
        self.handle_key_press(key)
        self.volume_render_input_queue.put(key)

    def volume_renderer_event(self):
        task = self.volume_render_output_queue.get()

        if task is not None:
            if isinstance(task, str):
                if task == "exit":
                    return

                if len(task) > 10 and task[:10] == "iso_value_":
                    iso_value = round(float(task[10:]), 3)
                    self.iso_value_text.set_text(f"Iso value\n{iso_value}")
                    self.fig_hm.canvas.blit(self.iso_value_text_axes.bbox)
                    self.fig_hm.canvas.draw()

            self.handle_key_press(task.lower())

        threading.Timer(0, self.volume_renderer_event).start()


def main(args):

    torch.no_grad()
    device, _ = setup_device(args)

    print("Loading modules...")

    summary = Summary.load_from_path(args.path, device=device)
    embed = summary.metadata['z']
    structure_factors = summary.metadata['s']

    if embed.size(1) > 2:
        print("Creating 2D representation...")

        embed = (embed - embed.mean(dim=0, keepdim=True)) / (embed.std(dim=0, keepdim=True) + 1e-12)

        # embed = apply_tsne(embed.to(device), y_init=embed[:, :2], verbose=True)

        embed = pca_dim_reduction(embed)
        # embed = embed[:, :2]

        # umap_model = UMAP()
        # embed = umap_model.fit(x=embed)

        # embed = embed.cpu().detach().numpy().astype(np.float32)

        # import umap
        # embed = umap.UMAP().fit_transform(embed.astype(np.float32))

        # from sklearn.manifold import TSNE
        # embed = TSNE(n_components=2, perplexity=200).fit_transform(embed.astype(np.float32))

        # import gpumap
        # embed = gpumap.GPUMAP().fit_transform(embed).astype(np.float32)

        # from sklearn.decomposition import PCA
        # embed = PCA(n_components=2).fit_transform(embed.astype(np.float32))

        # embed = torch.from_numpy(embed)

    print("Visualizing representation...")

    viewer = Viewer(summary, embed, structure_factors, device, args.path, args.bfac_step)


def append_args(parser):
    parser.add_argument('path', help='input summary file or logdir', type=str)
    parser.add_argument('--gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--dont_cache_embed', action="store_true")
    parser.add_argument('--ignore_cached_embed', action="store_true")
    parser.add_argument('--scale_invar', action="store_true")
    parser.add_argument('--bfac_step', type=float, default=20, help='B-factor steps')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Used to make a visualize SHA3D training summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)
