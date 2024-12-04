#!/usr/bin/env python

"""
Module for visualizing 3D spectral heterogeneity analysis (SHA) training results or summaries.
"""

import sys

import numpy as np
import argparse

import threading

import matplotlib.pylab as plt
from matplotlib.widgets import Button, PolygonSelector
from matplotlib.path import Path

import torch

import multiprocessing as mp

from positron.base import get_spectral_indices, spectra_to_grid, dft, idft
from positron.base.grid import load_mrc, save_mrc, gaussian_blur
from positron.base.plot import get_default_cmap
from positron.base.torch_utils import pca_dim_reduction
from positron.sha3d.summary import Summary
from positron.sha3d.train_utils import setup_device
from positron.sha3d.renderer import volumeRendererProcessLoop

# from matplotlib import backend_bases
# backend_bases.NavigationToolbar2.toolitems = (
#     ('Home', 'Reset original view', 'home', 'home'),
#     ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
#     ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
# )


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


DEBUG = False
DEBUG_COUNTER = 0


def debug_print(msg):
    global DEBUG, DEBUG_COUNTER
    if DEBUG:
        print(f"Viewer {str(DEBUG_COUNTER).zfill(3)}: {msg}")
        DEBUG_COUNTER += 1


def debug_decorator(func):
    def wrapper(*args, **kwargs):
        debug_print(f"{func.__name__}()")
        return func(*args, **kwargs)
    return wrapper


class Viewer:
    @torch.no_grad()
    def __init__(self, summary, embed, structure_factors, device, title=None, bfac_step=20):
        debug_print("init() begin")
        self.summary = summary
        self.embed = embed
        self.structure_factors = structure_factors
        self.structure_factors_mean = structure_factors.mean(0)
        self.device = device
        
        self.subset_index = 1
        self.subset_selection_ongoing = False
        self.selector_obj = None

        # SPECTRAL MODULATION -------------------------------------------------------------------------------

        self.mask = summary.basis.sum(dim=0) != 0

        self.ft_shape = list(summary.basis.shape[1:])
        self.ft_shape[-1] = self.ft_shape[-1] // 2 + 1
        self.spectral_indices = get_spectral_indices(self.ft_shape, device=self.device, centered=False)

        self.voxel_size = 1
        if "voxel_size" in summary.metadata:
            self.voxel_size = max(summary.metadata['voxel_size'], 0.01)

        self.ref_basis_df = torch.zeros([summary.basis.shape[0]] + self.ft_shape, device="cpu", dtype=torch.complex64)
        for i in range(summary.basis.shape[0]):
            self.ref_basis_df[i] = dft(summary.basis[i].to("cpu"), real_in=True, center=False)

        # LATENT VISUALIZATION -------------------------------------------------------------------------------

        self.fig_hm, self.ax_hm = plt.subplots(figsize=(7, 7))  # Heat map

        if title is not None:
            self.ax_hm.title.set_text(title)

        self.ax_hm.axis('off')
        plt.tight_layout()

        print(f"Found {embed.size(0)} points.")

        z = embed.cpu().numpy().astype(np.float32)

        z_min = np.min(z, axis=0)
        z_max = np.max(z, axis=0)
        self.coord = (z - z_min) / (z_max - z_min)

        # VOLUME RENDERER QUEUES -----------------------------------------------------------------------

        self.volume_render_input_queue = mp.Queue()  # Input to the volume renderer
        self.volume_render_output_queue = mp.Queue()  # Output from the volume renderer

        # BUTTONS & TEXT BOXES -------------------------------------------------------------------------

        self.clear_button_axes = plt.axes([0.01, 0.01, 0.1, 0.05])
        self.clear_button = Button(self.clear_button_axes, 'clear\n(Esc)')

        self.subset_button_axes = plt.axes([0.12, 0.01, 0.1, 0.05])
        self.subset_button = Button(self.subset_button_axes, 'subset')

        self.save_volumes_button_axes = plt.axes([0.23, 0.01, 0.1, 0.05])
        self.save_volumes_button = Button(self.save_volumes_button_axes, 'save\nvolumes')

        self.save_images_button_axes = plt.axes([0.34, 0.01, 0.1, 0.05])
        self.save_images_button = Button(self.save_images_button_axes, 'save\nGIF')

        # Heat-map setting ---------

        self.hm_sigma = 4

        self.hm_up_button_axes = plt.axes([0.585, 0.037, 0.02, 0.023])
        self.hm_up_button = Button(self.hm_up_button_axes, '↑')

        self.hm_down_button_axes = plt.axes([0.585, 0.01, 0.02, 0.023])
        self.hm_down_button = Button(self.hm_down_button_axes, '↓')

        self.hm_text_axes = plt.axes([0.61, 0.01, 0.1, 0.05])
        self.hm_sigma_text = make_text_box(self.hm_text_axes, f"Smooth\n{self.hm_sigma}")

        self.hm_contrast = 0.2

        self.hmc_up_button_axes = plt.axes([0.445, 0.037, 0.02, 0.023])
        self.hmc_up_button = Button(self.hmc_up_button_axes, '↑')

        self.hmc_down_button_axes = plt.axes([0.445, 0.01, 0.02, 0.023])
        self.hmc_down_button = Button(self.hmc_down_button_axes, '↓')

        self.hmc_text_axes = plt.axes([0.47, 0.01, 0.1, 0.05])
        self.hmc_sigma_text = make_text_box(self.hmc_text_axes, f"Contrast\n{self.hm_contrast}")

        # B-Factor setting ---------

        self.bfactor = 0
        self.bfactor_step = bfac_step

        self.bfactor_up_button_axes = plt.axes([0.725, 0.037, 0.02, 0.023])
        self.bfactor_up_button = Button(self.bfactor_up_button_axes, '↑')

        self.bfactor_down_button_axes = plt.axes([0.725, 0.01, 0.02, 0.023])
        self.bfactor_down_button = Button(self.bfactor_down_button_axes, '↓')

        self.bfactor_text_axes = plt.axes([0.75, 0.01, 0.1, 0.05])
        self.bfactor_text = make_text_box(self.bfactor_text_axes, f"B-factor\n{self.bfactor}")

        # Iso-Value setting ---------

        self.iso_value_up_button_axes = plt.axes([0.865, 0.037, 0.02, 0.023])
        self.iso_value_up_button = Button(self.iso_value_up_button_axes, '↑')

        self.iso_value_down_button_axes = plt.axes([0.865, 0.01, 0.02, 0.023])
        self.iso_value_down_button = Button(self.iso_value_down_button_axes, '↓')

        self.iso_value_text_axes = plt.axes([0.89, 0.01, 0.1, 0.05])
        self.iso_value_text = make_text_box(self.iso_value_text_axes, "Iso value\n-")

        # EVENT HANDLERS ------------------------------------------------------------------------------

        self.circles = []
        self.circles_coord = []
        self.selected_ids = []
        self.volumes = []
        self.lines = None

        self.selector_line = {'color': '#c596fb', 'linewidth': 4, 'alpha': 0.8}

        self.clear_button.on_clicked(self.clear_selection)
        self.subset_button.on_clicked(self.subset_selection)
        self.save_volumes_button.on_clicked(self.save_selected_volumes)
        self.save_images_button.on_clicked(self.save_volume_images)
        self.hm_up_button.on_clicked(self.raise_hm_sigma)
        self.hm_down_button.on_clicked(self.lower_hm_sigma)
        self.hmc_up_button.on_clicked(self.raise_hm_contrast)
        self.hmc_down_button.on_clicked(self.lower_hm_contrast)
        self.bfactor_up_button.on_clicked(self.raise_bfactor)
        self.bfactor_down_button.on_clicked(self.lower_bfactor)
        self.iso_value_up_button.on_clicked(self.raise_iso_value)
        self.iso_value_down_button.on_clicked(self.lower_iso_value)

        # MAKE HEAT MAP -------------------------------------------------------------------------------------

        self.hm_cm = get_default_cmap()

        self.marker_size = 0.007
        self.hm_bins = 800

        c = np.unique(self.coord, axis=0)
        c = np.round(c * (self.hm_bins - 1)).astype(int)

        self.hm_sharp = np.zeros([self.hm_bins, self.hm_bins])
        np.add.at(self.hm_sharp, (c[:, 1], c[:, 0]), 1)

        self.update_hm()

        #  --------------------------------------------------------------------------------

        self.set_default_volume()  # Warm up

        self.fig_hm.canvas.mpl_connect('button_press_event', self.onClickHm)
        self.fig_hm.canvas.mpl_connect('key_press_event', self.onKeyPressEvent)
        self.ax_hm.callbacks.connect('xlim_changed', self.axis_change_event)

        # VOLUME RENDERER PROCESS ---------------------------------------------------------------------------

        volume_render_process = mp.Process(
            target=volumeRendererProcessLoop,
            args=(
                self.volume_render_input_queue,
                self.volume_render_output_queue
            )
        )
        volume_render_process.start()

        volume_render_event_thread = threading.Thread(target=self.volume_renderer_event)
        volume_render_event_thread.start()

        # Start process loop --------------------------------------------------------------------------------

        debug_print("init() show")

        try:
            plt.show()
        except KeyboardInterrupt:
            print("Exiting!")

        debug_print("init() exit")

        self.volume_render_input_queue.put("exit")
        self.volume_render_output_queue.put("exit")
        volume_render_process.join()
        volume_render_process.terminate()
        volume_render_event_thread.join()

    @debug_decorator
    def axis_change_event(self, event_ax):
        pass
        #print("New axis y-limits are",  event_ax.get_ylim())

    @debug_decorator
    def get_volume(self, sf):
        vol = self.summary(sf.to(self.device))
        return vol.detach().cpu().numpy()

    @debug_decorator
    def set_default_volume(self):
        self.volume_render_input_queue.put([
            self.get_volume(self.structure_factors_mean)
        ])

    @debug_decorator
    def clear_selection(self, _=None):
        for i in range(len(self.circles)):
            self.circles[i].remove()

        self.circles.clear()
        self.circles_coord.clear()
        self.selected_ids.clear()
        self.volumes.clear()
        if self.lines is not None:
            self.lines.remove()
            self.lines = None

        self.clear_subset_selection()

        self.set_default_volume()
        self.fig_hm.canvas.draw()

    @debug_decorator
    def clear_subset_selection(self, _=None):
        if self.selector_obj is not None:
            self.selector_obj.clear()
            del self.selector_obj
            self.selector_obj = None
            self.subset_selection_ongoing = False

    @debug_decorator
    def ui_visible(self, visible=False):
        self.clear_button_axes.set_visible(visible)
        self.subset_button_axes.set_visible(visible)
        self.save_volumes_button_axes.set_visible(visible)
        self.save_images_button_axes.set_visible(visible)
        self.hm_up_button_axes.set_visible(visible)
        self.hm_down_button_axes.set_visible(visible)
        self.bfactor_up_button_axes.set_visible(visible)
        self.bfactor_down_button_axes.set_visible(visible)
        self.iso_value_up_button_axes.set_visible(visible)
        self.iso_value_down_button_axes.set_visible(visible)

        self.hm_text_axes.set_visible(visible)
        self.bfactor_text_axes.set_visible(visible)
        self.iso_value_text_axes.set_visible(visible)

    @debug_decorator
    def save_dataset_indices(self, vertices):
        path = Path(vertices)
        mask = path.contains_points(self.coord)
        indices = np.arange(0, self.coord.shape[0])[mask]

        path = f"subset_{self.subset_index}.csv"
        percent = round((len(indices) / self.coord.shape[0]) * 100, 2)
        print(f"Saving indices of {len(indices)} ({percent}%) selected particles to {path}")
        np.savetxt(path, indices, delimiter=',', fmt='%d')

        path = f"subset_{self.subset_index}.png"
        print(f"Saving plot of selection to {path}")
        self.ui_visible(False)
        plt.savefig(path)
        self.ui_visible(True)

        # self.ax_hm.plot(self.coord[mask, 0], self.coord[mask, 1])
        
        N = min(len(indices), 500)
        if N > 0:
            render_indices = np.random.choice(indices, N, replace=False) if len(indices) > N else indices
            vol = self.get_volume(self.structure_factors[render_indices[0]])
            for i in range(len(render_indices) - 1):
                vol += self.get_volume(self.structure_factors[render_indices[i + 1]])
            vol /= N
            self.volumes.append(vol)
            self.volume_render_input_queue.put(self.volumes)

        self.subset_index += 1
        self.clear_subset_selection()

    @debug_decorator
    def subset_selection(self, _=None):
        self.clear_selection()
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

    @debug_decorator
    def save_selected_volumes(self, _=None):
        print('Saving selected structures to MRC-files:')
        for i, v in enumerate(self.volumes):
            path = f"particle_id_{str(self.selected_ids[i])}.mrc"
            print(f" {path}")
            save_mrc(v, path)

    @debug_decorator
    def save_volume_images(self, _=None):
        self.volume_render_input_queue.put("save_images")

    @torch.no_grad()
    @debug_decorator
    def update_hm(self):
        if self.hm_contrast < 0.01:
            sharp = self.hm_sharp
        else:
            sharp = 1 - (self.hm_sharp + 1) ** (-self.hm_contrast)
        blur = gaussian_blur(sharp, self.hm_sigma)

        self.ax_hm.imshow(blur, cmap=self.hm_cm, extent=(0, 1, 1, 0))

        self.ax_hm.set_xlim([0, 1])
        self.ax_hm.set_ylim([0, 1])

        self.hm_sigma_text.set_text(f"Smooth\n{round(self.hm_sigma)}")
        self.fig_hm.canvas.blit(self.hm_text_axes.bbox)

        self.hmc_sigma_text.set_text(f"Contrast\n{round(self.hm_contrast, 1)}")
        self.fig_hm.canvas.blit(self.hmc_text_axes.bbox)

        self.fig_hm.canvas.draw()

    @torch.no_grad()
    @debug_decorator
    def update_bfactor(self):
        idx = torch.linspace(0, 2, self.ft_shape[-1] * 2, device=self.device)
        res2 = torch.square(idx / self.voxel_size)
        profile = torch.exp(-self.bfactor / 4. * res2)
        grid = spectra_to_grid(profile, self.spectral_indices)

        for i in range(self.summary.basis.shape[0]):
            df = self.ref_basis_df[i].to(self.device) * grid
            base = idft(df, real_in=True, centered=False)
            self.summary.basis[i] = base * self.mask

        if len(self.volumes) > 0:
            self.volumes = []
            for i in range(len(self.selected_ids)):
                idx = self.selected_ids[i]
                vol = self.get_volume(self.structure_factors[idx])
                self.volumes.append(vol)
            self.volume_render_input_queue.put(self.volumes)
        else:
            self.set_default_volume()

        self.bfactor_text.set_text(f"B-factor\n{round(self.bfactor)}")
        self.fig_hm.canvas.blit(self.iso_value_text_axes.bbox)
        self.fig_hm.canvas.draw()

    @debug_decorator
    def raise_hm_sigma(self, _=None):
        self.hm_sigma += 1
        self.update_hm()

    @debug_decorator
    def lower_hm_sigma(self, _=None):
        self.hm_sigma = max(2, self.hm_sigma - 1)
        self.update_hm()

    @debug_decorator
    def raise_hm_contrast(self, _=None):
        self.hm_contrast += 0.2
        self.update_hm()

    @debug_decorator
    def lower_hm_contrast(self, _=None):
        self.hm_contrast = max(0, self.hm_contrast - 0.2)
        self.update_hm()

    @debug_decorator
    def raise_bfactor(self, _=None):
        self.bfactor += self.bfactor_step
        self.update_bfactor()

    @debug_decorator
    def lower_bfactor(self, _=None):
        self.bfactor -= self.bfactor_step
        self.update_bfactor()

    @debug_decorator
    def raise_iso_value(self, _=None):
        self.volume_render_input_queue.put("up")

    @debug_decorator
    def lower_iso_value(self, _=None):
        self.volume_render_input_queue.put("down")

    @debug_decorator
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
            if self.lines is not None:
                self.lines.remove()
                self.lines = None
            if len(self.circles_coord) > 1:
                c = np.array(self.circles_coord)
                self.lines, = self.ax_hm.plot(c[:, 0], c[:, 1], 'k-', alpha=0.5)
            
            if len(self.volumes) == 0:
                self.set_default_volume()
            else:
                self.volume_render_input_queue.put(self.volumes)
            self.fig_hm.canvas.draw()

    @debug_decorator
    def handle_key_press(self, key):
        if key == 'escape':
            self.clear_selection()
        elif key == "left":
            self.lower_bfactor()
        elif key == "right":
            self.raise_bfactor()

    @debug_decorator
    def onKeyPressEvent(self, event):
        key = event.key
        self.handle_key_press(key)
        self.volume_render_input_queue.put(key)

    @debug_decorator
    def volume_renderer_event(self):
        while True:
            task = self.volume_render_output_queue.get()

            if task is not None and isinstance(task, str):
                debug_print(f"volume_renderer_event() task: {task}")

                if task == "exit":
                    break
                elif len(task) > 10 and task[:10] == "iso_value_":
                    iso_value = round(float(task[10:]), 3)
                    self.iso_value_text.set_text(f"Iso value\n{iso_value}")
                    self.fig_hm.canvas.blit(self.iso_value_text_axes.bbox)
                    self.fig_hm.canvas.draw()
                elif len(task) > 4 and task[:4] == "key_":
                    self.handle_key_press(task[4:].lower())

        debug_print(f"volume_renderer_event() loop exit")


def main(args):

    torch.no_grad()
    device, _ = setup_device(args)

    print("Loading modules...")

    summary = Summary.load_from_path(args.path, device=device)
    embed = summary.metadata['z']
    structure_factors = summary.metadata['s']

    if embed.size(1) > 2:
        print("Creating 2D representation...")

        embed = (embed - embed.mean(dim=0, keepdim=True)) / (embed.std() + 1e-12)

        if args.umap:
            embed = embed.cpu().detach().numpy().astype(np.float32)

            print("Applying UMAP...")
            import umap
            embed = umap.UMAP().fit_transform(embed.astype(np.float32))

            embed = torch.from_numpy(embed)
        else:
            print("Applying PCA...")
            embed = pca_dim_reduction(embed)
            embed = embed[:, :2]

        # from sklearn.manifold import TSNE
        # embed = TSNE(n_components=2, perplexity=200).fit_transform(embed.astype(np.float32))

        # import gpumap
        # embed = gpumap.GPUMAP().fit_transform(embed).astype(np.float32)

    print("Visualizing representation...")

    viewer = Viewer(summary, embed, structure_factors, device, args.path, args.bfac_step)


def append_args(parser):
    parser.add_argument('path', help='input summary file or logdir', type=str)
    parser.add_argument('--gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--dont_cache_embed', action="store_true")
    parser.add_argument('--ignore_cached_embed', action="store_true")
    parser.add_argument('--scale_invar', action="store_true")
    parser.add_argument('--umap', action="store_true")
    parser.add_argument('--bfac_step', type=float, default=20, help='B-factor steps')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Used to make a visualize SHA3D training summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)
