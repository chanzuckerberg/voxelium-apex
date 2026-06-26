#!/usr/bin/env python

"""
Voxelium-Apex command line interface.

This module defines the full ``rich_click`` command tree. It is deliberately
kept free of heavy imports (``torch``, ``voxelium``, ``matplotlib``, ...): the
option/argument definitions only depend on ``rich_click`` so that displaying the
help menu for any command never pays the cost of importing PyTorch.

The actual work lives in the individual command modules. Their ``main(args)``
functions (and the heavy modules they pull in) are imported lazily *inside* the
command callbacks, i.e. only when a command is actually executed.
"""

import voxelium_apex.groups as groups
from argparse import Namespace
import rich_click as click
from pathlib import Path
import voxelium_apex

# Render argument tables in the help output (e.g. positional ``log_dir``).
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = False
_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


def _options(decls):
    """Apply a list of ``click`` option/argument decorators to a callback.

    The decorators are applied in reverse so that the help output lists them in
    the same order as ``decls`` (top-to-bottom).
    """

    def decorator(func):
        for decl in reversed(decls):
            func = decl(func)
        return func

    return decorator


def _run(module_path, kwargs, func=None):
    """Lazily import ``module_path`` and dispatch to its ``main(args)``.

    Heavy imports happen here, on invocation, never at definition/help time.
    The collected ``click`` parameters are repackaged into an
    ``argparse.Namespace`` so the existing ``main`` implementations work
    unchanged. If ``func`` is given, that function is called directly with
    ``**kwargs`` instead.
    """
    import importlib

    module = importlib.import_module(module_path)
    if func is not None:
        raise SystemExit(getattr(module, func)(**kwargs))
    raise SystemExit(module.main(Namespace(**kwargs)))


@click.group(context_settings=_CONTEXT_SETTINGS)
@click.version_option(voxelium_apex.__version__, "--version", prog_name="voxelium-apex")
def cli():
    """Voxelium-Apex: cryo-EM/cryo-ET heterogeneity reconstruction analysis."""


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
@cli.group(context_settings=_CONTEXT_SETTINGS)
def analysis():
    """Spectral statistics and STAR-file utilities."""

@analysis.command("export", context_settings=_CONTEXT_SETTINGS, no_args_is_help=True)
@_options([
    click.option("--source-particles", "-sp", required=True, default='particles.star', type=str, help="Source Particles STAR-file."),
    click.option("--expanded-particles", "-ep", required=False, type=str, default=None, help="Expanded Particles STAR-file."), 
    click.option("--subset", "-s", required=False, type=str, default='subset_1.csv',
                 help="CSV-File containing row indices"),
    click.option("--output", "-o", "--out", type=str, default='subset_particles.star', help="Output STAR-file"),
])
def analysis_star_file_subset_export(**kwargs):
    """Export a subset of rows from a STAR-file."""
    _run("voxelium_apex.utils.star_file_subset", kwargs, func="export_subset")


@analysis.command("spectral", context_settings=_CONTEXT_SETTINGS)
@_options([
    click.argument("mrcs", nargs=-1, type=str),
    click.option("-f", "--fsc", is_flag=True, default=False,
                 help="Calculate Fourier shell correlations, first file is reference."),
    click.option("-p", "--powers", is_flag=True, default=False,
                 help="Calculate spectral powers"),
    click.option("-c", "--csv", type=str, default=None,
                 help="Rather than plot, dump to this CSV file path"),
    click.option("-m", "--mask", type=str, default=None, help="Solvent mask"),
    click.option("-v", "--voxel-size", type=float, default=None, help="Overwrite voxel size"),
    click.option("-r", "--max-resolution", type=float, default=None,
                 help="Maximum resolution (Angstrom) to calculate"),
    click.option("-t", "--title", type=str, default=None, help="Title to set for the plot"),
    click.option("--spherical-mask", is_flag=True, default=False,
                 help="Apply a real-space spherical mask"),
])
def analysis_spectral(**kwargs):
    """Plot or export a CSV of spectral statistics."""
    _run("voxelium_apex.utils.spectral", kwargs)


@analysis.command("star-subset", context_settings=_CONTEXT_SETTINGS)
@_options([
    click.option("--input", "-i", "--in", required=True, type=str, help="Input STAR-file"),
    click.option("--index", "-n", "--idx", required=True, type=str,
                 help="File containing row indices"),
    click.option("--output", "-o", "--out", type=str, default=None, help="Output STAR-file"),
    click.option("--overwrite", is_flag=True, default=False, help="Overwrite the output STAR-file"),
    click.option("--table", type=str, default="particles",
                 help="Name of the table to select from"),
    click.option("--exclude", is_flag=True, default=False,
                 help="Instead of including the indices in --index, exclude them"),
])
def analysis_star_subset(**kwargs):
    """Extract a subset of rows from a STAR-file."""
    _run("voxelium_apex.utils.star_file_subset", kwargs)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
@cli.command("extract", context_settings=_CONTEXT_SETTINGS, no_args_is_help=True)
@_options([
    # --- copick inputs / output source: exactly one of the two ---
    click.option("--picks-uri", '-pu', type=str, default=None,
                 help="copick picks URI 'object_name:user_id/session_id' (use '*' or omit for any). "
                      "Requires --copick-config. Mutually exclusive with --particles."),
    click.option("--config", '-cc', type=click.Path(exists=True, dir_okay=False, path_type=Path),
                 default=None, help="Path to the copick configuration file (required with --picks-uri)."),
    click.option("--runs", '-r', type=str, default=None,
                 help="Restrict copick picks to these run names (comma/space separated; default: all runs)."),
    # --- relion inputs ---
    click.option("--particles", '-p', type=click.Path(dir_okay=False, path_type=Path),
                 default=None, help="Path to a RELION particles.star file, relative to --workspace. Mutually exclusive with --picks-uri."),
    click.option("--tomograms", '-t', type=click.Path(dir_okay=False, path_type=Path),
                 required=True, help="Path to the tomograms.star file (one entry per tiltseries), relative to --workspace."),
    click.option("--motion", '-m', type=click.Path(dir_okay=False, path_type=Path),
                 default=None, help="Path to the motion.star file (one entry per particle), relative to --workspace."),
    click.option('--workspace', '-w', type=click.Path(exists=True, dir_okay=True, path_type=Path), default=None,
                 help="Path to the workspace directory. If omitted, the directory of the particles.star file is used."),
    # --- output ---
    click.option("--output", '-o', type=click.Path(file_okay=False, path_type=Path), required=False, default='output',
                 help="Directory where the extracted subtomograms are written."),
    # --- shared parameters ---
    click.option("--box-size", '-bs', type=int, required=True,
                 help="Box size of the extracted subtomograms, in pixels."),
click.option("--crop-size", '-cs', type=int, default=None,
                 help="Crop size in pixels. If omitted, defaults to the box size."),
    click.option("--bin", '-b', type=int, default=2, show_default=True, help="Binning factor for the subtomograms."),
    click.option("--debug", '-d', is_flag=True, default=False, help="Enable debug logging."),
])
def extract(**kwargs):
    """Extract subtomograms from a particles.star file or copick picks."""
    particles = kwargs.get("particles")
    picks_uri = kwargs.get("picks_uri")
    if bool(particles) == bool(picks_uri):
        raise click.UsageError(
            "Provide exactly one of --particles or --picks-uri."
        )
    if picks_uri and not kwargs.get("config"):
        raise click.UsageError("--picks-uri requires --config.")
    _run("voxelium_apex.importers.extract", kwargs)


# ---------------------------------------------------------------------------
# sha3d
# ---------------------------------------------------------------------------
@cli.group(context_settings=_CONTEXT_SETTINGS)
def sha3d():
    """Spectral heterogeneity analysis (SHA) 3D tools."""


@sha3d.command("train", context_settings=_CONTEXT_SETTINGS, no_args_is_help=True)
@_options([
    click.argument("log_dir", type=str,help="Output Directory where results are saved."),
    click.option("--input", "-i", type=str, default=None,
                 help="input job (job directory or optimizer-file)"),
    click.option("--particle_diameter", "--pd", type=int, default=None,
                 help="size of circular mask (ang)"),
    click.option("--circular_mask_thickness", "-cmt", type=int, default=20,
                 help="thickness of mask (ang)"),
    click.option("--batch_size", "-bs", type=int, default=256,
                 help="mini-batch size from training dataset"),
    click.option("--overwrite", "--ow", is_flag=True, default=False,
                 help="Overwrite an existing log directory"),
    click.option("--gpu", "-gpu", type=str, default=None, help="gpu to use"),
    click.option("--checkpoint_time", "-ct", type=int, default=10,
                 help="Minimum time in minutes between checkpoint saves"),
    click.option("--image_steps", "-is", type=int, default=500,
                 help="Log tensorboard images every n steps"),
    click.option("--stats_steps", "-ss", type=int, default=100,
                 help="Log tensorboard statistics every n steps"),
    click.option("--max_steps", "-ms", "--steps", type=int, default=int(1e9),
                 help="number of steps to train"),
    click.option("--max_train_epochs", "--epochs", "-e", type=int, default=10,
                 help="number of epochs to train"),
    click.option("--preload", "-p", is_flag=True, default=False,
                 help="Preload all particle data into memory before training"),
    click.option("--dont_finalize", is_flag=True, default=False,
                 help="Skip the finalization epoch after training"),
    click.option("--only_finalize", is_flag=True, default=False,
                 help="Skip training and only run the finalization epoch"),
    click.option("--pytorch_threads", type=int, default=6,
                 help="Number of PyTorch CPU threads"),
    click.option("--dataloader_threads", "--dj", type=int, default=2,
                 help="Number of dataloader worker threads"),
    click.option("--tomo", is_flag=True, default=False,
                 help="Enable subtomogram averaging mode"),
    click.option("--z_size", "-z", type=int, default=2,
                 help="Number of learnt representation dimensions."),
    click.option("--s_size", "-s", type=int, default=8, help="Number of structure basis."),
    click.option("--do_align", is_flag=True, default=False,
                 help="Do optimize pose and translation"),
    click.option("--do_ctf_optimization", is_flag=True, default=False,
                 help="Do optimize CTF defocus and angle"),
    click.option("--solvent_mask", "--sm", type=str, default=None,
                 help="MRC file with ones in the region that is not solvent (region of interest)"),
    click.option("--roi_mask", "-roi", type=str, default=None,
                 help="If a mask is provided, allow only structural heterogeneity inside the masked region."),
    click.option("--subtract_mask", type=str, default=None,
                 help="If a mask is provided, create a new particle stack where everything outside the mask is subtracted."),
    click.option("--subtract_buffer_size", type=int, default=500,
                 help="Maximum buffer size for subtracted data in MiB."),
    click.option("--profile_runtime", is_flag=True, default=False,
                 help="Profile runtime performance with the PyTorch profiler"),
    click.option("--decoder_begin_lr", type=click.FloatRange(min=0), default=0.1,
                 help="Starting learning rate of the structure decoder"),
    click.option("--decoder_lr", type=click.FloatRange(min=0), default=0.01,
                 help="Learning rate of the structure decoder"),
    click.option("--encoder_final_lr", type=click.FloatRange(min=0), default=1e-5,
                 help="Final learning rate of the encoders"),
    click.option("--encoder_lr", type=click.FloatRange(min=0), default=1e-4,
                 help="Learning rate of the encoders"),
    click.option("--relax_lr", type=click.FloatRange(min=0), default=0.02,
                 help="Learning rate for the relaxation during finalization"),
    click.option("--relax_iter", type=click.IntRange(min=0), default=10,
                 help="Number of iterations for the relaxation during finalization"),
    click.option("--grad_clip", type=click.FloatRange(min=0), default=1e-2,
                 help="Gradient clipping of the encoder"),
    click.option("--regularization", "-reg", type=click.FloatRange(min=0), default=0.,
                 help="Output power regularization"),
    click.option("--s_consistency_weight", type=click.FloatRange(min=0), default=1.,
                 help="Consistency of the embedding"),
    click.option("--s_consistency_scheduler", type=str, default=None,
                 help="Apply consistency loss with a schedule"),
    click.option("--smoothness_distance", type=click.FloatRange(min=0, max=1), default=.3,
                 help="Pair distance for smoothness loss"),
    click.option("--smoothness_distance_min", type=click.FloatRange(min=0), default=1e-2,
                 help="Minimum fraction of the mean pair distance for smoothness loss"),
    click.option("--s_l1_weight", "-l1", type=click.FloatRange(min=0), default=0,
                 help="S L1 loss weight"),
    click.option("--s_l2_weight", "-l2", type=click.FloatRange(min=0), default=1e-4,
                 help="S L2 loss weight"),
    click.option("--z_compactness_weight", "-zc", type=click.FloatRange(min=0), default=1e-4,
                 help="Compactness of the embedding"),
    click.option("--feature_bandpass", "-fb", type=str, default=None,
                 help="Feature extraction band filters (in Ångströms). Comma separated, eg. 3-20,5-20"),
    click.option("--mse_bandpass",  type=str, default=None,
                 help="MSE weighting band filters (in Ångströms), eg. 3-20"),
    click.option("--feature_noise_weight", is_flag=True, default=False,
                 help="Weight features by noise power rather than signal power"),
    click.option("--z_encoder_dims", "-zed", type=str, default="128,128,128",
                 help="Comma separated integers used for Z-encoder hidden layer dimensions."),
    click.option("--s_encoder_dims", "-sed", type=str, default="128,128,128,128",
                 help="Comma separated integers used for S-encoder hidden layer dimensions."),
    click.option("--dtype", type=str, default="float32",
                 help="Data type used for storing images in data set"),
    click.option("--dont_postprocess", is_flag=True, default=False,
                 help="Skip post-processing of the reconstruction after training"),
    click.option("--only_update_representation", is_flag=True, default=False,
                 help="Freeze the decoder and only update the latent representation"),
    click.option("--validation-fraction", type=float, default=0.05,
                 help="Fraction of dataset to be used for validation."),
    click.option("--cache", type=str, default=None, help="Cache directory"),
    click.option("--max_data_resolution", type=float, default=None,
                 help="Minimum data resolution, in Ångströms"),
    click.option("--dampen", type=float, default=1., help="Regularization parameter"),
])
def sha3d_train(**kwargs):
    """Train a spectral heterogeneity analysis (SHA) 3D model."""
    _run("voxelium_apex.sha3d.train.train", kwargs)


@sha3d.command("summary", context_settings=_CONTEXT_SETTINGS)
@_options([
    click.argument("logdir", type=str),
    click.option("--state", "-s", "--s", type=str, default=None, help="State label to choose"),
    click.option("--output", "-o", "--o", type=str, default=None, help="Output file"),
    click.option("--mask", "-m", "--m", type=str, default=None, help="Mask file"),
    click.option("--no_gzip", "-z", "--z", is_flag=True, default=False,
                 help="Do not compress the output file"),
])
def sha3d_summary(**kwargs):
    """Build a summary from a SHA3D training log directory."""
    _run("voxelium_apex.sha3d.analysis.summary", kwargs)


@sha3d.command("viewer", context_settings=_CONTEXT_SETTINGS, no_args_is_help=True)
@_options([
    click.argument("path", type=str, help="Path to the summary file to visualize."),
    click.option("--gpu", "-gpu", type=str, default=None, help="gpu to use"),
    click.option("--dont_cache_embed", is_flag=True, default=False),
    click.option("--ignore_cached_embed", is_flag=True, default=False),
    click.option("--scale_invar", is_flag=True, default=False),
    click.option("--umap", is_flag=True, default=False),
    click.option("--bfac_step", type=float, default=20, help="B-factor steps"),
])
def sha3d_viewer(**kwargs):
    """Visualize SHA3D training summaries."""
    _run("voxelium_apex.sha3d.view.viewer", kwargs)


@sha3d.command("submit", context_settings=_CONTEXT_SETTINGS)
@_options([
    click.option("--wd", type=str, default=None, help="working directory"),
    click.option("--logdir_root", type=str, default="positron_logdirs",
                 help="root of the log directories"),
    click.option("--defaults", type=str, default=".positron_defaults.json",
                 help="default value JSON-file"),
])
def sha3d_submit(**kwargs):
    """Simple GUI for submitting SHA3D jobs."""
    _run("voxelium_apex.sha3d.infra.job_submit", kwargs)


if __name__ == "__main__":
    cli()
