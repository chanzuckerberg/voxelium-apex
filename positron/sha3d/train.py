#!/usr/bin/env python3

"""
Module for training a 3D spectral heterogeneity analysis (SHA) model
"""
import argparse
import random
import time

from datetime import datetime

import torch
from torch.utils.data import DataLoader

from positron.sha3d.feature_extractor import FeatureExtractor
from positron.sha3d.loss_functions import similarity
from positron.sha3d.mask_applicator import MaskApplicator
from positron.sha3d.spectral_statistics import SpectralStatistics
from positron.base.single_particle_validation_sampler import SingleParticleValidationSampler
from positron.base.subtomo_validation_sampler import SubtomoValidationSampler
from positron.sha3d.subtraction import SubtractionHelper
from positron.sha3d.train_arguments import append_train_arguments

from positron.base.torch_utils import make_series_line_fig, make_line_fig
from positron.sha3d.distributed_processing import DistributedProcessing
from positron.sha3d.tensorboard_utils import TensorboardSummary
from positron.sha3d.train_utils import *
from positron.base import load_mrc
from positron.base.spectral import fourier_shift_2d, spectral_index_from_resolution, spectrum_to_grid_mean
from positron.base.io_logger import IOLogger
from positron.sha3d.data_analysis_container import DatasetAnalysisContainer
from positron.sha3d.retention_classifier import RetentionClassifier


def cyclic_lr(min_lr, max_lr, x, max_x):
    lam = x / max_x
    cycle = (np.cos(np.pi * x / 2) + 1) / 2
    return (max_lr - min_lr) * cycle * cosine_descend(0.5, 1., lam) + min_lr


def train(rank, args, ddp_args):
    ###############################################
    # SETUP
    ###############################################

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # warnings.simplefilter('error', UserWarning)
    # torch.autograd.set_detect_anomaly(True)

    DistributedProcessing.process_setup(rank=rank, args=ddp_args)

    log_dir = args.log_dir
    torch.set_num_threads(args.pytorch_threads)  # TODO manage in distributed process flow

    device = DistributedProcessing.get_device()
    dac = DatasetAnalysisContainer.initialize_from_args(args, device=device)

    # Setup shortcuts
    rec = dac.reconstruction_container
    hvc = dac.hidden_variable_container

    dataset = dac.particle_dataset
    dataset_size = dataset.get_size()

    image_size = rec.image_size
    pixel_size = rec.voxel_size
    image_max_r = rec.max_r

    validation_fraction = args.validation_fraction

    train_batch_size = args.batch_size
    valid_batch_size = max(int(train_batch_size * validation_fraction) + 2, 8)

    if args.tomo:
        sampler = SubtomoValidationSampler(
            group_indices=dataset.part_tomo_idx,
            valid_fraction=validation_fraction,
            valid_batch_size=valid_batch_size,
            train_batch_size=train_batch_size
        )
    else:
        sampler = SingleParticleValidationSampler(
            num_samples=dataset_size,
            valid_fraction=validation_fraction,
            valid_batch_size=valid_batch_size,
            train_batch_size=train_batch_size
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=args.dataloader_threads
    )

    if args.do_align:
        hvc.do_align()
    if args.do_ctf_optimization:
        hvc.do_ctf_optimization()

    dataset.setup_ctfs(compute_ctf=False)

    ###############################################
    # SETUP OPTIMIZATION STATS
    ###############################################

    print("\nINITIALIZING RUN", datetime.now().ctime())

    timing_shown_count = 0

    print(sampler)
    print("Image size", image_size)
    print("Pixel size", round(pixel_size, 2))

    ###############################################
    # REAL-SPACE MASKS
    ###############################################

    solvent_mask = None
    if args.solvent_mask is not None:
        solvent_mask, _, _ = load_mrc(args.solvent_mask)
        solvent_mask = torch.Tensor(solvent_mask.copy()).to(device)

        if not torch.any((0. < solvent_mask) & (solvent_mask < 1.)):
            print("\nWARNING: solvent mask should only contain values in the range (including) zero and one.\n")

        # solvent_mask = solvent_mask.pow(1. / 10.)

    roi_mask = None
    do_roi = False
    if args.roi_mask is not None:
        roi_mask, _, _ = load_mrc(args.roi_mask)
        roi_mask = torch.Tensor(roi_mask.copy()).to(device)

        if not torch.any((0. < roi_mask) & (roi_mask < 1.)):
            print("\nWARNING: ROI mask should only contain values in the range (including) zero and one.\n")

        do_roi = True

    subtract_mask = None
    if args.subtract_mask is not None:
        subtract_mask, _, _ = load_mrc(args.roi_mask)
        subtract_mask = torch.Tensor(subtract_mask.copy()).to(device)

        if not torch.any((0. < subtract_mask) & (subtract_mask < 1.)):
            print("\nWARNING: Subtraction mask should only contain values in the range (including) zero and one.\n")

    ###############################################
    # TENSORBOARD
    ###############################################

    init_step_write_images = [50, 100, 200, 400, 800]

    summary = TensorboardSummary(log_dir, pixel_size, image_max_r)

    ###############################################
    # PROFILING
    ###############################################

    prof = None
    if args.profile_runtime:
        prof = torch.profiler.profile(
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            activities=[torch.profiler.ProfilerActivity.CPU],
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(log_dir, "profiler")),
            # record_shapes=True,
            # profile_memory=True,
            # with_stack=True
        )
        prof.__enter__()

    ###############################################
    # RUN TRAINING
    ###############################################

    last_save_time = time.time()

    max_train_epochs = args.max_train_epochs
    if args.only_finalize:
        max_epochs = rec.train_epoch + 1
    else:
        max_epochs = max_train_epochs if args.dont_finalize else max_train_epochs + 1

    # TODO remove voxel_size input
    feature_extractor = FeatureExtractor(
        decoder=rec.decoder,
        voxel_size=pixel_size,
        bandpass=rec.feature_bandpass,
        image_max_r=image_max_r
    )

    solvent_mask_applicator = MaskApplicator(
        rec.decoder.projector,
        solvent_mask=solvent_mask,
        roi_mask=roi_mask
    )

    subtract_during_finalize = subtract_mask is not None

    # TODO remove this
    subtraction_helper = SubtractionHelper(
        log_dir=os.path.join(log_dir, "subtract"),
        rec=rec,
        mask=subtract_mask,
        image_max_r=image_max_r,
        buffer_size=args.subtract_buffer_size * 1024**2
    )

    print(f"Feature extraction bandpass indices (max index is {rec.max_r}):", end="")
    for bf in rec.feature_bandpass:
        print(f" {bf[0]}-{bf[1]}", end="")
    print("")
    print(f"MSE weight bandpass indices (max index is {rec.max_r}): {rec.mse_bandpass[0]}-{rec.mse_bandpass[1]}")

    do_tomo = args.tomo
    
    z_relax_losses = np.zeros(args.relax_iter)
    z_relax_losses_count = 0

    try:
        for epoch in np.arange(rec.train_epoch, max_epochs):
            dt = time.time()
            sampler.train()

            finalize = False
            if args.only_finalize or epoch == max_train_epochs:
                print("Finalizing...")
                finalize = True
                sampler.eval()
                torch.no_grad()

            final_train_epochs = False
            if epoch >= max_train_epochs - 1:
                final_train_epochs = True
                if not finalize:
                    print("Last training epochs...")

            if finalize and subtract_during_finalize:
                subtraction_helper.initialize()

            for batch_idx, sample in enumerate(data_loader):
                step = rec.train_step
                if step >= args.max_steps:
                    raise StopIteration("Maximum step/epoch count reached.")

                if step == 0:
                    summary.write_hidden_variable(hvc)  # Write initial values

                epoch_partial = epoch + float(batch_idx) / float(len(data_loader))

                log_stats = step % args.stats_steps == 0 and step > 0 and not finalize
                log_images = (step % args.image_steps == 0 and step > 0
                              or step in init_step_write_images) and not finalize

                summary.set_step(step)

                tomo_groups = None
                if do_tomo:
                    _, tomo_groups = torch.unique(sample["tomo_idx"], return_inverse=True)
                    tomo_groups = tomo_groups.to(device)

                particle_idx = sample["idx"]

                if finalize:
                    rec.eval()
                else:
                    rec.train()
                rec.zero_grad()

                train_mask = sampler.get_current_train_mask(particle_idx).to(device)
                this_batch_size = len(train_mask)

                y_ft, hv = preprocess_batch_data(sample, dac)

                tt = time.time()

                # TODO Clean-up
                y_weight = rec.stats.get_y_weight()
                x_signal_pow = rec.stats.get_x_signal()

                if args.feature_noise_weight:
                    wy = y_weight
                else:
                    wy = y_weight * x_signal_pow

                if step > 200:
                    fsc_mask = rec.stats.get_fsc_spectrum() < 0.5
                    wy[fsc_mask] = 0

                s0_ = rec.s0_ema if do_roi else None

                features = feature_extractor(
                    hv=hv, y=y_ft, wy=wy, wx=torch.ones_like(wy), groups=tomo_groups, s0=s0_)

                if log_stats:
                    summary.add_scalar("Features/mean", features.mean())
                    summary.add_scalar("Features/std", features.std())

                features = rec.normalize_features(features)
                hvc.set_metadata('feature', particle_idx, features[tomo_groups] if do_tomo else features)

                invert_timing = time.time() - tt

                z, _ = rec.z_encode(features)
                s = rec.s_encode(z)

                hvc.set_metadata('z', particle_idx, z[tomo_groups] if do_tomo else z)
                hvc.set_metadata('s', particle_idx, s[tomo_groups] if do_tomo else s)

                if finalize:
                    if args.relax_iter > 0:
                        s_relaxed = s.detach()

                        from torch.optim import LBFGS

                        # Clone z without detaching so it stays in the computation graph
                        z_relaxed = z.detach().requires_grad_(True)
                        z_relaxed.retain_grad()

                        optimizer = torch.optim.Adam([z_relaxed], lr=args.relax_lr)
                        
                        for i in range(args.relax_iter):
                            def closure():
                                rec.zero_grad()
                                optimizer.zero_grad()

                                # Compute s_relaxed based on the updated z_relaxed
                                s_relaxed = rec.s_encode(z_relaxed)

                                # Use the appropriate s based on the tomo condition
                                s_ = s_relaxed[tomo_groups] if do_tomo else s_relaxed

                                x_ft = rec.decoder(s=s_, max_r=image_max_r, rot_matrices=hv["rot_matrices"])
                                x_ft_shift = fourier_shift_2d(x_ft, hv["shifts_resid"])
                                x = x_ft_shift * hv['ctfs_'][..., None]

                                x_ = torch.view_as_complex(x)
                                y_ft_ = torch.view_as_complex(y_ft)
                                y_weight = rec.stats.get_y_weight(eps=1e-3)

                                spectral_mask = Cache.get_spectral_mask(
                                    x_ft.shape[1:-1],
                                    max_r=image_max_r,
                                    device=device
                                )

                                weight = y_weight / (y_weight.mean() + 1e-3)
                                if step > 200:
                                    fsc_mask = rec.stats.get_fsc_spectrum() < 0.3
                                    weight[fsc_mask] = 1e-3
                                weight *= rec.get_mse_weight_spectrum().to(device)
                                weight_grid = Cache.spectra_to_grids(weight, hv['ctfs_'].shape[1:], image_max_r)
                                x_ = torch.view_as_complex(x)
                                y_ft_ = torch.view_as_complex(y_ft)
                                square_error = (x_ - y_ft_.detach()).abs().square()
                                square_error_w = square_error * weight_grid[None]
                                loss = (
                                        square_error_w[:, spectral_mask].mean(0).sum() /
                                        (weight_grid[spectral_mask].sum() + 1e-12)
                                )
                                
                                # Backpropagate the loss to calculate gradients
                                loss.backward()

                                return loss
                            
                            z_relax_losses[i] += optimizer.step(closure).item()
                        
                        z_relax_losses_count += 1

                        # Compute s_relaxed based on the updated z_relaxed
                        s_relaxed = rec.s_encode(z_relaxed)

                        hvc.set_metadata('z_relaxed', particle_idx, z_relaxed[tomo_groups] if do_tomo else z_relaxed)
                        hvc.set_metadata('s_relaxed', particle_idx, s_relaxed[tomo_groups] if do_tomo else s_relaxed)
                else:
                    if do_tomo:
                        z = z[tomo_groups]
                        s = s[tomo_groups]

                    if log_stats:
                        summary.add_scalar(f"Z/std", z.std())
                        summary.add_scalar(f"Z/mean", z.mean())

                        summary.add_scalar(f"S/std", s.std())
                        summary.add_scalar(f"S/mean", s.mean())
                        summary.add_scalar(f"S/norm mean", s.square().sum(1).sqrt().mean())
                        summary.add_scalar(f"S/norm std", s.square().sum(1).sqrt().std())

                        summary.add_scalar(f"S0/std", s[:, 0].std())
                        summary.add_scalar(f"S0/mean", s[:, 0].mean())
                        summary.add_scalar(f"S0/norm", s[:, 0].square().mean().sqrt())

                        if rec.s0_ema is not None:
                            summary.add_scalar(f"S0/ema", rec.s0_ema)

                    tt = time.time()

                    x_ft = rec.decoder(s=s, max_r=image_max_r, rot_matrices=hv["rot_matrices"])

                    x_ft_shift = fourier_shift_2d(x_ft, hv["shifts_resid"])
                    x = x_ft_shift * hv['ctfs_'][..., None]

                    spectral_mask = Cache.get_spectral_mask(
                        x_ft.shape[1:-1],
                        max_r=image_max_r,
                        device=device
                    )

                    y_weight = rec.stats.get_y_weight(eps=1e-3)
                    weight = y_weight / (y_weight.mean() + 1e-3)
                    if step > 200:
                        fsc_mask = rec.stats.get_fsc_spectrum() < 0.3
                        weight[fsc_mask] = 1e-3
                    weight *= rec.get_mse_weight_spectrum().to(device)
                    weight_grid = Cache.spectra_to_grids(weight, hv['ctfs_'].shape[1:], image_max_r)
                    x_ = torch.view_as_complex(x)
                    y_ft_ = torch.view_as_complex(y_ft)
                    square_error_train = (x_[train_mask] - y_ft_[train_mask].detach()).abs().square()
                    square_error_valid = (x_[~train_mask] - y_ft_[~train_mask].detach()).abs().square()
                    square_error_train_w = square_error_train * weight_grid[None]
                    weighted_mse = (
                            square_error_train_w[:, spectral_mask].mean(0).sum() /
                            (weight_grid[spectral_mask].sum() + 1e-12)
                    )
                    if step > 0:
                        total_loss = weighted_mse

                        fsc_spectrum = rec.stats.get_fsc_spectrum().clip(0.01, 0.99)

                        if args.regularization > 0:
                            regularization_weight = Cache.spectra_to_grids(
                                1 - fsc_spectrum, hv['ctfs_'].shape[1:], image_max_r)
                            x_ = torch.view_as_complex(x_ft)
                            regularization_w = x_.abs().square() * regularization_weight
                            regularization_loss = (
                                    regularization_w[:, spectral_mask].mean(0).sum() /
                                    (regularization_weight[spectral_mask].sum() + 1e-12)
                            )

                            if log_stats:
                                summary.add_scalar(f"Loss/Regularization", regularization_loss)
                            total_loss += regularization_loss * args.regularization

                        if args.s_consistency_weight > 0 and args.smoothness_distance > 0:
                            distances = torch.cdist(features, features)
                            distances.fill_diagonal_(float('inf'))
                            _, closest_indices = torch.min(distances, dim=1)
                            closest_point = features[closest_indices]
                            feature_noise = (closest_point - features).abs().clip(1e-3) * torch.randn_like(features)
                            features_ = features + feature_noise * args.smoothness_distance

                            z_noise, _ = rec.z_encode(features_)
                            s_noise = rec.s_encode(z_noise, features=features_)

                            if do_tomo:
                                s_noise = s_noise[tomo_groups]

                            s_ = s[:, 1:] if do_roi else s
                            s_noise_ = s_noise[:, 1:] if do_roi else s_noise

                            s_consistency_loss_train = (s_ - s_noise_)[train_mask].square().sum(1).mean()
                            z_compactness_loss_train = z_noise.square().sum(1).mean()

                            if args.s_consistency_scheduler == "ramp":
                                weight = min(1., epoch_partial / (max_train_epochs - 1))
                            elif args.s_consistency_scheduler == "cosine":
                                weight = cosine_ascend(0.5, 1, epoch_partial / (max_train_epochs - 1))
                            else:
                                weight = 1.

                            if log_stats:
                                summary.add_scalar(f"Loss/Consistency Weight", weight)
                                summary.add_scalar(f"Loss/Consistency", s_consistency_loss_train)
                                summary.add_scalar(f"Loss/Compactness", z_compactness_loss_train)
                            total_loss += s_consistency_loss_train * weight * args.s_consistency_weight
                            total_loss += z_compactness_loss_train * args.z_compactness_weight

                        if args.s_l1_weight > 0:
                            s_ = s[:, 1:] if do_roi else s
                            s_l1_loss = s_.abs().sum(1).mean()

                            total_loss += s_l1_loss * args.s_l1_weight
                            if log_stats:
                                summary.add_scalar(f"Loss/S L1", s_l1_loss)

                        if args.s_l2_weight > 0:
                            s_ = s[:, 1:] if do_roi else s
                            s_l2_loss = s_.square().sum(1).mean()

                            total_loss += s_l2_loss * args.s_l2_weight
                            if log_stats:
                                summary.add_scalar(f"Loss/S L2", s_l2_loss)

                        total_loss.backward()

                        decoder_lr = ((args.decoder_begin_lr - args.decoder_lr) *
                                      cosine_descend(50, 150, step) + args.decoder_lr)
                        rec.set_decoder_lr(decoder_lr)
                        if not final_train_epochs:
                            rec.decoder_opt.step(fsc_spectrum=fsc_spectrum)

                        _, data_ctf_spectra, avg_ctf2 = hvc.get_data_stats(0)

                        solvent_mask_applicator(data_ctf_spectra)

                        rec.clip_grad(args.grad_clip)
                        encoder_lr = args.encoder_final_lr if final_train_epochs else args.encoder_lr
                        rec.set_encoder_lr(encoder_lr)
                        rec.adam_opt.step()
                        
                        rec.eval()

                        if log_stats:
                            mse = torch.mean(square_error_train[:, spectral_mask])
                            summary.add_scalar(f"Loss/MSE", mse)
                            summary.add_scalar(f"Loss/MSE weighted", weighted_mse)
                            summary.add_scalar(f"Loss/Total", total_loss)
                            summary.add_scalar(f"Learning Rates/Encoders", encoder_lr)
                            summary.add_scalar(f"Learning Rates/Decoder", decoder_lr)

                        if not final_train_epochs:
                            with torch.no_grad():
                                reg_count = min(valid_batch_size * 2, this_batch_size - 1)
                                
                                train_mask_ = train_mask[:reg_count]
                                s_ = s[:reg_count].detach()
                                ctf_ = hv['ctfs_'][:reg_count, ..., None]
                                y_ = y_ft[:reg_count]

                                rot = hv["rot_matrices"][:reg_count]
                                shifts = hv["shifts_resid"][:reg_count]

                                x_ft = rec.decoder(s=s_, max_r=image_max_r, rot_matrices=rot)
                                x_ft_shift = fourier_shift_2d(x_ft, shifts)
                                x_ = x_ft_shift * ctf_

                            rec.stats.update(
                                x=x_, y=y_, ctf2=avg_ctf2,
                                train_mask=train_mask_,
                                valid_mask=~train_mask_,
                                mse=square_error_valid.mean(0),
                                momentum=0.99
                            )

                        if log_stats:
                            summary.write_stats(x_ft, y_ft, hv["amp"], hv["amp_ctf"])
                            summary.add_scalars(rec.stats.get_scalar_summary())

                        if log_images:
                            summary.write_images(x_ft, y_ft, hv['ctfs_'])

                            reg = rec.stats.get_spectral_summary()
                            for key in reg:
                                summary.add_figure(key, make_series_line_fig(reg[key]))

                            summary.add_figure("basis powers", make_series_line_fig(rec.decoder_opt.get_stats()))

                    rec.train_step += 1

                train_timing = time.time() - tt
                total_timing = time.time() - dt

                if prof is not None:
                    prof.step()

                if timing_shown_count == 0 and step > 0:
                    print("Loop has started...")
                    timing_shown_count += 1
                elif (step % 10001 == 0 or timing_shown_count < 10 and step % 10 == 0) and step > 0:
                    print(
                        "Step:", step,
                        "Timing:", round(total_timing, 3),
                        f"({round(invert_timing, 3)}, {round(train_timing, 3)})")
                    timing_shown_count += 1

                dt = time.time()

            hvc.finalize_first_epoch()
            summary.write_hidden_variable(hvc)

            if finalize:
                hvc.metadata_finalized = True
                if subtract_during_finalize:
                    subtraction_helper.finalize()

                print(f"Finished finalization")
                break
            else:
                rec.train_epoch += 1
                print(f"Epoch number {epoch + 1} complete at step {step}")

            if time.time() - last_save_time > args.checkpoint_time * 60:
                dac.save_to_logdir(log_dir)
                last_save_time = time.time()

    except StopIteration as e:
        print(e)
    except(KeyboardInterrupt, SystemExit):
        print("Exiting!")

    if z_relax_losses_count > 0:
        z_relax_losses = z_relax_losses / z_relax_losses_count
        summary.add_figure("Relax Loss", make_line_fig(np.arange(len(z_relax_losses)), z_relax_losses))

    dac.save_to_logdir(log_dir)

    if prof is not None:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    DistributedProcessing.process_cleanup()


def append_args(parser):
    append_train_arguments(parser)


def main(args):
    log_dir = args.log_dir

    # Remove log directory if overwriting
    if args.overwrite:
        if os.path.isdir(args.log_dir):
            tb_logdir = os.path.join(args.log_dir, "tb")
            if os.path.isdir(tb_logdir):
                shutil.rmtree(tb_logdir)

    if not os.path.isdir(log_dir):
        print("Creating log-directory:", log_dir)
        os.mkdir(log_dir)

    sys.stdout = IOLogger(os.path.join(log_dir, 'std.out'))

    for arg in sys.argv:
        print(arg, end=" ")
    print(end="\n\n")
        
    print(args, end="\n\n")
    print(f"Running pytorch version {torch.__version__}")

    DistributedProcessing.global_setup(
        args=args, main_fn=train, verbose=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Train a spectral heterogeneity analysis (SHA) 3D model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)

