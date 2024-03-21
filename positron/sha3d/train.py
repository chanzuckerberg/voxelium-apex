#!/usr/bin/env python3

"""
Module for training a 3D spectral heterogeneity analysis (SHA) model
"""
import argparse
import random
import time

from datetime import datetime

from torch.utils.data import DataLoader

from positron.sha3d.feature_extractor import FeatureExtractor
from positron.sha3d.loss_functions import batch_triplet_loss, cosine_similarity_loss, bsc_loss, tsne_loss, \
    cosine_similarity_loss2, similarity_loss
from positron.sha3d.mask_applicator import MaskApplicator
from positron.sha3d.regularizer import Regularizer
from positron.base.single_particle_validation_sampler import SingleParticleValidationSampler
from positron.base.subtomo_validation_sampler import SubtomoValidationSampler
from positron.sha3d.subtraction import SubtractionHelper
from positron.sha3d.train_arguments import append_train_arguments

from positron.base.torch_utils import make_series_line_fig
from positron.sha3d.distributed_processing import DistributedProcessing
from positron.sha3d.tensorboard_utils import TensorboardSummary
from positron.sha3d.train_utils import *
from positron.base import load_mrc
from positron.base.spectral import fourier_shift_2d, spectral_index_from_resolution
from positron.base.io_logger import IOLogger
from positron.sha3d.data_analysis_container import DatasetAnalysisContainer
from positron.sha3d.retention_classifier import RetentionClassifier


def get_lr(step, args):
    return (args.begin_lr - args.lr) * cosine_descend(50, 150, step) + args.lr


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
            group_indices=dataset.part_group_idx,
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

    print("\nINITIALIZING TRAINING", datetime.now().ctime())

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

    roi_mask = None
    if args.roi_mask is not None:
        roi_mask, _, _ = load_mrc(args.roi_mask)
        roi_mask = torch.Tensor(roi_mask.copy()).to(device)

        if not torch.any((0. < roi_mask) & (roi_mask < 1.)):
            print("\nWARNING: ROI mask should only contain values in the range (including) zero and one.\n")

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

    regularizer = Regularizer(
        image_size=image_size,
        filter_cutoff_idx=spectral_index_from_resolution(args.filter_resolution, image_size, pixel_size),
        lam=args.lam
    ).to(device)

    if args.only_finalize:
        max_epochs = rec.train_epoch + 1
    else:
        max_epochs = args.max_train_epochs if args.dont_finalize else args.max_train_epochs + 1

    # TODO remove voxel_size input
    feature_extractor = FeatureExtractor(
        decoder=rec.decoder,
        voxel_size=pixel_size,
        bandpass=rec.feature_bandpass,
        image_max_r=image_max_r,
        do_roi=roi_mask is not None
    )

    solvent_smoothing = args.solvent_smoothing / pixel_size if args.solvent_smoothing is not None else 1
    solvent_mask_applicator = MaskApplicator(
        rec.decoder.projector,
        solvent_mask=solvent_mask,
        roi_mask=roi_mask,
        solvent_smoothing=solvent_smoothing
    )

    subtract_during_finalize = subtract_mask is not None

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

    do_tomo = args.tomo

    s_retention = RetentionClassifier(rec.s_size).to(device)

    try:
        for epoch in np.arange(rec.train_epoch, max_epochs):
            dt = time.time()
            sampler.train()

            finalize = False
            if not finalize and (args.only_finalize or epoch == args.max_train_epochs):
                print("Finalizing...")
                finalize = True
                sampler.eval()
                torch.no_grad()

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

                particle_groups = None
                if do_tomo:
                    _, particle_groups = torch.unique(sample["group_idx"], return_inverse=True)
                    particle_groups = particle_groups.to(device)

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
                y_weight = regularizer.get_y_weight()
                x_signal_pow = regularizer.get_x_signal()
                x_noise_pow = regularizer.get_x_noise()
                snr = y_weight * x_signal_pow

                s0 = None
                if epoch >= 2:
                    s0 = hvc.get_metadata('s', particle_idx)[:, 0].to(device)

                features = feature_extractor(
                    hv=hv, y=y_ft, wy=snr, wx=x_noise_pow, accumulate_stats=not finalize, s0=s0, groups=particle_groups)

                if log_stats:
                    summary.add_scalar("Features/mean", features.mean())
                    summary.add_scalar("Features/std", features.std())

                features = rec.normalize_features(features)

                invert_timing = time.time() - tt

                if do_tomo:
                    f = features[particle_groups]
                else:
                    f = features
                hvc.set_metadata('feature', particle_idx, f)

                features_ = features if finalize else features + torch.randn_like(features) * args.feature_noise
                z, _ = rec.encode(features_, noise=0 if finalize else args.encoder_noise)
                s = rec.s_encoder(z, noise=0 if finalize else args.decoder_noise)
                # z, s, mu, log_var = rec.vae(
                #     features_,
                #     encoder_noise=0 if finalize else args.encoder_noise,
                #     decoder_noise=0 if finalize else args.decoder_noise,
                #     reparam=args.kl_weight > 0
                # )

                if do_tomo:
                    z = z[particle_groups]
                    s = s[particle_groups]
                    # mu = mu[particle_groups]
                    # log_var = log_var[particle_groups]

                # hvc.set_metadata('log_var', particle_idx, log_var)
                # hvc.set_metadata('z', particle_idx, mu)
                hvc.set_metadata('z', particle_idx, z)
                hvc.set_metadata('s', particle_idx, s)

                if finalize and subtract_during_finalize:
                    subtraction_helper(s, sample, hv)

                if not finalize:
                    feature_extractor.track_s0(s[:, 0])

                    s_retention(logit=s, labels=train_mask, make_summary=log_stats)
                    if log_stats:
                        summary.add_scalars(s_retention.get_summary())

                    if log_stats:
                        summary.add_scalar(f"Z/std", z.std())
                        summary.add_scalar(f"Z/mean", z.mean())
                        summary.add_scalar(f"S/std", s.std())
                        summary.add_scalar(f"S/mean", s.mean())
                        summary.add_scalar(f"S/S norm mean", s.square().sum(1).mean())
                        summary.add_scalar(f"S/S norm std", s.square().sum(1).std())
                        summary.add_scalar(f"S0/std", s[:, 0].std())
                        summary.add_scalar(f"S0/mean", s[:, 0].mean())

                    tt = time.time()

                    x_ft = rec.decoder(s=s, max_r=image_max_r, rot_matrices=hv["rot_matrices"])

                    x_ft_shift = fourier_shift_2d(x_ft, hv["shifts_resid"])
                    x = x_ft_shift * hv['ctfs_'][..., None]

                    spectral_mask = Cache.get_spectral_mask(
                        x_ft.shape[1:-1],
                        max_r=image_max_r,
                        device=device
                    )

                    y_weight = regularizer.get_y_weight(eps=1e-3)
                    weight = y_weight / (y_weight.mean() + 1e-3)
                    weight_grid = Cache.spectra_to_grids(weight, hv['ctfs_'].shape[1:], image_max_r)
                    x_ = torch.view_as_complex(x)
                    y_ft_ = torch.view_as_complex(y_ft)
                    square_error_train = (x_[train_mask] - y_ft_[train_mask].detach()).abs().square()
                    square_error_train_w = square_error_train * weight_grid[None]
                    weighted_mse = (
                            square_error_train_w[:, spectral_mask].mean(0).sum() /
                            (weight_grid[spectral_mask].sum() + 1e-12)
                    )
                    if step > 0:
                        total_loss = weighted_mse

                        fsc_spectrum = regularizer.get_fsc_spectrum().clip(0.01, 0.99)

                        regularization_weight = Cache.spectra_to_grids(
                            1 - fsc_spectrum, hv['ctfs_'].shape[1:], image_max_r)
                        regularization_loss = torch.mean(x_ft.square().sum(-1) * regularization_weight[None])
                        if log_stats:
                            summary.add_scalar(f"Loss/Regularization", regularization_loss)
                        total_loss += regularization_loss

                        # if args.s_l1_weight > 0:
                        #     s_norm_loss = s.abs().sum(1).mean()
                        #     total_loss += s_norm_loss * args.s_l1_weight
                        #     if log_stats:
                        #         summary.add_scalar(f"Loss/S L1", s_norm_loss)

                        if args.s_l2_weight > 0:
                            s_norm_loss = s.square().sum(1).mean()
                            total_loss += s_norm_loss * args.s_l2_weight
                            if log_stats:
                                summary.add_scalar(f"Loss/S L2", s_norm_loss)

                        # if args.z_contrastive_weight > 0 or args.s_contrastive_weight > 0:
                        #     features_ = features + torch.randn_like(features) * args.feature_noise
                        #     _, s_aug, mu_aug, _ = rec.vae(
                        #         features_,
                        #         encoder_noise=args.encoder_noise,
                        #         decoder_noise=args.decoder_noise,
                        #         reparam=args.kl_weight > 0
                        #     )
                        #     if do_tomo:
                        #         mu_aug = mu_aug[particle_groups]
                        #         s_aug = s_aug[particle_groups]

                        # if args.z_contrastive_weight > 0:
                        #     z_contrastive_loss = batch_triplet_loss(
                        #         anchor=mu[train_mask],
                        #         target=mu_aug[train_mask],
                        #         margin=args.z_contrastive_margin
                        #     )
                        #     total_loss += z_contrastive_loss * args.z_contrastive_weight
                        #     if log_stats:
                        #         summary.add_scalar(f"Loss/Z contrastive", z_contrastive_loss)

                        # if args.s_contrastive_weight > 0:
                        #     s_contrastive_loss = batch_triplet_loss(
                        #         anchor=s[train_mask],
                        #         target=s_aug[train_mask],
                        #         margin=args.s_contrastive_margin
                        #     )
                        #     total_loss += s_contrastive_loss * args.s_contrastive_weight
                        #     if log_stats:
                        #         summary.add_scalar(f"Loss/S contrastive", s_contrastive_loss)

                        # if args.kl_weight > 0:
                        #     kld_loss = torch.mean(
                        #         -0.5 * torch.sum(
                        #             1 + log_var[train_mask] - mu[train_mask] ** 2 - log_var[train_mask].exp(),
                        #             dim=1),
                        #         dim=0)
                        #     total_loss += kld_loss * args.kl_weight
                        #     if log_stats:
                        #         summary.add_scalar(f"Loss/KL Divergence", kld_loss)

                        features_ = features + torch.randn_like(features) * args.feature_noise
                        z_, _ = rec.encode(features_, noise=args.encoder_noise)
                        s_ = rec.s_encoder(z_, noise=args.decoder_noise)

                        consistency_loss = similarity_loss(s, s_)
                        if log_stats:
                            summary.add_scalar(f"Loss/Consistency", consistency_loss)
                        total_loss += consistency_loss * args.consistency_weight

                        total_loss.backward()

                        if log_stats:
                            mse = torch.mean(square_error_train[:, spectral_mask])
                            summary.add_scalar(f"Loss/MSE", mse)
                            summary.add_scalar(f"Loss/MSE weighted", weighted_mse)
                            summary.add_scalar(f"Loss/Total", total_loss)

                        reg_count = min(valid_batch_size * 2, this_batch_size - 1)

                        lr = get_lr(step, args)

                        rec.decoder_opt.set_lr(lr)
                        rec.decoder_opt.step(fsc_spectrum=fsc_spectrum)

                        _, data_ctf_spectra, avg_ctf2 = hvc.get_data_stats(0)

                        solvent_mask_applicator(data_ctf_spectra)

                        rec.clip_grad(args.grad_clip)
                        rec.adam_opt.step()

                        rec.zero_grad()
                        rec.eval()

                        with torch.no_grad():
                            train_mask_ = train_mask[:reg_count]
                            s_ = s[:reg_count].detach()
                            ctf_ = hv['ctfs_'][:reg_count, ..., None]
                            y_ = y_ft[:reg_count]

                            rot = hv["rot_matrices"][:reg_count]
                            shifts = hv["shifts_resid"][:reg_count]

                            x_ft = rec.decoder(s=s_, max_r=image_max_r, rot_matrices=rot)
                            x_ft_shift = fourier_shift_2d(x_ft, shifts)
                            x_ = x_ft_shift * ctf_

                        regularizer.update(
                            x=x_, y=y_, ctf2=avg_ctf2,
                            train_mask=train_mask_,
                            valid_mask=~train_mask_,
                            momentum=0.99
                        )

                        if log_stats:
                            summary.write_stats(x_ft, y_ft, hv["amp"], hv["amp_ctf"])

                        if log_images:
                            summary.write_images(x_ft, y_ft, hv['ctfs_'])

                            reg = regularizer.get_spectral_summary()
                            for key in reg:
                                summary.add_figure(key, make_series_line_fig(reg[key]))

                            summary.add_figure("basis powers", make_series_line_fig(rec.decoder_opt.get_stats()))

                    rec.train_step += 1

                train_timing = time.time() - tt
                total_timing = time.time() - dt

                if prof is not None:
                    prof.step()

                if timing_shown_count == 0 and step > 0:
                    print("Training has started...")
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

    if time.time() - last_save_time > 2:  # Don't save it just made a saved (less than 2 secs ago)
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

    print(args)
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

