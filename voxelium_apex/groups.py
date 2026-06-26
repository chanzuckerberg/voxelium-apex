import rich_click as click

click.rich_click.OPTION_GROUPS = {
    "vxm-apex sha3d train": [
        {
            "name": "I/O",
            "options": [
                "--input",
                "--overwrite",
                "--cache",
                "--dtype",
                "--tomo",
            ],
        },
        {
            "name": "Training",
            "options": [
                "--batch_size",
                "--max_steps",
                "--max_train_epochs",
                "--checkpoint_time",
                "--validation-fraction",
                "--preload",
                "--dont_finalize",
                "--only_finalize",
                "--only_update_representation",
                "--dont_postprocess",
            ],
        },
        {
            "name": "Hardware",
            "options": [
                "--gpu",
                "--pytorch_threads",
                "--dataloader_threads",
                "--profile_runtime",
            ],
        },
        {
            "name": "Model architecture",
            "options": [
                "--z_size",
                "--s_size",
                "--z_encoder_dims",
                "--s_encoder_dims",
            ],
        },
        {
            "name": "Masking",
            "options": [
                "--particle_diameter",
                "--circular_mask_thickness",
                "--solvent_mask",
                "--roi_mask",
                "--subtract_mask",
                "--subtract_buffer_size",
                "--max_data_resolution",
            ],
        },
        {
            "name": "Pose & CTF",
            "options": [
                "--do_align",
                "--do_ctf_optimization",
            ],
        },
        {
            "name": "Learning rates",
            "options": [
                "--encoder_lr",
                "--encoder_final_lr",
                "--decoder_begin_lr",
                "--decoder_lr",
                "--relax_lr",
                "--relax_iter",
                "--grad_clip",
            ],
        },
        {
            "name": "Loss & regularization",
            "options": [
                "--regularization",
                "--dampen",
                "--s_consistency_weight",
                "--s_consistency_scheduler",
                "--smoothness_distance",
                "--smoothness_distance_min",
                "--s_l1_weight",
                "--s_l2_weight",
                "--z_compactness_weight",
            ],
        },
        {
            "name": "Feature extraction",
            "options": [
                "--feature_bandpass",
                "--mse_bandpass",
                "--feature_noise_weight",
            ],
        },
        {
            "name": "Logging",
            "options": [
                "--image_steps",
                "--stats_steps",
            ],
        },
    ],
    "vxm-apex extract": [
        {
            "name": "Relion Inputs",
            "options": [
                "--particles",
                "--tomograms",
                "--motion",
                "--workspace",
            ],
        },
        {
            "name": "Parameters",
            "options": [
                "--box-size",
                "--tiltseries-relative-dir",
                "--crop-size",
                "--bin",
                "--debug",
            ],
        },
        {
            "name": "Copick Inputs",
            "options": [
                "--picks-uri",
                "--config",
                "--runs",
            ],
        },
    ],
}