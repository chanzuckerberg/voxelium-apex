from positron.base.args import range_limited_float_type, range_limited_int_type


def append_train_arguments(parser):
    parser.add_argument('input', help='input job (job directory or optimizer-file)', type=str)
    parser.add_argument('log_dir', type=str, metavar='log_dir', help='path to load a model')
    parser.add_argument('--particle_diameter', help='size of circular mask (ang)', type=int, default=None)
    parser.add_argument('--circular_mask_thickness', help='thickness of mask (ang)', type=int, default=20)
    parser.add_argument('--batch_size', help='mini-batch size from training dataset', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gpu', dest='gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--checkpoint_time', help='Minimum time in minutes between checkpoint saves', type=int,
                        default=10)
    parser.add_argument("--image_steps", type=int, default=500, help="Log tensorboard images every n steps")
    parser.add_argument("--stats_steps", type=int, default=100, help="Log tensorboard statistics every n steps")
    parser.add_argument('--max_steps', dest='max_steps', type=int, default=int(1e9), help='number of steps to train')
    parser.add_argument('--max_train_epochs', type=int, default=int(10), help='number of epochs to train')
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--dont_finalize', action='store_true')
    parser.add_argument('--only_finalize', action='store_true')
    parser.add_argument('--pytorch_threads', type=int, default=6)
    parser.add_argument('--dataloader_threads', type=int, default=4)
    parser.add_argument('--tomo', action='store_true')

    parser.add_argument(
        '--z_size',
        help='Number of learnt representation dimensions.',
        type=int, default=2
    )
    parser.add_argument(
        '--s_size',
        help='Number of structure basis.',
        type=int, default=8
    )

    parser.add_argument(
        '--do_align',
        help='Do optimize pose and translation',
        action="store_true"
    )
    parser.add_argument(
        '--do_ctf_optimization',
        help='Do optimize CTF defocuse and angle',
        action="store_true"
    )
    parser.add_argument(
        '--solvent_mask',
        help='MRC file with ones in the region that is not solvent (region of interest)',
        type=str, default=None
    )
    parser.add_argument(
        '--solvent_smoothing',
        help='Smoothing kernel size (in Ångströms) of the solvent region (zero=no smoothing).',
        type=range_limited_float_type(0, 10), default=None
    )
    parser.add_argument(
        '--subtract_mask',
        help='If a mask is provided, create a new particle stack where everything outside the mask is subtracted.',
        type=str, default=None
    )
    parser.add_argument(
        '--roi_mask',
        help='If a mask is provided, allow only structural heterogeneity inside the masked region.',
        type=str, default=None
    )
    parser.add_argument(
        '--subtract_buffer_size',
        help='Maximum buffer size for subtracted data in MiB.',
        type=int, default=500
    )
    parser.add_argument('--profile_runtime', action='store_true')

    parser.add_argument(
        '--lr',
        help='Learning rate of the structure decoder',
        type=range_limited_float_type(0), default=0.001
    )
    parser.add_argument(
        '--begin_lr',
        help='Starting learning rate of the structure decoder',
        type=range_limited_float_type(0), default=0.1
    )

    parser.add_argument(
        '--lr_encoder',
        help='Learning rate of the encoder',
        type=range_limited_float_type(0), default=0.0004
    )

    parser.add_argument(
        '--wd_encoder',
        help='Weight decay of the encoder',
        type=range_limited_float_type(0), default=1e-2
    )

    parser.add_argument(
        '--grad_clip',
        help='Gradient clipping of the encoder',
        type=range_limited_float_type(0), default=1e-2
    )

    parser.add_argument(
        '--s_l1_weight', '--sl1',
        help='The weight for the L1 norm loss for S',
        type=range_limited_float_type(0), default=0.
    )

    parser.add_argument(
        '--s_l2_weight', '--sl2',
        help='The weight for the L2 norm loss for S',
        type=range_limited_float_type(0), default=0.0001
    )

    parser.add_argument(
        '--consistency_weight', '--cw',
        help='Self consistency of the S vector',
        type=range_limited_float_type(0), default=1.
    )

    parser.add_argument(
        '--encoder_noise',
        help='Contrastive learning model layer augmentation noise',
        type=range_limited_float_type(0), default=0.
    )

    parser.add_argument(
        '--decoder_noise',
        help='Contrastive learning model layer augmentation noise',
        type=range_limited_float_type(0), default=0.
    )

    parser.add_argument(
        '--feature_noise',
        help='Contrastive learning feature augmentation noise',
        type=range_limited_float_type(0), default=0.01
    )

    parser.add_argument(
        '--feature_bandpass',
        help='Feature extraction band filters (in Ångströms). Comma separated.',
        type=str, default=None
    )

    parser.add_argument(
        '--z_encoder_dims',
        help='Comma separated integers used for Z-encoder hidden layer dimensions.',
        type=str, default="128,128,128"
    )

    parser.add_argument(
        '--s_encoder_dims',
        help='Comma separated integers used for S-encoder hidden layer dimensions.',
        type=str, default="128,128,128,128"
    )

    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float32", 
        help="Data type used for storing images in data set"
    )
    parser.add_argument('--dont_postprocess', action='store_true')
    parser.add_argument('--only_update_representation', action='store_true')
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Fraction of dataset to be used for validation."
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache directory"
    )
    parser.add_argument(
        "--filter_resolution",
        type=float,
        default=20,
        help="Filter resolution cut-off, in Ångströms"
    )
    parser.add_argument(
        "--max_data_resolution",
        type=float,
        default=None,
        help="Minimum data resolution, in Ångströms"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.,
        help="Regularization parameter"
    )
    parser.add_argument(
        "--lam_base",
        type=float,
        default=1.,
        help="Regularization parameter"
    )
    parser.add_argument(
        "--dampen",
        type=float,
        default=1.,
        help="Regularization parameter"
    )
