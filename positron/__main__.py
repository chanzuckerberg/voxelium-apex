#!/usr/bin/env python

"""
Positron - Cryo-EM data analysis framework
"""


def main():
    import argparse
    import positron
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version=f'positron {positron.__version__}')

    import positron.analysis.spectral
    import positron.analysis.star_file_subset

    import positron.sha3d.train
    import positron.sha3d.summary
    import positron.sha3d.viewer
    import positron.sha3d.job_submit

    modules = {
        "analysis_spectral": positron.analysis.spectral,
        "analysis_star_subset": positron.analysis.star_file_subset,
        "SHA3D": positron.sha3d.train,
        "SHA3D_summary": positron.sha3d.summary,
        "SHA3D_viewer": positron.sha3d.viewer,
        "SHA3D_submit": positron.sha3d.job_submit,
    }

    subparsers = parser.add_subparsers(title='Choose a module')
    subparsers.required = 'True'

    for key in modules:
        module_parser = subparsers.add_parser(key, description=modules[key].__doc__)
        modules[key].append_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()
        raise TypeError  # Let's keep this here for now


if __name__ == '__main__':
    main()