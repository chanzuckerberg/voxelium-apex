#!/usr/bin/env python
"""
Voxelium-Apex: A cryo-EM data analysis framework
"""

import argparse


def main():
    import voxelium_apex
    from voxelium_apex.analysis import spectral, star_file_subset
    from voxelium_apex.sha3d import train, summary, viewer, job_submit 

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"voxelium_apex {voxelium_apex.__version__}")

    modules = {
        "analysis_spectral": spectral,
        "analysis_star_subset": star_file_subset,
        "SHA3D": train,
        "SHA3D_summary": summary,
        "SHA3D_viewer": viewer,
        "SHA3D_submit": job_submit,
    }

    subparsers = parser.add_subparsers(
        title="Choose a module",
        dest="command",
        required=True,
    )

    for name, mod in modules.items():
        module_parser = subparsers.add_parser(name, description=getattr(mod, "__doc__", None))
        mod.append_args(module_parser)
        module_parser.set_defaults(func=mod.main)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
