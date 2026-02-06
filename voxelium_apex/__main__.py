#!/usr/bin/env python
"""
Voxelium-Apex: A cryo-EM data analysis framework
"""

import argparse


def main():
    import voxelium_apex as vxa

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"voxelium_apex {vxa.__version__}")

    modules = {
        "analysis_spectral": vxa.analysis.spectral,
        "analysis_star_subset": vxa.analysis.star_file_subset,
        "SHA3D": vxa.sha3d.train,
        "SHA3D_summary": vxa.sha3d.summary,
        "SHA3D_viewer": vxa.sha3d.viewer,
        "SHA3D_submit": vxa.sha3d.job_submit,
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
