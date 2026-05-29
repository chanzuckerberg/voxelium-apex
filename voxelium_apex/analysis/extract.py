#!/usr/bin/env python

"""
Subtomogram extraction.

Thin wrapper around ``zarr_particle_tools`` that extracts subtomograms from
either a RELION ``particles.star`` file or directly from copick picks (via a
picks URI + copick config). Both paths share the same tomograms/tiltseries
inputs and extraction geometry (box size, crop size, binning).

The following extraction policy is fixed and not exposed as options:

    * CTF premultiplication is disabled (``no_ctf=True``)
    * output mrcs are written as float16 (``float16=True``)
    * existing output is overwritten (``overwrite=True``)
    * Fourier-space stacks are not written (``write_fourier=False``)
    * circular cropping is kept enabled (``no_circle_crop=False``)

Heavy imports (``zarr_particle_tools``, ``copick``) are performed inside
``main`` so that the command can be listed/helped without importing them.
"""


# Fixed extraction policy shared by both extraction paths.
_FIXED_POLICY = dict(
    no_ctf=True,
    float16=True,
    overwrite=True,
    write_fourier=False,
    no_circle_crop=False,
)


def parse_picks_uri(uri):
    """Parse a copick picks URI ``object_name:user_id/session_id``.

    Follows the copick URI grammar: missing components and ``*`` wildcards
    become ``None`` (i.e. "match any"), which is what
    ``zarr_particle_tools`` expects when querying copick picks.

    Returns:
        tuple: ``(object_name, user_id, session_id)`` where each may be ``None``.
    """
    uri = uri.strip()
    if uri.startswith("re:"):
        raise ValueError("Regex picks URIs ('re:' prefix) are not supported by extract.")

    if ":" in uri:
        object_name, rest = uri.split(":", 1)
    else:
        object_name, rest = uri, ""

    user_id = session_id = ""
    if rest:
        if "/" in rest:
            user_id, session_id = rest.split("/", 1)
        else:
            user_id = rest

    def _norm(value):
        return None if value in ("", "*") else value

    return _norm(object_name), _norm(user_id), _norm(session_id)


def _split_csv(value):
    """Split a comma/space separated string into a list, or ``None`` if empty."""
    if not value:
        return None
    items = [item.strip() for item in value.replace(",", " ").split()]
    return items or None


def main(args):
    from pathlib import Path

    from zarr_particle_tools.core.helpers import setup_logging
    from zarr_particle_tools.subtomo_extract import (
        parse_extract_local_copick_subtomograms,
        parse_extract_local_subtomograms,
    )

    setup_logging(debug=getattr(args, "debug", False))

    def _path(value):
        return Path(value) if value is not None else None

    common = dict(
        box_size=args.box_size,
        output_dir=_path(args.output),
        crop_size=args.crop_size,
        bin=args.bin,
        tiltseries_relative_dir=_path(args.tiltseries_relative_dir),
        tomograms_starfile=_path(args.tomograms),
        **_FIXED_POLICY,
    )

    if args.particles:
        parse_extract_local_subtomograms(
            particles_starfile=_path(args.particles),
            debug=getattr(args, "debug", False),
            **common,
        )
    else:
        object_name, user_id, session_id = parse_picks_uri(args.picks_uri)
        parse_extract_local_copick_subtomograms(
            copick_config=_path(args.config),
            copick_name=object_name,
            copick_user_id=user_id,
            copick_session_id=session_id,
            copick_run_names=_split_csv(getattr(args, "runs", None)),
            **common,
        )
