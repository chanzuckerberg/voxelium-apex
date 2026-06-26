import time
from pathlib import Path
import pandas as pd

def tomo2py(
    workspace_dir: Path,
    particles_file: Path,
    tomograms_file: Path,
    compute_eulers: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert the particles file from RELION-5 tomo to single particle metadata.

    Parameters
    ----------
    workspace_dir : Path
        Path to the RELION-5 workspace directory.
    particles_file : Path
        Path to the RELION-5 particles file FROM AN EXTRACT JOB,
        relative to `workspace_dir`.
    tomograms_file : Path
        Path to the RELION-5 tomograms file relative to `workspace_dir`.
    compute_eulers : bool, optional
        If True, compute the Euler angles following the RELION zyz convention.
        If False, the rotation matrices are added to the DataFrame
        in the "rotMat" column.
        The default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the single particle metadata.
    """
    from scipy.spatial.transform import Rotation as R
    from tqdm import tqdm
    import numpy as np
    import starfile

    from voxelium_apex.importers.utils import (
        compute_defocus_offset,
        compute_particle_rot_mats,
        compute_tiltseries_proj_mats,
    )

    # Add the workspace directory to the path if it is not None
    if workspace_dir is not None:
        tomograms_file = workspace_dir / tomograms_file

    # Read the particles and tomograms star files
    particles_data = starfile.read(particles_file)
    tomograms_df = starfile.read(tomograms_file)
    particles_general = particles_data.get("general", {})
    assert (
        particles_general.get("rlnTomoSubTomosAre2DStacks", 0) == 1
    ), "This function assumes the extracted particles are 2D stacks!"

    optics_df = particles_data["optics"]
    particles_df = particles_data["particles"]

    originXYZAngst = particles_df[
        ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    ].to_numpy()

    assert np.all(
        originXYZAngst == 0
    ), "The particles must be centered (e.g. by running an Extract job)"

    particles_df["particleRotationMatrix"] = compute_particle_rot_mats(particles_df)

    # Renaming this since rlnImageName in a single particle stack
    # also contains the index of the image in the stack.
    particles_df.rename(columns={"rlnImageName": "rlnTomoImageName"}, inplace=True)

    # Add the optics group data from optics_df to particles_df
    particles_df = particles_df.merge(optics_df, how="left", on="rlnOpticsGroup")

    ts_df = compute_tiltseries_proj_mats(tomograms_df, workspace_dir, verbose=verbose)

    if verbose:
        print(
            (
                f"Generating single particle metadata for {len(particles_df)} "
                "2D tomo particle stacks..."
            ),
            end=" ",
            flush=True,
        )
        t0 = time.time()

    single_particles = []
    particles_df_group = particles_df.groupby("rlnTomoName")
    for tomo_name, particles_tomo_df in tqdm(particles_df_group, desc="Converting particles to single particle metadata"):
        ts_tomo_df = ts_df[tomo_name]

        for _, p_row in particles_tomo_df.iterrows():
            visible_frames_str = p_row["rlnTomoVisibleFrames"][1:-1].split(",")
            visible_frames = [int(vf) == 1 for vf in visible_frames_str]
            ts_particle_df = ts_tomo_df[visible_frames]

            particle_coords = p_row[
                [
                    "rlnCenteredCoordinateXAngst",
                    "rlnCenteredCoordinateYAngst",
                    "rlnCenteredCoordinateZAngst",
                ]
            ].to_numpy()

            new_particles = []
            for i_tilt, (_, ts_row) in enumerate(ts_particle_df.iterrows()):
                tilt_rot_mat = ts_row["tiltImageRotationMatrix"]
                rot_mat_particle = p_row["particleRotationMatrix"] @ tilt_rot_mat

                tilt_proj_mat = ts_row["tiltImageProjectionMatrix"]
                handedness = ts_row["rlnTomoHand"]
                defocus_slope = ts_row.get("rlnTomoDefocusSlope", 1.0)

                # TODO: NEED TO ADD THE CTF SCALE FACTOR FROM TS_ROW TO THE NEW_ROW.
                # DONE, BUT IS THERE ANYTHING ELSE THAT I MISSED? CHECK!!

                defocus_dz = compute_defocus_offset(
                    tilt_proj_mat, particle_coords, handedness, defocus_slope
                )

                new_row = {
                    "rlnTomoName": tomo_name,
                    "rlnImageName": f"{i_tilt+1}@{p_row['rlnTomoImageName']}",
                    "rlnTomoParticleName": p_row["rlnTomoParticleName"],
                    # NOTE: Particles are centered (if coming from an Extract job)
                    "rlnOriginXAngst": 0.0,
                    "rlnOriginYAngst": 0.0,
                    "rlnNormCorrection": p_row["rlnNormCorrection"],
                    "rlnGroupNumber": p_row["rlnGroupNumber"],
                    "rlnRandomSubset": p_row["rlnRandomSubset"],
                    "rlnOpticsGroup": p_row["rlnOpticsGroup"],
                    "rlnTomoNominalStageTiltAngle": ts_row[
                        "rlnTomoNominalStageTiltAngle"
                    ],
                    "rlnTomoNominalTiltAxisAngle": ts_row[
                        "rlnTomoNominalTiltAxisAngle"
                    ],
                    "rlnMicrographPreExposure": ts_row["rlnMicrographPreExposure"],
                    "rlnTomoNominalDefocus": ts_row["rlnTomoNominalDefocus"],
                    "rlnDefocusU": ts_row["rlnDefocusU"] + defocus_dz,
                    "rlnDefocusV": ts_row["rlnDefocusV"] + defocus_dz,
                    # "rlnCtfAstigmatism": ts_row["rlnCtfAstigmatism"],
                    "rlnDefocusAngle": ts_row["rlnDefocusAngle"],
                    "rlnCtfFigureOfMerit": ts_row["rlnCtfFigureOfMerit"],
                    "rlnCtfMaxResolution": ts_row["rlnCtfMaxResolution"],
                    "rlnCtfScalefactor": ts_row["rlnCtfScalefactor"],
                    "rlnVoltage": p_row["rlnVoltage"],
                    "rlnSphericalAberration": p_row["rlnSphericalAberration"],
                    "rlnAmplitudeContrast": p_row["rlnAmplitudeContrast"],
                    "rlnTomoTiltSeriesPixelSize": p_row["rlnTomoTiltSeriesPixelSize"],
                    "rlnCtfDataAreCtfPremultiplied": p_row[
                        "rlnCtfDataAreCtfPremultiplied"
                    ],
                    "rlnImageDimensionality": p_row["rlnImageDimensionality"],
                    "rlnTomoSubtomogramBinning": p_row["rlnTomoSubtomogramBinning"],
                    "rlnImagePixelSize": p_row["rlnImagePixelSize"],
                    "rlnImageSize": p_row["rlnImageSize"],
                }

                if compute_eulers:
                    eulers = R.from_matrix(rot_mat_particle).as_euler(
                        "ZYZ", degrees=True
                    )

                    new_row["rlnAngleRot"] = eulers[0]
                    new_row["rlnAngleTilt"] = eulers[1]
                    new_row["rlnAnglePsi"] = eulers[2]
                else:
                    new_row["rotationMatrix"] = rot_mat_particle

                new_particles.append(new_row)
            single_particles += new_particles
    single_particles_df = pd.DataFrame(single_particles)

    if verbose:
        print(f"done in {time.time() - t0:.2f} seconds.", flush=True)
        print(f"Total number of particles: {len(single_particles_df)}")

    return single_particles_df
