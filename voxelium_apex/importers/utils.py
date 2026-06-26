import time

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as R


def compute_particle_rot_mats(particles_df: pd.DataFrame) -> list[np.ndarray]:
    """Compute the rotation matrices for the *tomo* particles in a tilt series.

    Parameters
    ----------
    particles_df : pd.DataFrame
        The particles table from the particles STAR file from an Extract job.

    Returns
    -------
    list[np.ndarray]
        A list of rotation matrices for each *tomography* particle.
    """
    n_particles = len(particles_df)
    angles = particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy()
    rot_mats = R.from_euler("ZYZ", angles, degrees=True).as_matrix()

    if "rlnTomoSubtomogramRot" in particles_df.columns:
        subtomo_angles = particles_df[
            ["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"]
        ].to_numpy()
        subtomo_rot_mats = R.from_euler("ZYZ", subtomo_angles, degrees=True).as_matrix()
    else:
        subtomo_rot_mats = np.repeat(np.eye(3)[None, ...], n_particles, axis=0)

    for i in range(n_particles):
        rot_mats[i] = rot_mats[i] @ subtomo_rot_mats[i]

    return list(rot_mats)


def compute_tiltseries_proj_mats(tomograms_df, workspace_dir, verbose=False) -> dict:
    """Compute the projection matrices for the tilt images in all tilt series.

    Parameters
    ----------
    tomograms_df : pd.DataFrame
        The data in the tomograms STAR file.
    workspace_dir : Path
        The path to the RELION-5 workspace directory.
    verbose : bool, optional
        If True, print progress messages. The default is False.

    Returns
    -------
    dict
        A dictionary with the tilt series names as keys and the metadata for each
        tilt series including the projection matrices in the tiltImageProjectionMatrix
        column.
    """
    if verbose:
        print(
            f"Computing projection matrices for {len(tomograms_df)} tilt series...",
            end=" ",
            flush=True,
        )
        t0 = time.time()

    ts_df = {}
    tomograms_df.set_index("rlnTomoName", inplace=True)
    for tomo_name, tomo_row in tomograms_df.iterrows():

        if workspace_dir is None:
            tilt_series_file = tomo_row["rlnTomoTiltSeriesStarFile"]
        else:
            tilt_series_file = workspace_dir / tomo_row["rlnTomoTiltSeriesStarFile"]
        # tilt_series_file = workspace_dir / tomo_row["rlnTomoTiltSeriesStarFile"]
        ts_tomo_df = starfile.read(tilt_series_file)

        if "rlnTomoXTilt" not in ts_tomo_df.columns:
            ts_tomo_df["rlnTomoXTilt"] = 0.0

        # Add all columns of tomo_row to all rows of ts_tomo_df
        for col in tomograms_df.columns:
            if col not in ts_tomo_df.columns:
                ts_tomo_df[col] = tomograms_df[col].loc[tomo_name]

        rot_and_proj_mats = ts_tomo_df.apply(compute_tiltimage_proj_mat, axis=1)
        tilt_rot_mats, tilt_proj_mats = zip(*rot_and_proj_mats)

        ts_tomo_df["tiltImageRotationMatrix"] = tilt_rot_mats
        ts_tomo_df["tiltImageProjectionMatrix"] = tilt_proj_mats

        ts_df[tomo_name] = ts_tomo_df

    if verbose:
        print(f"done in {time.time() - t0:.2f} seconds.", flush=True)

    return ts_df


def compute_tiltimage_proj_mat(ts_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Compute the rotation matrix for a tilt image in a tilt series.

    Parameters
    ----------
    ts_row : pd.Series
        A row in a tilt series STAR file, containing the metadata for a tilt image.

    Returns
    -------
    np.ndarray
        The projection matrix for the tilt image.
    """
    # NOTE: working with Angstrom here since we only need the projection
    # matrices to compute the defocus offset. For the same reason, we
    # don't apply the shifts to/from to the centre of the tomgram or tilt imgages

    # These are extrinsic Euler angles (see paper)
    x_tilt, y_tilt, z_rot = ts_row[["rlnTomoXTilt", "rlnTomoYTilt", "rlnTomoZRot"]]
    Rx = R.from_euler("x", x_tilt, degrees=True)
    Ry = R.from_euler("y", y_tilt, degrees=True)
    Rz = R.from_euler("z", z_rot, degrees=True)
    Rzyx = Rz * Ry * Rx

    rot_mat = np.eye(4)
    rot_mat[:3, :3] = Rzyx.inv().as_matrix()

    shifts = ts_row[["rlnTomoXShiftAngst", "rlnTomoYShiftAngst"]].to_numpy()
    shift_mat = np.eye(4)
    shift_mat[:2, 3] = shifts

    rot_mat_inv = np.eye(4)
    rot_mat_inv[:3, :3] = Rzyx.as_matrix()
    proj_mat = shift_mat @ rot_mat_inv

    return rot_mat[:3, :3], proj_mat


def compute_defocus_offset(
    tilt_proj_mat: np.ndarray,
    particle_coords: np.ndarray,
    handedness: int,
    defocus_slope: float = 1.0,
) -> float:
    """Compute the defocus offset for a particle in a tilt image.

    Parameters
    ----------
    tilt_proj_mat : np.ndarray, shape (4, 4)
        The projection matrix for the tilt image.
    particle_coords : np.ndarray, shape (3,)
        The coordinates of the particle in centred Angstroms
        rlnCenteredCoordinate<X/Y/Z>Angst columns in the particles STAR file.
    handedness : int
        The handedness of the tilt series (1 or -1)
        rlnTomoHand in the tomograms.star file.
    defocus_slope : float
        The defocus slope for the tilt series. Default is 1.0.
        rlnTomoDefocusSlope in the tomograms.star file.

    Returns
    -------
    float
        The defocus offset for the particle in the tilt image.
        This is the distance from the tilt image centre to the particle
        in the tilted specimen along the z-axis of the tomogram.
    """
    proj_pos = tilt_proj_mat @ np.append(particle_coords, 1.0)
    proj_centre = tilt_proj_mat @ np.array([0.0, 0.0, 0.0, 1.0])

    return (proj_pos[2] - proj_centre[2]) * handedness * defocus_slope
