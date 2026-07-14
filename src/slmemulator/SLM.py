###########################################
# slmemulator code for the TOV data
# Author: Sudhanva Lalit
###########################################

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from .config import get_paths
from .TOV_class import TOV

__all__ = ["SLM", "augment_data_multiple_columns", "solve_tov"]


def augment_data_multiple_columns(X):
    r"""
    Augment the data matrix X with nonlinear terms for multiple variables.

    Parameters:
        X (np.ndarray): The data matrix where each row is a variable, and each
            column is a snapshot in time.

    Returns:
        augmented_X (np.ndarray): The augmented data matrix with quadratic and
            cross-product terms appended below the original rows.
    """
    n_variables, n_snapshots = X.shape

    pairs = list(combinations_with_replacement(range(n_variables), 2))
    augmented_X = np.empty((n_variables + len(pairs), n_snapshots), dtype=X.dtype)
    augmented_X[:n_variables, :] = X

    for row_idx, (i, j) in enumerate(pairs, start=n_variables):
        augmented_X[row_idx, :] = X[i, :] * X[j, :]

    return augmented_X


def SLM(X, dt, error_threshold=1e-4, max_r=None):
    r"""
    Dynamic Mode Decomposition of the augmented data.
    Automatically determines the number of modes (r) based on an error threshold.

    Parameters:
        X (np.ndarray): The data matrix where each row is a variable, and each
            column is a snapshot in time. Expected to be log-transformed where
            appropriate.
        dt (float): The time difference of linear DMDs.
        error_threshold (float): (Optional) The maximum allowed absolute
            difference between the original data and the DMD reconstruction.
            Defaults to 1e-4.
        max_r (int): (Optional) The maximum number of modes to consider.
            If None, goes up to the maximum possible rank (min(X.shape)).

    Returns:
        tuple: (Phi, omega, lambda_vals, b, Xdmd, S, r_optimal) where Phi are
        the DMD modes (truncated to the original variables), omega the
        continuous-time eigenvalues, lambda_vals the discrete-time
        eigenvalues, b the mode amplitudes, Xdmd the reconstruction of the
        original (non-augmented) variables, S the singular values of the
        augmented snapshot matrix, and r_optimal the selected rank.
    """
    n = X.shape[0]  # Original number of variables before augmentation

    X_augmented = augment_data_multiple_columns(X)
    X1 = X_augmented[:, :-1]  # All columns except the last
    X2 = X_augmented[:, 1:]  # All columns except the first

    # Compute SVD of X1 once
    U_full, S_full, Vt_full = np.linalg.svd(X1, full_matrices=False)

    max_possible_r = min(X1.shape)
    max_r_to_check = max_possible_r if max_r is None else min(max_r, max_possible_r)

    # Reconstruction times (all snapshots)
    t = np.arange(X1.shape[1] + 1) * dt
    x1 = X1[:, 0]

    r_optimal = 1
    min_error = np.inf
    best = None

    # Find the smallest r whose reconstruction meets the threshold; otherwise
    # keep the r with the smallest error.
    for r_current in range(1, max_r_to_check + 1):
        U_r = U_full[:, :r_current]
        S_r_inv = np.diag(1.0 / S_full[:r_current])
        V_r = Vt_full[:r_current, :]

        Atilde = U_r.T @ X2 @ V_r.T @ S_r_inv
        lambda_vals, W_r = np.linalg.eig(Atilde)  # discrete-time eigenvalues

        Phi = X2 @ V_r.T @ S_r_inv @ W_r  # DMD modes
        omega = np.log(lambda_vals) / dt  # continuous-time eigenvalues

        # DMD mode amplitudes and reconstruction
        b = np.linalg.lstsq(Phi, x1, rcond=None)[0]
        time_dynamics = b[:, np.newaxis] * np.exp(omega[:, np.newaxis] * t)
        Xdmd = (Phi @ time_dynamics)[:n, :]  # truncate to original variables

        current_error = np.max(np.abs(X - Xdmd))

        if current_error <= error_threshold or current_error < min_error:
            r_optimal = r_current
            min_error = current_error
            best = (Phi[:n, :], omega, lambda_vals, b, Xdmd)

        if current_error <= error_threshold:
            break

    print(f"Optimal 'r' determined: {r_optimal} (Max absolute error = {min_error:.6f})")

    best_Phi, best_omega, best_lambda_vals, best_b, best_Xdmd = best
    return best_Phi, best_omega, best_lambda_vals, best_b, best_Xdmd, S_full, r_optimal


def solve_tov(fileName, tidal=False, parametric=False, mseos=True):
    r"""
    Solves the TOV equation and returns radius, central pressure and mass.

    Parameters:
        fileName (str): Filename containing the EOS in the format nb (fm^-3),
            E (MeV), P (MeV/fm^3). For non-parametric runs this is the name of
            a file in the packaged EOS_Data; for parametric runs it is looked
            up in the generated EOS_files directory.
        tidal (bool): Also compute the tidal Love number k2. Default False.
        parametric (bool): Whether the EOS file comes from a parametric run.
        mseos (bool): For parametric runs, MSEOS (True) or Quarkyonia (False).

    Returns:
        dataArray (np.ndarray): Data array containing radii, central pressure
            and mass (includes tidal deformability k_2 if tidal is True).
    """
    paths = get_paths()

    if not parametric:
        # EOS file shipped with the package
        eos_file_path = paths["package_eos_data_dir"] / fileName
        if not eos_file_path.is_file():
            raise FileNotFoundError(
                f"Internal EOS file '{fileName}' not found in package data."
            )
        tov_path_target = paths["user_tov_data_dir"]
    elif mseos:
        eos_file_path = paths["mseos_path_specific"] / fileName
        tov_path_target = paths["mseos_tov_path_specific"]
    else:
        eos_file_path = paths["qeos_path_specific"] / fileName
        tov_path_target = paths["qeos_tov_path_specific"]

    tov = TOV(str(eos_file_path), tidal=tidal)
    tov_path_target.mkdir(parents=True, exist_ok=True)

    tov.tov_routine(verbose=False, write_to_file=False)
    print("R of 1.4 solar mass star: ", tov.canonical_NS_radius())

    dataArray = [
        tov.total_radius.flatten(),
        tov.total_pres_central.flatten(),
        tov.total_mass.flatten(),
    ]
    if tidal:
        dataArray.append(tov.k2.flatten())
    dataArray = np.asarray(dataArray, dtype=np.float64)

    # Output name: MR_<params>.txt for parametric files, MR_<name>_TOV.txt otherwise
    name_parts = Path(fileName).name.removesuffix(".txt").split("_")
    if len(name_parts) > 2:
        output_file_name = "MR_" + "_".join(name_parts[1:]) + ".txt"
    else:
        output_file_name = "_".join(["MR", name_parts[0], "TOV"]) + ".txt"

    np.savetxt(tov_path_target / output_file_name, dataArray.T, fmt="%1.8e")
    return dataArray
