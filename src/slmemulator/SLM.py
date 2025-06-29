###########################################
# slmemulator code for the TOV data
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################

import numpy as np
import os
from itertools import combinations_with_replacement
import importlib.resources as pkg_resources
from .TOV_class import TOV  # Corrected to relative import

from .config import get_paths  # Corrected to relative import

# NEW: Define the importable path for internal EOS_Data
_INTERNAL_EOS_DATA_PATH = "slmemulator.EOS_Data"
_SLM_PACKAGE_NAME = "slmemulator"

p0 = 1.285e3


def augment_data_multiple_columns(X):
    r"""
    Augment the data matrix X with nonlinear terms for multiple variables.

    Parameters:
    X : np.ndarray
        The data matrix where each row is a variable, and each column is a snapshot in time.

    Returns:
    augmented_X : np.ndarray
        The augmented data matrix with nonlinear terms.
    """
    n_variables, n_snapshots = X.shape

    # Calculate the number of augmented terms
    num_augmented_terms = sum(
        1 for _ in combinations_with_replacement(range(n_variables), 2)
    )

    # Pre-allocate the augmented matrix
    augmented_X = np.empty(
        (n_variables + num_augmented_terms, n_snapshots), dtype=X.dtype
    )
    augmented_X[:n_variables, :] = X

    # Add quadratic terms and cross-products
    row_idx = n_variables
    for i, j in combinations_with_replacement(range(n_variables), 2):
        augmented_X[row_idx, :] = X[i, :] * X[j, :]
        row_idx += 1

    return augmented_X


# Numpy based SLM
def SLM(X, dt, error_threshold=1e-4, max_r=None):
    r"""
    Dynamic Mode decomposition for the augmented Data.
    Automatically determines the number of modes (r) based on an error threshold.

    Parameters:
    X: np.ndarray
        The data matrix where each row is a variable, and each column is a snapshot in time.
        Expected to be log-transformed where appropriate.
    dt: np.float
        Delta T: the time difference of linear DMDs.
    error_threshold: float, optional
        The maximum allowed absolute difference between the original data and the DMD reconstruction.
        Defaults to 1e-4.
    max_r: int, optional
        The maximum number of modes to consider. If None, it will go up to
        the maximum possible rank (min(X.shape)).
    """
    n = X.shape[0]  # Original number of variables before augmentation

    X_augmented = augment_data_multiple_columns(X)
    X1 = X_augmented[:, :-1]  # All columns except the last
    X2 = X_augmented[:, 1:]  # All columns except the first

    # Compute SVD of X1 once
    U_full, S_full, Vt_full = np.linalg.svd(X1, full_matrices=False)

    # Determine the maximum possible rank
    max_possible_r = min(X1.shape)
    if max_r is None:
        max_r_to_check = max_possible_r
    else:
        max_r_to_check = min(max_r, max_possible_r)

    # Initialize r and best_error
    r_optimal = 1
    min_error = float("inf")
    best_Xdmd = None
    best_Phi = None
    best_omega = None
    best_lambda_vals = None
    best_b = None

    # Iterate through possible ranks to find the optimal 'r'
    for r_current in range(1, max_r_to_check + 1):
        U_r = U_full[:, :r_current]
        S_r_inv = np.diag(1.0 / S_full[:r_current])
        V_r = Vt_full[:r_current, :]

        # Compute Atilde
        Atilde = U_r.T @ X2 @ V_r.T @ S_r_inv

        # Compute eigenvectors and eigenvalues
        D, W_r = np.linalg.eig(Atilde)

        Phi_current = X2 @ V_r.T @ S_r_inv @ W_r  # DMD modes
        lambda_vals_current = D  # discrete-time eigenvalues
        omega_current = np.log(lambda_vals_current) / dt  # continuous-time eigenvalue

        # Compute DMD mode amplitudes b
        x1 = X1[:, 0]
        b_current = np.linalg.lstsq(Phi_current, x1, rcond=None)[0]

        # DMD reconstruction for the current 'r'
        mm1 = X1.shape[1] + 1
        t = np.arange(mm1) * dt

        time_dynamics_current = b_current[:, np.newaxis] * np.exp(
            omega_current[:, np.newaxis] * t
        )
        Xdmd2_current = Phi_current @ time_dynamics_current

        # Truncate to original number of variables (log-transformed)
        Xdmd_current_original_vars = Xdmd2_current[:n, :]

        # Calculate error (max absolute difference)
        # Using original X (log-transformed) for comparison
        current_error = np.max(np.abs(X - Xdmd_current_original_vars))

        print(f"Testing r={r_current}: Max absolute error = {current_error:.6f}")

        if current_error <= error_threshold:
            r_optimal = r_current
            min_error = current_error
            # Store the results for the optimal r
            best_Xdmd = Xdmd_current_original_vars
            best_Phi = Phi_current[:n, :]  # Truncate Phi to original variables
            best_omega = omega_current
            best_lambda_vals = lambda_vals_current
            best_b = b_current
            break  # Found the smallest r that satisfies the threshold

        # If we didn't meet the threshold, but this r gives the best error so far, keep its results
        if current_error < min_error:
            min_error = current_error
            r_optimal = r_current
            best_Xdmd = Xdmd_current_original_vars
            best_Phi = Phi_current[:n, :]
            best_omega = omega_current
            best_lambda_vals = lambda_vals_current
            best_b = b_current

    print(f"Optimal 'r' determined: {r_optimal} (Max absolute error = {min_error:.6f})")

    # If no 'r' met the threshold, use the one that gave the minimum error
    if best_Xdmd is None:  # This should not happen if max_r_to_check >= 1
        # Fallback to a default if no optimal r is found (e.g., r=1)
        r_optimal = 1
        U_r = U_full[:, :r_optimal]
        S_r_inv = np.diag(1.0 / S_full[:r_optimal])
        V_r = Vt_full[:r_optimal, :]
        Atilde = U_r.T @ X2 @ V_r.T @ S_r_inv
        D, W_r = np.linalg.eig(Atilde)
        best_Phi = X2 @ V_r.T @ S_r_inv @ W_r
        best_lambda_vals = D
        best_omega = np.log(best_lambda_vals) / dt
        x1 = X1[:, 0]
        best_b = np.linalg.lstsq(best_Phi, x1, rcond=None)[0]
        mm1 = X1.shape[1] + 1
        t = np.arange(mm1) * dt
        time_dynamics = best_b[:, np.newaxis] * np.exp(best_omega[:, np.newaxis] * t)
        best_Xdmd = (best_Phi @ time_dynamics)[:n, :]
        print(
            f"Warning: No r met the threshold. Using r={r_optimal} with max error {np.max(np.abs(X - best_Xdmd)):.6f}"
        )

    return best_Phi, best_omega, best_lambda_vals, best_b, best_Xdmd, S_full, r_optimal


def solve_tov(fileName, tidal=False, parametric=False, mseos=True):
    r"""
    Solves the TOV equation and returns radius, mass and central pressure

    Parameters:
        fileName (str): Filename containing the EOS in the format nb (fm^-3),
            E (MeV), P (MeV/fm^3)

    Returns:
        dataArray (array): Data array containing radii, central pressure
            and mass.
    """
    # Get current paths from the config module
    paths = get_paths()

    eos_file_path = None
    tov_path_target = None

    if parametric is False:
        # Access EOS_Data as package resource
        try:
            slm_package_base = pkg_resources.files(_SLM_PACKAGE_NAME)
            # Get a Path-like object to the EOS_Data file within the installed package
            eos_file_path = pkg_resources.files(_INTERNAL_EOS_DATA_PATH).joinpath(
                fileName
            )
            eos_data_resource_dir = slm_package_base.joinpath("EOS_Data")
            eos_file_path = eos_data_resource_dir.joinpath(fileName)
            # When passing to TOV, ensure it's a string path if the solver expects it
            file = TOV(str(eos_file_path), tidal=tidal)
            tov_path_target = paths["tov_data_dir"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Internal EOS file '{fileName}' not found in package data."
            )
    else:
        # Access EOS_files from external configured path
        if mseos is True:
            eos_file_path = paths["mseos_path"] / fileName
            tov_path_target = paths["mseos_tov_path"]
        else:
            eos_file_path = paths["qeos_path"] / fileName
            tov_path_target = paths["qeos_tov_path"]
            print("Path:", eos_file_path)

        file = TOV(str(eos_file_path), tidal=tidal)  # Pass as string path

    # Now tov_path_target is guaranteed to be assigned
    if not tov_path_target.exists():
        tov_path_target.mkdir(parents=True, exist_ok=True)

    file.tov_routine(verbose=False, write_to_file=False)
    print("R of 1.4 solar mass star: ", file.canonical_NS_radius())
    dataArray = [
        file.total_radius.flatten(),
        file.total_pres_central.flatten(),
        file.total_mass.flatten(),
    ]
    if tidal is True:
        dataArray.append(file.k2.flatten())
        # dataArray.append(file.tidal_deformability.flatten()[::-1])

    dataArray = np.asarray(dataArray, dtype=np.float64)

    # Construct the output filename more robustly
    name_parts = os.path.basename(fileName).strip(".txt").split("_")
    print("Name parts:", name_parts)
    if len(name_parts) > 2:
        output_file_name = "MR_" + "_".join(name_parts[1:]) + ".txt"
    else:
        output_file_name = "_".join(["MR", name_parts[0], "TOV"]) + ".txt"

    # Save directly to the target path without changing directory
    output_full_path = tov_path_target / output_file_name
    np.savetxt(output_full_path, dataArray.T, fmt="%1.8e")
    return dataArray
