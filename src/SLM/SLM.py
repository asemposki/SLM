###########################################
# SLM code for the TOV data
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################

import numpy as np
import os
import shutil
import sys
import json
from itertools import combinations_with_replacement
import time
import importlib.resources as pkg_resources  # Already present, good!
from TOV_class import TOVsolver
from plotData import plot_eigs, plot_S, plot_dmd, plot_dmd_rad

# NEW: Import get_paths from your package's config module
from SLM.config import get_paths  #

# NEW: Define the importable path for internal EOS_Data
_INTERNAL_EOS_DATA_PATH = "SLM.EOS_Data"  #


# REMOVE THESE LINES, as they are no longer needed
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)
# from src import ( # This entire block should be removed
#     SRC_DIR,
#     EOS_CODES_DIR,
#     EOS_DATA_DIR,
#     EOS_FILES_DIR,
#     RESULTS_PATH,
#     TOV_PATH,
#     PLOTS_PATH,
#     MSEOS_PATH,
#     QEOS_PATH,
#     QEOS_TOV_PATH,
#     MSEOS_TOV_PATH,
#     SLM_RES_MSEOS,
#     SLM_RES_QEOS,
# )

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
    paths = get_paths()  #

    eos_file_path = None
    tov_path_target = None

    if parametric is False:
        # Access EOS_Data as package resource
        try:
            # Get a Path-like object to the EOS_Data file within the installed package
            eos_file_path = pkg_resources.files(_INTERNAL_EOS_DATA_PATH).joinpath(
                fileName
            )
            # When passing to TOVsolver, ensure it's a string path if the solver expects it
            file = TOVsolver(str(eos_file_path), tidal=tidal)
            tov_path_target = paths["tov_data_dir"]  #
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Internal EOS file '{fileName}' not found in package data."
            )
    else:
        # Access EOS_files from external configured path
        if mseos is True:
            eos_file_path = paths["mseos_path"] / fileName  #
            tov_path_target = paths["mseos_tov_path"]  #
        else:
            eos_file_path = paths["qeos_path"] / fileName  #
            tov_path_target = paths["qeos_tov_path"]  #
            print("Path:", eos_file_path)  #

        file = TOVsolver(str(eos_file_path), tidal=tidal)  # Pass as string path

    # Now tov_path_target is guaranteed to be assigned
    if not os.path.exists(tov_path_target):  # Use os.path.exists with Path objects
        os.makedirs(tov_path_target)  # Use os.makedirs with Path objects

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
    output_full_path = tov_path_target / output_file_name  #
    np.savetxt(output_full_path, dataArray.T, fmt="%1.8e")  #
    # shutil.move is not needed if you save directly to the target path.
    # If the file is created in CWD first (e.g. by another library), then uncomment:
    # shutil.move(output_file_name, output_full_path)
    return dataArray


def main(fileName, tidal=False, parametric=False, mseos=True):
    r"""
    Main function to run the SLM code. Solves the TOV equation and
    computes the SLM modes.

    Parameters:
        fileName (str): Filename containing the EOS in the format nb (fm^-3),
            E (MeV), P (MeV/fm^3)
        svdSize_unused (int): This parameter is now unused as 'r' is automatically determined.
        tidal (bool): Whether to include tidal deformability
        parametric (bool): Whether the EOS is parametric
        mseos (bool): Whether to use MSEOS

    Returns:
        linT (np.ndarray): Time vector
        phi (np.ndarray): DMD modes
        omega (np.ndarray): Continuous-time eigenvalues
        lam (np.ndarray): Discrete-time eigenvalues
        b (np.ndarray): DMD mode amplitudes
        Xdmd (np.ndarray): DMD reconstruction
        HFTime (float): Time taken for solving TOV
        DMDTime (float): Time taken for DMD
    """
    startHFTime = time.time()
    if tidal is True:
        radius, pcentral, mass, tidal_def = solve_tov(
            fileName, tidal, parametric, mseos
        )
    else:
        radius, pcentral, mass = solve_tov(fileName, tidal, parametric, mseos)
    endHFTime = time.time()

    # Assign variables directly from the solve_tov output
    r_orig = radius
    p_orig = pcentral
    m_orig = mass

    linT = np.arange(len(p_orig))

    # Prepare X for DMD, handling tidal deformability
    X_list = [np.log(r_orig), np.log(p_orig), np.log(m_orig)]
    if tidal is True:
        X_list.append(np.log(tidal_def))
    X = np.asarray(X_list, dtype=np.float64)

    startDMDTime = time.time()
    # Call SLM without svdSize, it will determine 'r' automatically
    # Pass original X for error calculation
    phi, omega, lam, b, Xdmd, S, r_auto = SLM(
        X, (linT[-1] - linT[0]) / len(linT), error_threshold=1e-4
    )  # Added error_threshold
    endDMDTime = time.time()

    # Extract real parts for plotting and reconstruction
    rad_DMD = np.exp(Xdmd[0].real)
    pres_DMD = np.exp(Xdmd[1].real)
    mass_DMD = np.exp(Xdmd[2].real)

    # Get paths for plotting
    paths = get_paths()
    plots_dir = paths["plots_dir"]  #
    plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure plots directory exists

    # Make plots
    # Plot the S values
    plot_S(S, save_dir=plots_dir)  # Pass save_dir

    # Plot the eigenvalues
    plot_eigs(lam, filename="eigenValues.pdf", save_dir=plots_dir)  # Pass save_dir

    # Plot the DMDs
    fileNames = ["radiusDMD.png", "pressureDMD.png", "massDMD.png"]
    ylabels = ["Radius (km) ", r"Pressure $(MeV/fm^3)$", r"Mass $M_{(\odot)}$"]
    if tidal is True:
        fileNames.append("tidalDMD.png")
        ylabels.append(r"$\Lambda$")
    plot_dmd(
        linT, X, Xdmd, fileNames, ylabels, fileName, save_dir=plots_dir
    )  # Pass save_dir

    # Plot Mass vs Radius
    ylabels_rad_plots = [r"Pressure $(MeV/fm^3)$", r"Mass $M_{(\odot)}$"]
    newFiles = ["pressure_radius.png", "mass_radius.png"]
    if tidal is True:
        ylabels_rad_plots.append(r"$k_2$")
        newFiles.append("tidal_radius.png")
    plot_dmd_rad(
        X, Xdmd, newFiles, ylabels_rad_plots, fileName, save_dir=plots_dir
    )  # Pass save_dir

    # maximum values
    max_mass_DMD = np.max(mass_DMD)
    # Using np.argmax for efficiency and robustness
    max_index = np.argmax(mass_DMD)
    max_radius_DMD = rad_DMD[max_index]

    print(
        "DMD Maximum mass: {}; maximum radius: {}".format(max_mass_DMD, max_radius_DMD)
    )
    HFTime = endHFTime - startHFTime
    DMDTime = endDMDTime - startDMDTime
    return linT, phi, omega, lam, b, Xdmd, HFTime, DMDTime


def complex_encoder(obj):
    r"""
    Complex encoder for JSON serialization. Converts complex numbers to
    real and imaginary parts.
    """
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    raise TypeError("Type not serializable")


if __name__ == "__main__":
    argv = sys.argv
    (fileName, tidal, parametric, mseos) = argv[1:]
    nameList = os.path.basename(fileName).strip(".txt").split("_")
    name = "SLM_" + "_".join(nameList[1:]) + ".txt"

    t, phi, omega, lam, b, Xdmd, HFTime, DMDTime = main(
        fileName, eval(tidal), eval(parametric), eval(mseos)
    )

    # Get paths for results
    paths = get_paths()  #

    # Determine output directory based on parametric and mseos flags
    output_dir = paths["results_dir"]  # Default to general results directory
    if eval(parametric) is True:
        if eval(mseos) is True:
            output_dir = paths["slm_res_mseos"]  #
        else:
            output_dir = paths["slm_res_qeos"]  #

    if not output_dir.exists():  # Use Path.exists()
        output_dir.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir()

    # REMOVED: os.chdir(output_dir) - Avoid changing the current working directory

    # Prepare Xdmd for serialization, ensuring real parts and correct order
    XdmdRes = np.exp(Xdmd.real)

    data = dict()
    data.update({"time": t.tolist()})
    if len(nameList) > 2:
        data.update({"val": nameList[2:]})
    data.update({"phi": np.array(phi, dtype=np.complex64).tolist()})
    data.update({"omega": omega.real.astype("float64").tolist()})
    data.update({"lam": np.array(lam, dtype=np.complex64).tolist()})
    data.update({"b": b.real.astype("float64").tolist()})
    data.update(
        {"Xdmd": XdmdRes.real.astype("float64").tolist()}
    )  # Ensure real and float64
    data.update({"hftime": HFTime})
    data.update({"dmdtime": DMDTime})

    serializable_data = json.dumps(
        data, default=complex_encoder, sort_keys=True, indent=4
    )
    # Print the resolved output path instead of current working directory
    output_file_full_path = output_dir / name  #
    print("Output Path: ", output_file_full_path)  #
    with open(output_file_full_path, "w") as file:  # Write directly to the full path
        file.write(serializable_data)
