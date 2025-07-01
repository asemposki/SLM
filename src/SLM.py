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
from TOV_class import TOVsolver
from plotData import plot_eigs, plot_S, plot_dmd, plot_dmd_rad

# Ensure that the parent directory (project root) is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src import (
    SRC_DIR,
    EOS_CODES_DIR,
    EOS_DATA_DIR,
    EOS_FILES_DIR,
    RESULTS_PATH,
    TOV_PATH,
    PLOTS_PATH,
    MSEOS_PATH,
    QEOS_PATH,
    QEOS_TOV_PATH,
    MSEOS_TOV_PATH,
    SLM_RES_MSEOS,
    SLM_RES_QEOS,
)

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
def SLM(X, dt, error_threshold=1e-4, max_r=None, modes: int = None):
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
    
    if modes is None:
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
            # r_optimal = 1
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
    else:
        U_r = U_full[:, :modes]
        S_r_inv = np.diag(1.0 / S_full[:modes])
        V_r = Vt_full[:modes, :]

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

        r_optimal = None
        min_error = None
        # Store the results for the optimal r
        best_Xdmd = Xdmd_current_original_vars
        best_Phi = Phi_current[:n, :]  # Truncate Phi to original variables
        best_omega = omega_current
        best_lambda_vals = lambda_vals_current
        best_b = b_current

    return best_Phi, best_omega, best_lambda_vals, best_b, best_Xdmd, S_full, r_optimal


def solve_tov(fileName, tidal=False, parametric=False, mseos=True, pres_init=None):
    r"""
    Solves the TOV equation and returns radius, mass and central pressure

    Parameters:
        fileName (str): Filename containing the EOS in the format nb (fm^-3),
            E (MeV), P (MeV/fm^3)

    Returns:
        dataArray (array): Data array containing radii, central pressure
            and mass.
    """
    # Initialize tov_path_target to ensure it's always assigned
    tov_path_target = None  # Initialize to a default or None

    if parametric is False:
        eos_file = os.path.join(EOS_DATA_DIR, fileName)
        tov_path_target = TOV_PATH  # Assign for the non-parametric case
    else:
        if mseos is True:
            eos_file = os.path.join(MSEOS_PATH, fileName)
            tov_path_target = MSEOS_TOV_PATH
        else:
            eos_file = os.path.join(QEOS_PATH, fileName)
            tov_path_target = QEOS_TOV_PATH
            print("Path:", eos_file)

    # Now tov_path_target is guaranteed to be assigned
    if not os.path.exists(tov_path_target):
        os.makedirs(tov_path_target)

    # Replace the filename and run the code
    file = TOVsolver(eos_file, tidal=tidal)
    file.tov_routine(verbose=False, write_to_file=True, pres_init=pres_init)
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
    name_parts = ".".join((os.path.basename(fileName).split("."))[:-1]).split("_")
    print(name_parts)
    print("Name parts:", name_parts)
    if len(name_parts) > 2:
        output_file_name = "MR_" + "_".join(name_parts[1:]) + ".dat"
    else:
        output_file_name = "_".join(["MR", name_parts[0], "TOV"]) + ".dat"

    np.savetxt(output_file_name, dataArray.T, fmt="%1.8e")
    if not os.path.exists(tov_path_target):
        shutil.move(output_file_name, tov_path_target)
    else:
        print("file already exists")
    return dataArray


def main(fileName, tidal=False, parametric=False, mseos=True, pres_init=None):
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
            fileName, tidal, parametric, mseos, pres_init
        )
    else:
        radius, pcentral, mass = solve_tov(fileName, tidal, parametric, mseos, pres_init)
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

    # Make plots
    # Plot the S values
    plot_S(S)

    # Plot the eigenvalues
    plot_eigs(lam, filename="eigenValues.pdf")

    # Plot the DMDs
    fileNames = ["radiusDMD.png", "pressureDMD.png", "massDMD.png"]
    ylabels = ["Radius (km) ", r"Pressure $(MeV/fm^3)$", r"Mass $M_{(\odot)}$"]
    if tidal is True:
        fileNames.append("tidalDMD.png")
        ylabels.append(r"$\Lambda$")
    plot_dmd(linT, X, Xdmd, fileNames, ylabels, fileName)

    # Plot Mass vs Radius
    ylabels_rad_plots = [r"Pressure $(MeV/fm^3)$", r"Mass $M_{(\odot)}$"]
    newFiles = ["pressure_radius.png", "mass_radius.png"]
    if tidal is True:
        ylabels_rad_plots.append(r"$k_2$")
        newFiles.append("tidal_radius.png")
    plot_dmd_rad(X, Xdmd, newFiles, ylabels_rad_plots, fileName)

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
    # The svdSize argument is still passed from the command line, but now unused in main()
    (fileName, tidal, parametric, mseos, pres_init) = argv[1:]
    nameList = ".".join(os.path.basename(fileName).split(".")[:-1]).split("_")
    name = "SLM_" + "_".join(nameList) + ".dat"
    t, phi, omega, lam, b, Xdmd, HFTime, DMDTime = main(
        fileName, eval(tidal), eval(parametric), eval(mseos), eval(pres_init)
    )

    # Determine output directory based on parametric and mseos flags
    output_dir = RESULTS_PATH
    if eval(parametric) is True:
        if eval(mseos) is True:
            output_dir = SLM_RES_MSEOS
        else:
            output_dir = SLM_RES_QEOS

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)  # Change directory to save results

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
    print("Path: ", os.getcwd())
    with open(f"{name}", "w") as file:
        file.write(serializable_data)
