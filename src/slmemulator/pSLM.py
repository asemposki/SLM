import numpy as np
import os
import sys
from itertools import combinations_with_replacement
from sklearn.preprocessing import StandardScaler
from slmemulator.config import get_paths
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# Get all the defined paths from config.py
paths = get_paths()


def gaussian_kernel(x1, x2, sigma=1.0):
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    dist_sq = np.sum((x1_arr - x2_arr) ** 2)
    return np.exp(-dist_sq / (2 * sigma**2 + 1e-9))


class ParametricSLM:
    def __init__(
        self, fileList, filePath, tidal=False, error_threshold=1e-4, max_r=None
    ):
        # fileList is expected to be a list of Path objects from testpSLM.py
        self.fileList = [Path(f) for f in fileList]  # Ensure all are Path objects
        self.filePath = Path(filePath)  # Ensure filePath is a Path object
        self.tidal = tidal
        self.error_threshold = error_threshold
        self.max_r = max_r
        self.n = None  # Number of quantities/rows in the data X
        self.mm1 = (
            None  # Number of time points/columns in the data X (m-1 snapshots for DMD)
        )

        # Lists to store DMD components for each training file
        self.Phi_list = []
        self.omega_list = []
        self.D_list = []
        self.b_list = []
        self.train_data_shapes = []  # Store shapes (n, mm1) of training data

        # Store extracted parameters for each file, in the same order as fileList
        self.file_params = []
        self.param_scaler = None  # Scaler for parameters, fitted in .fit()
        self.scaled_params = None  # Scaled parameters of training data

    @staticmethod
    def augment_data(X):
        """
        Augments the input data X by adding quadratic terms (X_i * X_j).

        Args:
            X (np.ndarray): Original data matrix (n, m).

        Returns:
            np.ndarray: Augmented data matrix.
        """
        n, m = X.shape
        out = [X]
        # Add quadratic terms (combinations with replacement of rows of X)
        for i, j in combinations_with_replacement(range(n), 2):
            out.append(X[i] * X[j])
        # Add cubic terms if needed (example commented out)
        # for i, j, k in combinations_with_replacement(range(n), 3):
        #     out.append(X[i] * X[j] * X[k])
        return np.vstack(out)

    @staticmethod
    def _extract_params_from_tov_filename(filename_stem, is_mseos_run):
        """
        Extracts numerical parameters from TOV data filenames based on EOS type.
        Expected formats:
        - MSEOS: TOV_MS_ls_lv_zeta_xi
        - Quarkyonia: TOV_QEOS_lam_kappa

        Args:
            filename_stem (str): The filename without extension (e.g., 'TOV_MS_0.00_0.00_1.00e-04_0.0').
            is_mseos_run (bool): True if the run is for MSEOS, False for Quarkyonia.

        Returns:
            list: A list of floats representing the extracted parameters, or None if parsing fails.
        """
        parts = filename_stem.split("_")

        if is_mseos_run:
            # Expected format: TOV_MS_Ls_Lv_zeta_xi
            if len(parts) == 6 and parts[0] == "TOV" and parts[1] == "MS":
                try:
                    # Parameters are Ls, Lv, zeta, xi
                    return [
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                    ]
                except ValueError as e:
                    print(f"Error parsing MSEOS filename {filename_stem}: {e}")
                    return None
            else:
                print(
                    f"Warning: Filename format not recognized for MSEOS: {filename_stem}"
                )
                return None
        else:  # Quarkyonia
            # Expected format: TOV_QEOS_lambda_kappa
            if len(parts) == 4 and parts[0] == "TOV" and parts[1] == "QEOS":
                try:
                    # Parameters are Lambda, Kappa
                    return [float(parts[2]), float(parts[3])]
                except ValueError as e:
                    print(f"Error parsing Quarkyonia filename {filename_stem}: {e}")
                    return None
            else:
                print(
                    f"Warning: Filename format not recognized for Quarkyonia: {filename_stem}"
                )
                return None

    def _SLM_auto_r(self, X, dt):
        """
        Performs Sparse Linear Modeling (SLM) with automatic determination of rank 'r'.

        Args:
            X (np.ndarray): Data matrix (n_features, n_snapshots).
            dt (float): Time step between snapshots.

        Returns:
            dict: Dictionary containing best DMD components (Phi, omega, D, b, etc.)
                  and the reconstruction error.
        """
        n = X.shape[0]  # Original number of features
        X_aug = self.augment_data(X)  # Augmented data

        # Ensure augmented_n is set correctly here from the first data processing
        if self.n is None:  # This should be consistent across all files
            self.n = X.shape[0]
            self.augmented_n = X_aug.shape[0]
            self.mm1 = X.shape[1]  # Number of snapshots

        X1, X2 = X_aug[:, :-1], X_aug[:, 1:]  # Shifted data matrices for DMD

        # Check for empty X1 or X2
        if X1.shape[1] == 0 or X2.shape[1] == 0:
            return {
                "err": float("inf")
            }  # Cannot perform DMD if no snapshots or only one snapshot

        # SVD on X1
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)

        # Determine optimal rank 'r'
        max_r_actual = (
            min(X1.shape) if self.max_r is None else min(self.max_r, min(X1.shape))
        )

        # Use a more robust threshold if S is very small
        singular_value_threshold = S[0] * self.error_threshold if S.size > 0 else 0
        r_auto = np.sum(S > singular_value_threshold)
        r = max(1, min(r_auto, max_r_actual))  # Ensure r is at least 1 and within max_r

        best = {"err": float("inf")}  # Stores the best DMD result

        # Iterate through possible ranks (or just use the determined r)
        # In SLM_auto_r, typically you would find one optimal 'r' or iterate
        # This implementation iterates to find the best 'r'
        for current_r in range(1, r + 1):  # Iterate up to the chosen rank
            U_r = U[:, :current_r]
            S_r_diag = S[:current_r]
            Vt_r = Vt[:current_r, :]

            # Prevent division by zero / very small numbers in S_r_inv
            S_r_diag[S_r_diag < 1e-12] = 1e-12
            S_r_inv = np.diag(1.0 / S_r_diag)  # Inverse of diagonal singular values

            # Compute A_tilde (Koopman operator approximation)
            Atilde = U_r.T @ X2 @ Vt_r.T @ S_r_inv

            # Check for invalid values after matrix multiplication
            if not np.all(np.isfinite(Atilde)):
                # print(f"Warning: Atilde contains non-finite values for r={current_r}. Skipping.")
                continue

            try:
                D, W = np.linalg.eig(
                    Atilde
                )  # Eigenvalues (D) and Eigenvectors (W) of Atilde
            except np.linalg.LinAlgError as e:
                # print(f"Warning: LinAlgError during eigendecomposition for r={current_r}: {e}. Skipping.")
                continue

            # Compute Koopman modes (Phi)
            # Use pseudo-inverse for numerical stability, especially if S_r is ill-conditioned
            # Phi = X2 @ np.linalg.pinv(X1) @ U_r @ np.linalg.inv(S_r) @ W # Alternative form
            Phi = (
                X2 @ Vt_r.T @ S_r_inv @ W
            )  # Original form, relies on S_r_inv being robust

            # Compute DMD frequencies (omega)
            # dt is important here if D are discrete-time eigenvalues
            omega = np.log(D) / dt

            # Compute initial amplitudes (b)
            # Use lstsq for robustness instead of direct inverse if Phi is ill-conditioned
            b = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]

            # Reconstruct the data (Xdmd)
            t_reconstruction = (
                np.arange(X.shape[1]) * dt
            )  # Use all original time points
            time_dynamics = b[:, np.newaxis] * np.exp(
                omega[:, np.newaxis] * t_reconstruction[np.newaxis, :]
            )
            Xdmd_full_aug = Phi @ time_dynamics
            Xdmd = Xdmd_full_aug[
                : self.n, :
            ]  # Trim back to original number of features

            # Calculate reconstruction error
            err = np.max(np.abs(X - Xdmd))

            # Update best result if current error is better
            if err < best["err"]:
                best = dict(
                    r=current_r,
                    D=D,
                    U=U_r,
                    W=W,
                    V=Vt_r,
                    S=S_r_diag,
                    b=b,
                    Atilde=Atilde,
                    Phi=Phi,
                    omega=omega,
                    Xdmd=Xdmd,
                    err=err,
                )

            # Early exit if error threshold is met
            if err <= self.error_threshold:
                break

        # If no valid r found (e.g., all iterations failed or X1 was empty)
        if best["err"] == float("inf"):
            # print(f"Warning: No valid DMD model could be fitted for the given data. Max error remains Inf.")
            # Return a default structure to prevent errors downstream
            return dict(
                r=0,
                D=np.array([]),
                U=np.array([[]]),
                W=np.array([[]]),
                V=np.array([[]]),
                S=np.array([]),
                b=np.array([]),
                Atilde=np.array([[]]),
                Phi=np.array([[]]),
                omega=np.array([]),
                Xdmd=np.zeros_like(X),
                err=float("inf"),
            )
        return best

    def fit(self):
        """
        Fits the Parametric SLM model by processing each file in fileList,
        performing DMD, and storing the DMD components along with extracted parameters.
        """
        self.Phi_list = []
        self.omega_list = []
        self.D_list = []
        self.b_list = []
        self.file_params = []  # Reset for clarity, though init should handle it
        self.train_data_shapes = []  # Store shapes for consistency check in predict

        processed_files_count = 0
        for file_path_obj in self.fileList:  # Iterate through Path objects
            # Validate file_path_obj is a Path object and exists
            if not isinstance(file_path_obj, Path) or not file_path_obj.exists():
                print(f"Skipping invalid or non-existent file: {file_path_obj}")
                continue

            # Extract parameters from the filename (e.g., TOV_MS_Ls_Lv_zeta_xi.txt)
            # We assume the file name format is consistent with _extract_params_from_tov_filename
            # We determine mseos based on 'MS' or 'QEOS' in the filename
            is_mseos = "MS" in file_path_obj.stem  # Check if filename contains 'MS'
            # Call the static method using self.
            params = self._extract_params_from_tov_filename(
                file_path_obj.stem, is_mseos
            )

            if params is None:  # _extract_params_from_tov_filename prints warning
                continue  # Skip if parameters could not be extracted

            try:
                # Load data from the file
                # The file is a Path object, np.loadtxt can directly handle it
                data = np.loadtxt(file_path_obj)

                if not np.all(np.isfinite(data)):
                    print(
                        f"Skipping {file_path_obj.name}: contains NaN or Inf values. Please check data quality."
                    )
                    continue

                # Process data (assuming data has columns like Radius, Mass, etc.)
                # X should be (num_quantities, num_points)
                epsilon = 1e-9  # Small constant to prevent log(0)
                if self.tidal:
                    # Assuming 4 columns: Radius, Mass, Central Pressure, Tidal Deformability
                    # Ensure data has at least 4 columns
                    if data.shape[1] < 4:
                        print(
                            f"Skipping {file_path_obj.name}: Expected at least 4 columns for tidal data, got {data.shape[1]}."
                        )
                        continue
                    X = np.log(data[:, [0, 1, 2, 3]].T + epsilon)
                else:
                    # Assuming 3 columns: Radius, Mass, Central Pressure (or other relevant)
                    # Ensure data has at least 3 columns
                    if data.shape[1] < 3:
                        print(
                            f"Skipping {file_path_obj.name}: Expected at least 3 columns for non-tidal data, got {data.shape[1]}."
                        )
                        continue
                    X = np.log(data[:, [0, 1, 2]].T + epsilon)

                X = np.asarray(
                    X, dtype=np.float64
                )  # Ensure float64 for numerical stability

                # Store shape for consistency checks in predict
                if (
                    self.n is None
                ):  # Only set for the very first successfully processed file
                    self.n, self.mm1 = X.shape[0], X.shape[1]
                    self.augmented_n = self.augment_data(
                        np.zeros((self.n, self.mm1))
                    ).shape[0]
                elif X.shape[0] != self.n or X.shape[1] != self.mm1:
                    print(
                        f"Warning: {file_path_obj.name} has inconsistent shape {X.shape}. Expected ({self.n}, {self.mm1}). Skipping."
                    )
                    continue  # Skip files with inconsistent shapes for DMD

                dt = 1.0  # Assuming uniform time step of 1.0 between data points
                result = self._SLM_auto_r(X, dt)  # Perform SLM and get optimal rank

                # Check if SLM_auto_r returned a valid result
                if result["err"] == float("inf"):
                    print(
                        f"Failed to fit SLM model for {file_path_obj.name}. Skipping."
                    )
                    continue

                # Store DMD components and parameters for successful fits
                self.Phi_list.append(result["Phi"])
                self.omega_list.append(result["omega"])
                self.D_list.append(result["D"])
                self.b_list.append(result["b"])
                self.file_params.append(
                    params
                )  # Store parameters for the processed file
                self.train_data_shapes.append(
                    X.shape
                )  # Store shape for this specific training data

                processed_files_count += 1
                # print(f"Successfully processed {file_path_obj.name}. Max error: {result['err']:.4e}")

            except Exception as e:
                print(f"Error processing file {file_path_obj.name}: {e}. Skipping.")
                continue

        if processed_files_count == 0:
            raise RuntimeError(
                "No valid training files processed! Please check your input data and `error_threshold`."
            )

        # Fit StandardScaler on the collected parameters
        self.params = np.array(self.file_params)
        self.param_scaler = StandardScaler().fit(self.params)
        self.scaled_params = self.param_scaler.transform(self.params)

        # Optional: Print DMD fit errors
        # print("\nDMD fit errors for training files:")
        # for i, file_path_obj in enumerate(self.fileList): # Use self.fileList to match actual processed files
        #     if i < len(self.dmd_errors): # Ensure index is valid
        #         print(f"{file_path_obj.name}: max abs error = {self.dmd_errors[i]:.4e}")

    def predict(self, param_values, k=1, output_interp=False, distance_threshold=None):
        """
        Predicts the dynamics (Xdmd) for a given set of parameters by finding nearest neighbors
        in the parameter space and averaging their DMD components or using the closest one.

        Args:
            param_values (list or np.ndarray): A list or array of parameter values to predict for
                                                         (e.g., [Ls, Lv, zeta, xi] for MSEOS). This argument is mandatory.
            k (int): Number of nearest neighbors to use for prediction.
                     Defaults to 1 (pure nearest neighbor).
            output_interp (bool): If True, interpolate the final Xdmd outputs by averaging curves.
                                  If False (default) and k>1, averages the DMD components (Phi, omega, D, b).
            distance_threshold (float, optional): If the distance to the closest neighbor
                                                  exceeds this threshold, a warning is printed.

        Returns:
            tuple: (Phi_avg/Phi_nn, omega_avg/omega_nn, D_avg/D_nn, b_avg/b_nn, Xdmd_predicted, t)
                   - Reconstructed data (Xdmd_predicted) and related averaged DMD components.
        Raises:
            ValueError: If the model has not been fitted or no training data is available.
        """
        if not self.Phi_list:
            raise ValueError("Model not fitted. Call .fit() first.")
        if not self.file_params:
            raise ValueError("No training data parameters available from fit().")

        # If param_values are provided (the main use case for prediction)
        theta = np.asarray(param_values, dtype=np.float64)

        # Scale the input parameters using the fitted scaler
        try:
            scaled_theta = self.param_scaler.transform([theta])[0]
        except ValueError as e:
            raise ValueError(
                f"Parameter scaling failed for {theta}. Ensure input matches training dimensions. Error: {e}"
            )

        # Calculate Euclidean distances to all scaled training parameters
        dists = np.linalg.norm(self.scaled_params - scaled_theta, axis=1)

        # Get indices of k smallest distances
        # Ensure k does not exceed the number of available training samples
        k_actual = min(k, len(self.file_params))
        if k_actual == 0:
            raise ValueError("No training data available to make a prediction.")

        sorted_indices = np.argsort(dists)
        idxs = sorted_indices[:k_actual]

        # Check if the closest neighbor is too far (optional warning/handling)
        if distance_threshold is not None and dists[idxs[0]] > distance_threshold:
            print(
                f"Warning: Closest neighbor distance ({dists[idxs[0]]:.4f}) exceeds threshold ({distance_threshold}). "
                f"Prediction might be unreliable for parameters {param_values}."
            )

        # Use the shape of the training data corresponding to the first nearest neighbor
        # as the reference shape for reconstruction (assuming consistency)
        if not self.train_data_shapes:
            raise ValueError(
                "Training data shapes not recorded during fit(). Cannot predict."
            )

        # Use the original (non-augmented) shape of training data for reconstruction
        ref_n_original, ref_mm1_original = self.train_data_shapes[
            idxs[0]
        ]  # Use shape from first nearest neighbor

        dt = 1.0  # Assuming original dt=1.0

        if output_interp and k_actual > 1:
            # Option 1: Interpolate outputs by averaging reconstructed curves
            curves = []
            for idx in idxs:
                Phi = self.Phi_list[idx]
                omega = self.omega_list[idx]
                b = self.b_list[idx]

                t_reconstruction = np.arange(ref_mm1_original) * dt
                time_dynamics = b[:, np.newaxis] * np.exp(
                    omega[:, np.newaxis] * t_reconstruction[np.newaxis, :]
                )
                Xdmd_full_aug = Phi @ time_dynamics
                Xdmd = Xdmd_full_aug[
                    :ref_n_original, :
                ]  # Trim back to original dimensions
                curves.append(Xdmd)

            Xdmd_predicted = np.mean(curves, axis=0)  # Average the reconstructed curves

            # For consistency, return averaged DMD components even if output is interpolated
            Phi_avg = np.mean([self.Phi_list[i] for i in idxs], axis=0)
            omega_avg = np.mean([self.omega_list[i] for i in idxs], axis=0)
            D_avg = np.mean([self.D_list[i] for i in idxs], axis=0)
            b_avg = np.mean([self.b_list[i] for i in idxs], axis=0)

            return (Phi_avg, omega_avg, D_avg, b_avg, Xdmd_predicted, t_reconstruction)

        else:
            # Option 2: Average DMD components (Phi, omega, D, b) from k nearest neighbors
            # (or just use the single nearest neighbor if k=1 or output_interp is False)
            Phi_avg = np.mean([self.Phi_list[i] for i in idxs], axis=0)
            omega_avg = np.mean([self.omega_list[i] for i in idxs], axis=0)
            D_avg = np.mean([self.D_list[i] for i in idxs], axis=0)
            b_avg = np.mean([self.b_list[i] for i in idxs], axis=0)

            t_reconstruction = (
                np.arange(ref_mm1_original) * dt
            )  # Time points for prediction
            time_dynamics = b_avg[:, np.newaxis] * np.exp(
                omega_avg[:, np.newaxis] * t_reconstruction[np.newaxis, :]
            )
            Xdmd_full_aug = Phi_avg @ time_dynamics
            Xdmd_predicted = Xdmd_full_aug[
                :ref_n_original, :
            ]  # Trim back to original dimensions

            return (Phi_avg, omega_avg, D_avg, b_avg, Xdmd_predicted, t_reconstruction)


# The get_param_from_filename function is removed as its functionality is now
# handled directly within ParametricSLM.fit() and _extract_params_from_tov_filename.


def is_on_boundary(param, param_min, param_max, tolerance=1e-5):
    """Checks if a parameter set is on the boundary of the parameter space."""
    for i in range(len(param)):
        if (
            abs(param[i] - param_min[i]) < tolerance
            or abs(param[i] - param_max[i]) < tolerance
        ):
            return True
    return False
