import numpy as np
import os
import sys
import time
import argparse
import json
from itertools import combinations_with_replacement
from plotData import plot_parametric_old
from sklearn.preprocessing import StandardScaler
import random

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
    TEST_DATA_PATH,
)


def gaussian_kernel(x1, x2, sigma=1.0):
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    dist_sq = np.sum((x1_arr - x2_arr) ** 2)
    return np.exp(-dist_sq / (2 * sigma**2 + 1e-9))


class ParametricDMD:
    def __init__(
        self, fileList, filePath, tidal=False, error_threshold=1e-4, max_r=None
    ):
        self.fileList = fileList
        self.filePath = filePath
        self.tidal = tidal
        self.error_threshold = error_threshold
        self.max_r = max_r
        self.n = None
        self.mm1 = None

    @staticmethod
    def augment_data(X):
        n, m = X.shape
        out = [X]
        for i, j in combinations_with_replacement(range(n), 2):
            out.append(X[i] * X[j])
        # for i, j, k in combinations_with_replacement(range(n), 3):
        #     out.append(X[i] * X[j] * X[k])
        return np.vstack(out)

    def _SLM_auto_r(self, X, dt):
        n = X.shape[0]
        X_aug = self.augment_data(X)
        X1, X2 = X_aug[:, :-1], X_aug[:, 1:]
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        max_r = min(X1.shape) if self.max_r is None else min(self.max_r, min(X1.shape))
        best = {"err": float("inf")}
        for r in range(1, max_r + 1):
            U_r, S_r, V_r = U[:, :r], S[:r], Vt[:r, :]
            S_r[S_r < 1e-12] = 1e-12
            S_r_inv = np.diag(1.0 / S_r)
            Atilde = U_r.T @ X2 @ V_r.T @ S_r_inv
            if not np.all(np.isfinite(Atilde)):
                continue
            try:
                D, W = np.linalg.eig(Atilde)
            except np.linalg.LinAlgError:
                continue
            Phi = X2 @ V_r.T @ S_r_inv @ W
            omega = np.log(D) / dt
            b = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]
            t = np.arange(X1.shape[1] + 1) * dt
            time_dynamics = b[:, np.newaxis] * np.exp(
                omega[:, np.newaxis] * t[np.newaxis, :]
            )
            Xdmd = (Phi @ time_dynamics)[:n, :]
            err = np.max(np.abs(X - Xdmd))
            if err < best["err"]:
                best = dict(
                    r=r,
                    D=D,
                    U=U_r,
                    W=W,
                    V=V_r,
                    S=S[:r],
                    b=b,
                    Atilde=Atilde,
                    Phi=Phi,
                    omega=omega,
                    Xdmd=Xdmd,
                    err=err,
                )
            if err <= self.error_threshold:
                break
        return best

    def fit(self):
        params, results, valid_files, dmd_errors = [], [], [], []
        for file in self.fileList:
            data = np.loadtxt(os.path.join(self.filePath, file)).T
            if not np.all(np.isfinite(data)):
                print(f"Skipping {file}: contains NaN or Inf")
                continue
            name_parts = file.strip("MR.txt").split("_")
            params.append([float(x) for x in name_parts[2:]])
            # X = np.log(data[:4]) if self.tidal else np.log(data[:3])
            # Modified lines:
            epsilon = 1e-9  # Define a small constant
            X = np.log(data[:4] + epsilon) if self.tidal else np.log(data[:3] + epsilon)
            X = np.asarray(X, dtype=np.float64)
            dt = 1.0
            result = self._SLM_auto_r(X, dt)
            results.append(result)
            valid_files.append(file)
            dmd_errors.append(result["err"])
            if self.n is None:
                self.n, self.mm1 = X.shape[0], X.shape[1]
        if self.n is None or self.mm1 is None:
            raise RuntimeError(
                "No valid training files found! Please check your input data."
            )
        self.fileList = valid_files
        self.params = np.array(params)
        self.param_scaler = StandardScaler().fit(self.params)
        self.scaled_params = self.param_scaler.transform(self.params)
        self.r_vals = np.array([res["r"] for res in results])
        self.augmented_n = self.augment_data(np.zeros((self.n, 2))).shape[0]
        self.D_list = [res["D"] for res in results]
        self.omega_list = [res["omega"] for res in results]
        self.b_list = [res["b"] for res in results]
        self.U_list = [res["U"] for res in results]
        self.W_list = [res["W"] for res in results]
        self.Phi_list = [res["Phi"] for res in results]
        self.train_results = results
        self.dmd_errors = dmd_errors
        # Error reporting:
        print("\nDMD fit errors for training files:")
        for file, err in zip(self.fileList, self.dmd_errors):
            print(f"{file}: max abs error = {err:.4e}")

    def pure_nn_interp(self, theta, feature_list):
        scaled_theta = self.param_scaler.transform([theta])[0]
        dists = np.linalg.norm(self.scaled_params - scaled_theta, axis=1)
        idx = np.argmin(dists)
        return feature_list[idx], idx, dists[idx]

    def predict(self, theta, k=1, output_interp=False, distance_threshold=None):
        theta = np.asarray(theta, dtype=np.float64)
        scaled_theta = self.param_scaler.transform([theta])[0]
        dists = np.linalg.norm(self.scaled_params - scaled_theta, axis=1)
        idxs = np.argsort(dists)[:k]
        # Optionally use output interpolation if requested or for far-away points
        use_interp = output_interp
        if distance_threshold is not None and dists[idxs[0]] > distance_threshold:
            use_interp = True
        if use_interp and k > 1:
            # Interpolate outputs (average the DMD curves)
            curves = []
            for idx in idxs:
                D = self.D_list[idx]
                omega = self.omega_list[idx]
                b = self.b_list[idx]
                Phi = self.Phi_list[idx]
                t = np.arange(self.mm1) * 1.0
                time_dynamics = b[:, np.newaxis] * np.exp(
                    omega[:, np.newaxis] * t[np.newaxis, :]
                )
                Xdmd = Phi @ time_dynamics
                curves.append(Xdmd)
            Xdmd_avg = np.mean(curves, axis=0)
            # Return first neighbor's other features for compatibility
            idx0 = idxs[0]
            return (
                self.Phi_list[idx0],
                self.omega_list[idx0],
                self.D_list[idx0],
                self.b_list[idx0],
                Xdmd_avg,
                t,
            )
        else:
            # Pure NN
            D = self.D_list[idxs[0]]
            omega = self.omega_list[idxs[0]]
            b = self.b_list[idxs[0]]
            Phi = self.Phi_list[idxs[0]]
            t = np.arange(self.mm1) * 1.0
            time_dynamics = b[:, np.newaxis] * np.exp(
                omega[:, np.newaxis] * t[np.newaxis, :]
            )
            Xdmd = Phi @ time_dynamics
            return Phi, omega, D, b, Xdmd, t


def get_param_from_filename(filename):
    """Extracts parameters from the filename."""
    name_parts = filename.strip("MR.txt").split("_")
    return [float(x) for x in name_parts[2:]]


def is_on_boundary(param, param_min, param_max, tolerance=1e-5):
    """Checks if a parameter set is on the boundary of the parameter space."""
    for i in range(len(param)):
        if (
            abs(param[i] - param_min[i]) < tolerance
            or abs(param[i] - param_max[i]) < tolerance
        ):
            return True
    return False


def main(
    tidal=False,
    mseos=False,
    error_threshold=1e-4,
    max_r=None,
    k=1,
    output_interp=False,
    distance_threshold=None,
    num_train_total=20,  # Total number of training samples
    num_boundary_train=10,  # Number of boundary samples for training
):
    tov_data_path = os.path.join(TOV_PATH, "MSEOS" if mseos else "QEOS")
    all_files = [f for f in os.listdir(tov_data_path) if f.endswith(".txt")]

    # Extract all parameters to determine the min/max for boundary definition
    all_params = [get_param_from_filename(f) for f in all_files]
    all_params_np = np.array(all_params)
    param_min = np.min(all_params_np, axis=0)
    param_max = np.max(all_params_np, axis=0)

    boundary_files = []
    inside_files = []

    for f in all_files:
        param = get_param_from_filename(f)
        if is_on_boundary(param, param_min, param_max):
            boundary_files.append(f)
        else:
            inside_files.append(f)

    # Ensure we don't request more samples than available
    num_boundary_train = min(num_boundary_train, len(boundary_files))
    num_inside_train = min(num_train_total - num_boundary_train, len(inside_files))

    if num_boundary_train + num_inside_train < num_train_total:
        print(
            f"Warning: Not enough files to select {num_train_total} training samples with {num_boundary_train} boundary and {num_inside_train} inside. Adjusting total training samples."
        )
        num_train_total = num_boundary_train + num_inside_train

    # Randomly sample from boundary and inside files
    train_files_boundary = random.sample(boundary_files, num_boundary_train)
    train_files_inside = random.sample(inside_files, num_inside_train)
    train_files = train_files_boundary + train_files_inside
    random.shuffle(train_files)  # Shuffle to mix boundary and inside files

    # The remaining files (not in train_files) can be considered for testing.
    # Filter out files that are already in train_files to avoid overlap.
    test_files = [f for f in all_files if f not in train_files]

    # Ensure test_files directory exists and copy test files
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    for file in test_files:
        src = os.path.join(tov_data_path, file)
        dst = os.path.join(TEST_DATA_PATH, file)
        if not os.path.exists(dst):
            try:
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
            except Exception as e:
                print(f"File copy failed for {file}: {e}")

    model = ParametricDMD(train_files, tov_data_path, tidal, error_threshold, max_r)
    model.fit()
    time_dict = {}
    for file in os.listdir(TEST_DATA_PATH):
        if "MR" in file:
            fileSplit = file.strip(".txt").split("_")
            testParam = [float(x) for x in fileSplit[2:]]
            starttime = time.time()
            # You can control these settings:
            # k = 3 (for 3-NN output interpolation), output_interp=True for smoother result
            phi, omega, lam, b, Xdmd, newT = model.predict(
                testParam,
                k=k,
                output_interp=output_interp,
                distance_threshold=distance_threshold,
            )
            stoptime = time.time()
            data = np.loadtxt(os.path.join(TEST_DATA_PATH, file))
            X = data.T
            name = "Data_" + "_".join(map(str, testParam))
            print(f"Names {name.split('_')}")
            print(f"Shape {Xdmd.shape} {X.shape}")
            if tidal:
                xlim = [(10, 25), (10, 25)]
                ylim = [(0, 3), (0, 0.15)]
            else:
                xlim = [(0, 25)]
                ylim = [(0, 3)]
            plot_parametric_old(Xdmd, X, name, tidal, xlim=xlim, ylim=ylim)
            print(f"plotted file: {name}")
            total_time = stoptime - starttime
            time_dict[file] = total_time
    with open(os.path.join(TEST_DATA_PATH, "slm_time_data.json"), "w") as file:
        json.dump(time_dict, file, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parametric DMD - Robust with Output Interpolation"
    )
    parser.add_argument("--tidal", action="store_true", help="Use tidal mode")
    parser.add_argument("--mseos", action="store_true", help="Use MSEOS (else QEOS)")
    parser.add_argument(
        "--error_threshold", type=float, default=1e-4, help="DMD error threshold"
    )
    parser.add_argument("--max_r", type=int, default=None, help="Max DMD rank")
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of neighbors to average (for output interpolation)",
    )
    parser.add_argument(
        "--output_interp",
        action="store_true",
        help="Interpolate output curves instead of pure NN",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=None,
        help="Only use output_interp if NN distance exceeds this threshold (optional)",
    )
    parser.add_argument(
        "--num_train_total",
        type=int,
        default=20,
        help="Total number of training data sets.",
    )
    parser.add_argument(
        "--num_boundary_train",
        type=int,
        default=10,
        help="Number of training data sets to pick from the boundary.",
    )
    args = parser.parse_args()
    main(
        args.tidal,
        args.mseos,
        args.error_threshold,
        args.max_r,
        args.k,
        args.output_interp,
        args.distance_threshold,
        args.num_train_total,
        args.num_boundary_train,
    )
