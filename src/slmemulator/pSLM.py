"""Parametric SLM built on Banach-GRIM kernel interpolation.

Fits directly on TOV data files (columns: radius, central pressure, mass
[, k2]) computed at known EOS parameters, and predicts the log-curves at new
parameter values by kernel interpolation of the full curve matrices across
min-max-normalized parameter space.

This replaces the earlier k-nearest-neighbour version: k-NN averaging of DMD
components is piecewise constant in parameter space and ill-posed for the
eigenpairs (mode ordering/sign ambiguity), whereas kernel interpolation of
the curves is exact at the training points and smooth in between. The DMD
components returned by predict() are computed *from the predicted curve*, so
they are always self-consistent.

For an end-to-end emulator that also generates the training data (EOS
generation + TOV solves), see slmemulator.TOVEmulator.
"""

from pathlib import Path

import numpy as np

from .banach_grim import BanachGRIMInterpolator
from .SLM import SLM

__all__ = ["ParametricSLM", "gaussian_kernel", "is_on_boundary"]


def gaussian_kernel(x1, x2, sigma=1.0):
    """Gaussian kernel between two parameter vectors (kept for backward
    compatibility; the interpolation itself uses banach_grim.rbf_kernel)."""
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    dist_sq = np.sum((x1_arr - x2_arr) ** 2)
    return np.exp(-dist_sq / (2 * sigma**2 + 1e-9))


class ParametricSLM:
    """
    Parametric SLM over a set of TOV data files.

    Example:
        pslm = ParametricSLM(fileList, filePath, tidal=True,
                             params=[[300, 0.1], [300, 0.3], ...])
        pslm.fit()
        Phi, omega, eigs, b, Xdmd, t = pslm.predict([400.0, 0.2])

    Parameters
    ----------
    fileList : sequence of str or Path
        TOV data files; each contains columns radius (km), central pressure
        (MeV/fm^3), mass (M_sun) and, if tidal, k2. All files must share the
        same number of rows.
    filePath : str or Path, optional
        Directory prepended to relative file names.
    tidal : bool
        Whether the files carry a 4th (k2) column to be used. Default False.
    params : array-like (n_files, n_params), optional
        EOS parameters of each file, in the same order as fileList. If not
        given, parameters are parsed from the file names (every underscore-
        separated token of the stem that parses as a float) — prefer passing
        them explicitly, since file-name parsing cannot recover values whose
        decimal points were replaced by underscores.
    reg : float
        Tikhonov regularization of the kernel interpolant. Default 1e-10.
    length_scale : float, optional
        RBF length scale in normalized parameter units (default: median
        pairwise distance of the normalized training parameters).
    error_threshold, max_r :
        Passed to SLM() when computing DMD components of predicted curves.
    """

    def __init__(
        self,
        fileList,
        filePath=None,
        tidal=False,
        params=None,
        reg=1e-10,
        length_scale=None,
        error_threshold=1e-6,
        max_r=None,
    ):
        base = Path(filePath) if filePath is not None else None
        self.fileList = [
            (base / f if base is not None and not Path(f).is_absolute() else Path(f))
            for f in fileList
        ]
        self.tidal = bool(tidal)
        self.reg = float(reg)
        self.length_scale = length_scale
        self.error_threshold = error_threshold
        self.max_r = max_r

        self._given_params = None if params is None else np.atleast_2d(
            np.asarray(params, dtype=float)
        )

        # Fitted state
        self.params = None            # (n_files, n_params)
        self.curves_log = None        # (n_files, n_quantities, n_points)
        self.n = None                 # number of quantities
        self.mm1 = None               # number of points per curve
        self._grim = None
        self._p_min = None
        self._p_range = None

    @staticmethod
    def params_from_filename(file_path):
        """Extract EOS parameters from a file name: every underscore-separated
        token of the stem that parses as a float (e.g.
        'MR_3.00e+02_1.00e-01.txt' -> [300.0, 0.1])."""
        values = []
        for token in Path(file_path).stem.split("_"):
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    def _load_curve(self, file_path):
        """Load one TOV data file -> log-curve matrix (n_quantities, n_points)."""
        data = np.loadtxt(file_path).T
        n_quantities = 4 if self.tidal else 3
        if data.shape[0] < n_quantities:
            raise ValueError(
                f"{file_path}: expected at least {n_quantities} columns, "
                f"got {data.shape[0]}."
            )
        eps = 1e-12  # guard against log(0)
        return np.log(np.abs(data[:n_quantities]) + eps)

    def _scale(self, params):
        return (np.asarray(params, dtype=float) - self._p_min) / self._p_range

    def fit(self):
        """Load all training files and fit the kernel interpolant."""
        curves, params = [], []
        for i, file_path in enumerate(self.fileList):
            if not file_path.exists():
                print(f"Skipping non-existent file: {file_path}")
                continue
            curve = self._load_curve(file_path)
            if not np.all(np.isfinite(curve)):
                print(f"Skipping {file_path.name}: non-finite values.")
                continue
            if curves and curve.shape != curves[0].shape:
                print(
                    f"Skipping {file_path.name}: inconsistent shape "
                    f"{curve.shape} (expected {curves[0].shape})."
                )
                continue
            curves.append(curve)
            if self._given_params is not None:
                params.append(self._given_params[i])
            else:
                parsed = self.params_from_filename(file_path)
                if not parsed:
                    raise ValueError(
                        f"Could not parse parameters from '{file_path.name}'; "
                        "pass params= explicitly."
                    )
                params.append(parsed)

        if not curves:
            raise RuntimeError("No valid training files processed.")

        self.curves_log = np.asarray(curves, dtype=float)
        self.params = np.asarray(params, dtype=float)
        self.n, self.mm1 = self.curves_log.shape[1], self.curves_log.shape[2]

        self._p_min = self.params.min(axis=0)
        self._p_range = self.params.max(axis=0) - self._p_min
        self._p_range[self._p_range == 0.0] = 1.0

        self._grim = BanachGRIMInterpolator(
            reg=self.reg, length_scale=self.length_scale, center=True
        )
        self._grim.fit(list(self.curves_log), self._scale(self.params), tol=1e-12)
        return self

    def predict_log(self, theta):
        """Interpolated log-curve matrix (n_quantities, n_points) at theta."""
        if self._grim is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self._grim.evaluate(self._scale(np.atleast_1d(theta)))

    def predict_curves(self, theta):
        """Interpolated curves in linear units (n_quantities, n_points)."""
        return np.exp(self.predict_log(theta))

    def predict(self, theta, dt=1.0):
        """
        Predict at parameter vector theta.

        Returns (Phi, omega, eigs, b, Xdmd, t) — the same shape of result as
        the earlier k-NN implementation, but the DMD components are computed
        from the kernel-interpolated curve, so they are self-consistent.
        Xdmd is the log-space SLM reconstruction; np.exp(Xdmd.real) gives the
        physical curves.
        """
        X_log = self.predict_log(theta)
        Phi, omega, eigs, b, Xdmd, _S, _r = SLM(
            X_log, dt=dt, error_threshold=self.error_threshold, max_r=self.max_r
        )
        t = np.arange(X_log.shape[1]) * dt
        return Phi, omega, eigs, b, Xdmd, t


def is_on_boundary(param, param_min, param_max, tolerance=1e-5):
    """Checks if a parameter set is on the boundary of the parameter space."""
    for i in range(len(param)):
        if (
            abs(param[i] - param_min[i]) < tolerance
            or abs(param[i] - param_max[i]) < tolerance
        ):
            return True
    return False
