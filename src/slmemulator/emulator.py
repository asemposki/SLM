"""SLM emulator of the TOV solver.

Trains on full TOV solutions computed over a family of EOS parameters and
afterwards predicts the mass-radius (and tidal) curves at new parameter
values in fractions of a millisecond — the star log-extended emulation:
curves are log-transformed and interpolated across parameter space with the
Banach-GRIM greedy kernel interpolator.

Typical use:

    from slmemulator import TOVEmulator, quarkyonia_generator

    emu = TOVEmulator(quarkyonia_generator, work_dir="EOS_files/QUARKYONIA")
    emu.train(train_params)          # slow: one TOV solve per training point
    curves = emu.predict([400, 0.2]) # fast: no TOV solve
    emu.save("qeos_emulator.npz")    # persist; reload with TOVEmulator.load
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from .banach_grim import BanachGRIMInterpolator
from .SLM import SLM
from .TOV_class import TOV

__all__ = ["TOVEmulator", "quarkyonia_generator", "mseos_generator"]

QUANTITY_NAMES = ("Radius", "Central Pressure", "Mass", "k2")


def quarkyonia_generator(params, output_path):
    """EOS generator for the Quarkyonia family; params = (lambda, kappa)."""
    from .EOS_Codes import generate_quarkyonia_eos

    generate_quarkyonia_eos(params[0], params[1], output_path)


def mseos_generator(params, output_path):
    """EOS generator for the Muller-Serot family; params = (Ls, Lv, zeta, xi)."""
    from .EOS_Codes import MSEOS

    MSEOS().main(params[0], params[1], params[2], params[3], out_file=output_path)


class TOVEmulator:
    """
    Emulator of the TOV solver over an EOS parameter family.

    `train` generates the EOS and solves the TOV equations at each training
    parameter point (expensive, ~20 s per point) and fits a Banach-GRIM
    kernel interpolant to the log-transformed curves over min-max-normalized
    parameters. `predict` then maps parameters -> curves directly, like
    pSLM's predict, at a ~10^5x speedup over the solver.

    Parameters
    ----------
    eos_generator : callable(params, output_path)
        Writes an EOS file (columns nb, E, P[, cs2]) for a parameter vector.
        `quarkyonia_generator` and `mseos_generator` are provided.
    work_dir : str or Path
        Directory where generated EOS files are written.
    param_names : sequence of str, optional
        Names of the parameters (used in reports).
    tidal : bool
        Emulate the tidal Love number k2 as well. Default True.
    sol_pts : int
        Integration steps for the TOV solver. Default 4000.
    reg, length_scale : float
        Banach-GRIM regularization and RBF length scale (None = median
        pairwise distance of the normalized training parameters).
    """

    def __init__(
        self,
        eos_generator,
        work_dir="EOS_files/emulator",
        param_names=None,
        tidal=True,
        sol_pts=4000,
        reg=1e-10,
        length_scale=None,
    ):
        self.eos_generator = eos_generator
        self.work_dir = Path(work_dir)
        self.param_names = list(param_names) if param_names else None
        self.tidal = bool(tidal)
        self.sol_pts = int(sol_pts)
        self.reg = float(reg)
        self.length_scale = length_scale

        self.n_quantities = 4 if self.tidal else 3
        self.quantity_names = QUANTITY_NAMES[: self.n_quantities]

        # Fitted state
        self.train_params = None
        self.train_curves_log = None  # (n_train, n_quantities, n_points)
        self._grim = None
        self._p_min = None
        self._p_range = None
        self.train_data_time_ = None
        self.fit_time_ = None

    # ------------------------------------------------------------------
    # Full-physics route (EOS generation + TOV solve)
    # ------------------------------------------------------------------
    def solve_log(self, params, tag=""):
        """Generate the EOS and solve the TOV equations for one parameter
        vector; returns the log-curve matrix (n_quantities, n_points)."""
        if self.eos_generator is None:
            raise RuntimeError(
                "No eos_generator available (emulator loaded without one)."
            )
        params = np.atleast_1d(np.asarray(params, dtype=float))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        name = "EOS" + tag + "_" + "_".join(f"{v:.6e}" for v in params) + ".txt"
        eos_path = self.work_dir / name
        self.eos_generator(params, eos_path)

        tov = TOV(eos_path, tidal=self.tidal, sol_pts=self.sol_pts)
        results = tov.tov_routine()
        return np.log(np.asarray(results[: self.n_quantities], dtype=np.float64))

    def solve(self, params, tag=""):
        """Full-physics curves in linear units (n_quantities, n_points)."""
        return np.exp(self.solve_log(params, tag=tag))

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------
    def _scale(self, params):
        return (np.asarray(params, dtype=float) - self._p_min) / self._p_range

    def train(self, train_params, curves_log=None):
        """
        Train on a set of parameter vectors, shape (n_train, n_params).

        If `curves_log` (matching list/array of log-curve matrices) is given,
        the expensive TOV solves are skipped and those curves are used
        directly.
        """
        train_params = np.atleast_2d(np.asarray(train_params, dtype=float))

        t0 = perf_counter()
        if curves_log is None:
            curves_log = [self.solve_log(p) for p in train_params]
        self.train_data_time_ = perf_counter() - t0

        self.train_params = train_params
        self.train_curves_log = np.asarray(curves_log, dtype=float)

        self._p_min = train_params.min(axis=0)
        self._p_range = train_params.max(axis=0) - self._p_min
        self._p_range[self._p_range == 0.0] = 1.0  # constant parameter dims

        t0 = perf_counter()
        self._grim = BanachGRIMInterpolator(
            reg=self.reg, length_scale=self.length_scale, center=True
        )
        self._grim.fit(
            list(self.train_curves_log), self._scale(train_params), tol=1e-12
        )
        self.fit_time_ = perf_counter() - t0
        return self

    def predict_log(self, params):
        """Emulated log-curve matrix (n_quantities, n_points) at `params`."""
        if self._grim is None:
            raise RuntimeError("Emulator is not trained; call train() first.")
        return self._grim.evaluate(self._scale(np.atleast_1d(params)))

    def predict(self, params):
        """Emulated curves in linear units: radius (km), central pressure
        (MeV/fm^3), mass (M_sun) and, if tidal, k2."""
        return np.exp(self.predict_log(params))

    def slm(self, params, dt=1, error_threshold=1e-6, max_r=None):
        """Run the SLM/DMD decomposition on the emulated curve at `params`.
        Returns the SLM tuple (Phi, omega, eigs, b, Xdmd, S, r)."""
        return SLM(
            self.predict_log(params),
            dt=dt,
            error_threshold=error_threshold,
            max_r=max_r,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """Save the trained state (parameters and curves) to an .npz file."""
        if self.train_params is None:
            raise RuntimeError("Nothing to save; call train() first.")
        np.savez(
            path,
            train_params=self.train_params,
            train_curves_log=self.train_curves_log,
            tidal=self.tidal,
            sol_pts=self.sol_pts,
            reg=self.reg,
            length_scale=(
                np.nan if self.length_scale is None else float(self.length_scale)
            ),
            param_names=np.array(
                self.param_names if self.param_names else [], dtype=str
            ),
        )

    @classmethod
    def load(cls, path, eos_generator=None, work_dir="EOS_files/emulator"):
        """Rebuild a trained emulator from an .npz file written by save().
        `eos_generator` is only needed if solve()/solve_log() will be used."""
        data = np.load(path, allow_pickle=False)
        length_scale = float(data["length_scale"])
        param_names = [str(n) for n in data["param_names"]] or None
        emulator = cls(
            eos_generator,
            work_dir=work_dir,
            param_names=param_names,
            tidal=bool(data["tidal"]),
            sol_pts=int(data["sol_pts"]),
            reg=float(data["reg"]),
            length_scale=None if np.isnan(length_scale) else length_scale,
        )
        emulator.train(data["train_params"], curves_log=data["train_curves_log"])
        return emulator
