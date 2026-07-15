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

import h5py
import numpy as np

from .banach_grim import BanachGRIMInterpolator
from .SLM import SLM
from .TOV_class import TOV

__all__ = [
    "TOVEmulator",
    "EOSDrawEmulator",
    "quarkyonia_generator",
    "mseos_generator",
    "load_eos_draws",
    "load_tov_curves",
]

QUANTITY_NAMES = ("Radius", "Central Pressure", "Mass", "k2")

# canonical curve names <- keys used in the draw result files
_TOV_KEY_ALIASES = {
    "Radius": ("total_radius", "radius"),
    "Central Pressure": ("total_pres_central", "cpres"),
    "Mass": ("total_mass", "mass"),
    "k2": ("k2",),
    "Tidal Deformability": ("tidal_deformability", "tidal"),
}


def load_eos_draws(path):
    """
    Load EOS draws from an .h5 file (keys dens, edens, pressure) or an .npz
    file (keys density/dens, edens, pres/pressure).

    Returns:
        (nb, P, E): the shared density grid (n_grid,) and the pressure and
        energy-density draws, each shaped (n_grid, n_draws).
    """
    path = Path(path)
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            nb, P, E = f["dens"][:], f["pressure"][:], f["edens"][:]
    elif path.suffix == ".npz":
        data = np.load(path)
        nb = data["density"] if "density" in data.files else data["dens"]
        P = data["pres"] if "pres" in data.files else data["pressure"]
        E = data["edens"]
    else:
        raise ValueError(f"Unsupported EOS draw file: '{path.suffix}'.")

    nb = nb[:, 0] if nb.ndim > 1 else nb
    P = P[:, None] if P.ndim == 1 else P
    E = E[:, None] if E.ndim == 1 else E
    return nb, P, E


def load_tov_curves(path):
    """
    Load TOV result curves from an .h5/.npz file. Recognizes the key
    conventions total_radius/total_pres_central/total_mass/k2/
    tidal_deformability (per-draw solver output) and radius/cpres/mass/k2/
    tidal (reduced ensembles).

    Returns:
        (names, curves): canonical quantity names and an array shaped
        (n_quantities, n_points, n_draws).
    """
    path = Path(path)
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            data = {k: f[k][:] for k in f.keys()}
    elif path.suffix == ".npz":
        loaded = np.load(path)
        data = {k: loaded[k] for k in loaded.files}
    else:
        raise ValueError(f"Unsupported TOV results file: '{path.suffix}'.")

    names, curves = [], []
    for name, aliases in _TOV_KEY_ALIASES.items():
        for key in aliases:
            if key in data and np.asarray(data[key]).ndim == 2:
                names.append(name)
                curves.append(np.asarray(data[key], dtype=float))
                break
    if not curves:
        raise ValueError(f"{path}: no recognized TOV curve keys in {list(data)}.")
    return names, np.stack(curves, axis=0)


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


class EOSDrawEmulator:
    """
    Emulates the TOV solver over nonparametric EOS *draws* (e.g. Gaussian
    process samples): given the pressure/energy-density curves of a new draw
    on the shared density grid, predicts the TOV solution curves without
    integrating the TOV equations.

    Method: each draw is featurized as its log10 pressure (and optionally
    log10 energy density) on the density grid, compressed by PCA and
    whitened; the log-transformed TOV curves are then interpolated across
    this feature space with the Banach-GRIM kernel interpolator. Quantities
    with nonpositive entries in the training data (e.g. glitched k2 or
    Lambda cells) are emulated linearly instead of logarithmically.

    Typical use:

        emu = EOSDrawEmulator.from_files("eos_9_draws.h5", "tov_9_results.h5")
        curves = emu.predict(P_new, E_new)   # (n_quantities, n_points)

    Parameters
    ----------
    n_components : int or None
        Number of PCA components of the draw features (capped at
        n_draws - 1). Default None: keep the components whose singular
        value exceeds ``sv_tol`` times the largest one — too few components
        collapses distinct draws onto nearly coincident feature points and
        destabilizes the interpolation, while trailing noise components
        dilute the kernel distances.
    sv_tol : float
        Relative singular-value cutoff used when n_components is None.
        Default 0.05.
    reg : float
        Banach-GRIM regularization. Default 1e-10.
    length_scale : float or None
        RBF length scale in whitened feature space (None: median pairwise
        distance heuristic).
    use_edens : bool
        Include log10 energy density in the features alongside log10
        pressure. Default True.
    method : {"grim", "pgreedy"}
        Interpolation fitting method (see BanachGRIMInterpolator). For
        large ensembles (thousands of draws) use "pgreedy" with a
        max_basis_size cap; "grim" re-runs recombination per functional and
        becomes expensive beyond a few hundred draws. Default "grim".
    max_basis_size : int or None
        Cap on the number of support draws kept by the interpolator
        (None: all).
    """

    def __init__(self, n_components=None, sv_tol=0.05, reg=1e-10,
                 length_scale=None, use_edens=True, method="grim",
                 max_basis_size=None):
        self.n_components = None if n_components is None else int(n_components)
        self.sv_tol = float(sv_tol)
        self.reg = float(reg)
        self.length_scale = length_scale
        self.use_edens = bool(use_edens)
        self.method = method
        self.max_basis_size = max_basis_size

        # fitted state
        self.quantity_names = None
        self._n_grid = None
        self._f_mean = None       # (n_features,)
        self._components = None   # (k, n_features)
        self._sv = None           # (k,)
        self._log_mask = None     # (n_quantities,) bool: log-transformed?
        self._curves_t = None     # (n_draws, n_q, n_pts) transformed curves
        self._scores = None       # (n_draws, k)
        self._grim = None
        self.fit_time_ = None

    # ------------------------------------------------------------------
    # Feature pipeline
    # ------------------------------------------------------------------
    def _features(self, P, E):
        """Feature matrix (n_draws, n_features) from draws (n_grid, n_draws)."""
        if np.any(P <= 0) or (self.use_edens and np.any(E <= 0)):
            raise ValueError("EOS draws must have positive pressure/edens.")
        F = np.log10(P).T
        if self.use_edens:
            F = np.hstack([F, np.log10(E).T])
        return F

    def _scores_of(self, F):
        """Whitened PCA scores of feature rows."""
        return (F - self._f_mean) @ self._components.T / self._sv

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, P, E, curves, quantity_names=None):
        """
        Fit the emulator.

        Parameters:
            P, E (np.ndarray): Pressure and energy-density draws on the
                shared density grid, shaped (n_grid, n_draws).
            curves (np.ndarray): TOV result curves, shaped
                (n_quantities, n_points, n_draws).
            quantity_names (sequence of str, optional): names of the
                quantities in `curves`.
        """
        t0 = perf_counter()
        P = np.asarray(P, dtype=float)
        E = np.asarray(E, dtype=float)
        curves = np.asarray(curves, dtype=float)
        n_q, n_pts, n_draws = curves.shape
        if P.shape[1] != n_draws:
            raise ValueError(
                f"{P.shape[1]} EOS draws but {n_draws} TOV result draws."
            )
        self._n_grid = P.shape[0]
        self.quantity_names = (
            list(quantity_names) if quantity_names else
            [f"Quantity {q}" for q in range(n_q)]
        )

        # PCA of the draw features
        F = self._features(P, E)
        self._f_mean = F.mean(axis=0)
        _, S, Vt = np.linalg.svd(F - self._f_mean, full_matrices=False)
        rank = int(np.sum(S > 1e-10 * S[0]))
        if self.n_components is None:
            k_auto = int(np.sum(S > self.sv_tol * S[0]))
            k = max(1, min(k_auto, n_draws - 1, rank))
        else:
            k = max(1, min(self.n_components, n_draws - 1, rank))
        self._components = Vt[:k]
        self._sv = S[:k]
        self._scores = self._scores_of(F)

        # per-quantity transform: log where strictly positive, else linear
        self._log_mask = np.array([np.all(curves[q] > 0) for q in range(n_q)])
        curves_t = curves.copy()
        for q in range(n_q):
            if self._log_mask[q]:
                curves_t[q] = np.log(curves[q])
        # snapshots per draw: (n_q, n_pts)
        self._curves_t = np.moveaxis(curves_t, -1, 0)

        self._grim = BanachGRIMInterpolator(
            reg=self.reg, length_scale=self.length_scale, center=True
        )
        self._grim.fit(
            list(self._curves_t),
            self._scores,
            max_basis_size=self.max_basis_size,
            tol=1e-12,
            method=self.method,
        )
        self.fit_time_ = perf_counter() - t0
        return self

    @classmethod
    def from_files(cls, eos_file, tov_file, **kwargs):
        """Build and fit an emulator from an EOS draw file and the matching
        TOV results file (draw columns must correspond)."""
        _, P, E = load_eos_draws(eos_file)
        names, curves = load_tov_curves(tov_file)
        return cls(**kwargs).fit(P, E, curves, quantity_names=names)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, P_draw, E_draw=None):
        """
        Predict the TOV curves for one EOS draw.

        Parameters:
            P_draw (np.ndarray): Pressure of the draw on the same density
                grid used in training, shape (n_grid,) or (n_grid, 1).
            E_draw (np.ndarray): Energy density of the draw (required when
                the emulator was fit with use_edens=True).

        Returns:
            curves (np.ndarray): Predicted curves, (n_quantities, n_points),
                in the same (linear) units as the training data.
        """
        if self._grim is None:
            raise RuntimeError("Emulator is not fitted; call fit() first.")
        P_draw = np.asarray(P_draw, dtype=float).reshape(-1, 1)
        if P_draw.shape[0] != self._n_grid:
            raise ValueError(
                f"Draw has {P_draw.shape[0]} grid points; trained with "
                f"{self._n_grid}."
            )
        if self.use_edens:
            if E_draw is None:
                raise ValueError("E_draw is required (fit with use_edens=True).")
            E_draw = np.asarray(E_draw, dtype=float).reshape(-1, 1)
        score = self._scores_of(self._features(P_draw, E_draw))[0]

        out = self._grim.evaluate(score)
        for q in range(out.shape[0]):
            if self._log_mask[q]:
                out[q] = np.exp(out[q])
        return out

    def predict_many(self, P, E=None):
        """Predict for several draws at once; P (and E) shaped
        (n_grid, n_draws). Returns (n_quantities, n_points, n_draws)."""
        P = np.asarray(P, dtype=float)
        draws = [
            self.predict(P[:, j], None if E is None else np.asarray(E)[:, j])
            for j in range(P.shape[1])
        ]
        return np.stack(draws, axis=-1)

    def slm(self, P_draw, E_draw=None, dt=1, error_threshold=1e-6, max_r=None):
        """SLM/DMD decomposition of the emulated (log-transformed) curves."""
        curves = self.predict(P_draw, E_draw)
        X_log = np.where(
            self._log_mask[:, None], np.log(np.abs(curves) + 1e-300), curves
        )
        return SLM(X_log, dt=dt, error_threshold=error_threshold, max_r=max_r)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """Save the trained state to an .npz file."""
        if self._grim is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        np.savez(
            path,
            n_components=-1 if self.n_components is None else self.n_components,
            sv_tol=self.sv_tol,
            method=self.method,
            max_basis_size=(
                -1 if self.max_basis_size is None else int(self.max_basis_size)
            ),
            reg=self.reg,
            length_scale=(
                np.nan if self.length_scale is None else float(self.length_scale)
            ),
            use_edens=self.use_edens,
            quantity_names=np.array(self.quantity_names, dtype=str),
            n_grid=self._n_grid,
            f_mean=self._f_mean,
            components=self._components,
            sv=self._sv,
            log_mask=self._log_mask,
            curves_t=self._curves_t,
            scores=self._scores,
        )

    @classmethod
    def load(cls, path):
        """Rebuild a trained emulator from an .npz file written by save()."""
        data = np.load(path, allow_pickle=False)
        ls = float(data["length_scale"])
        n_comp = int(data["n_components"])
        mbs = int(data["max_basis_size"]) if "max_basis_size" in data.files else -1
        emulator = cls(
            n_components=None if n_comp < 0 else n_comp,
            sv_tol=float(data["sv_tol"]) if "sv_tol" in data.files else 0.05,
            reg=float(data["reg"]),
            length_scale=None if np.isnan(ls) else ls,
            use_edens=bool(data["use_edens"]),
            method=str(data["method"]) if "method" in data.files else "grim",
            max_basis_size=None if mbs < 0 else mbs,
        )
        emulator.quantity_names = [str(n) for n in data["quantity_names"]]
        emulator._n_grid = int(data["n_grid"])
        emulator._f_mean = data["f_mean"]
        emulator._components = data["components"]
        emulator._sv = data["sv"]
        emulator._log_mask = data["log_mask"]
        emulator._curves_t = data["curves_t"]
        emulator._scores = data["scores"]
        emulator._grim = BanachGRIMInterpolator(
            reg=emulator.reg, length_scale=emulator.length_scale, center=True
        )
        emulator._grim.fit(
            list(emulator._curves_t),
            emulator._scores,
            max_basis_size=emulator.max_basis_size,
            tol=1e-12,
            method=emulator.method,
        )
        return emulator
