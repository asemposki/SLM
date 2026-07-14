"""Parametric SLM (pSLM) end-to-end test and benchmark.

Trains the SLM emulator of the TOV solver (slmemulator.TOVEmulator) on a
grid of EOS parameters and validates its predictions at held-out parameter
values against direct TOV solves. For every test point the script reports
the prediction accuracy and the wall time of both routes (EOS generation +
TOV solver vs emulator prediction), and writes the table to
Results/<EOS>/pSLM/pSLM_accuracy_timing_<eos>.txt.

The emulator interpolates the log-transformed TOV curve matrices across
min-max-normalized parameter space with the Banach-GRIM greedy kernel
interpolator; like pSLM's predict(theta), the trained emulator maps
parameters -> curves directly, with no TOV solve at prediction time.

Supported EOS families (parameter grids match tests/testSLM.py):
  * quarkyonia: lambda x kappa       (testSLM set: --eos quarkyonia --grid 2)
  * mseos:      Ls x Lv x zeta x xi  (4 x 4 x 2 x 2 = 64 training points)

Run e.g.:
    python testpSLM.py                          # Quarkyonia, 4x4 grid
    python testpSLM.py --grid 2                 # Quarkyonia, testSLM 2x2 grid
    python testpSLM.py --eos mseos              # MSEOS, testSLM grids (~25 min)
    python testpSLM.py --test-lam 370 --test-kappa 0.17   # single test point
"""

import argparse
import sys
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from slmemulator import (
    TOVEmulator,
    get_paths,
    mseos_generator,
    plot_slm_rad,
    quarkyonia_generator,
)

# MSEOS parameter grids (same sets as tests/testSLM.py)
MSEOS_GRIDS = {
    "Ls": np.linspace(0.0, 3e-3, 4),
    "Lv": np.linspace(0.0, 3e-2, 4),
    "zeta": np.linspace(1e-4, 2e-4, 2),
    "xi": np.linspace(0.0, 1.0, 2),
}


def cartesian(grids: list[np.ndarray]) -> np.ndarray:
    """Cartesian product of 1-D grids -> array (n_points, n_dims)."""
    return np.array(list(product(*grids)), dtype=float)


def midpoints(grid: np.ndarray) -> np.ndarray:
    """Midpoints of consecutive grid values (grid center for 2-pt grids)."""
    return 0.5 * (grid[:-1] + grid[1:])


def relative_errors(X_pred_log: np.ndarray, X_true_log: np.ndarray) -> np.ndarray:
    """Max relative error per quantity, computed in linear space."""
    pred, true = np.exp(X_pred_log), np.exp(X_true_log)
    return np.max(np.abs(pred - true) / np.abs(true), axis=1)


def build_problem(args):
    """Return (eos_generator, param_names, training grids) for the EOS family."""
    if args.eos == "mseos":
        return mseos_generator, list(MSEOS_GRIDS), list(MSEOS_GRIDS.values())
    lam_grid = np.linspace(*args.lam_range, args.grid)
    kappa_grid = np.linspace(*args.kappa_range, args.grid)
    return quarkyonia_generator, ["lambda", "kappa"], [lam_grid, kappa_grid]


def build_test_set(args, grids: list[np.ndarray]) -> np.ndarray:
    """Held-out test parameters: an explicit point if given (Quarkyonia only),
    otherwise the midpoints of the training grid cells (worst case for
    interpolation), optionally subsampled to --n-test points."""
    if args.eos == "quarkyonia" and (
        args.test_lam is not None or args.test_kappa is not None
    ):
        lam = args.test_lam if args.test_lam is not None else 370.0
        kappa = args.test_kappa if args.test_kappa is not None else 0.17
        return np.array([[lam, kappa]])

    test = cartesian([midpoints(g) for g in grids])
    if args.n_test is not None and args.n_test < len(test):
        rng = np.random.default_rng(args.seed)
        test = test[rng.choice(len(test), args.n_test, replace=False)]
    return test


def run_pslm_test(args) -> int:
    eos_label = "MSEOS" if args.eos == "mseos" else "Quarkyonia"
    paths = get_paths(
        output_base_dir=Path.cwd(), eos_name=eos_label, is_parametric_run=True
    )
    eos_dir = paths["current_eos_input_dir"]
    results_dir = paths["current_slm_results_dir"]
    plots_dir = paths["current_slm_plots_dir"]
    tests_dir = paths["current_slm_tests_dir"]
    for d in (eos_dir, results_dir, plots_dir, tests_dir):
        d.mkdir(parents=True, exist_ok=True)

    eos_generator, param_names, grids = build_problem(args)
    train_params = cartesian(grids)

    emulator = TOVEmulator(
        eos_generator,
        work_dir=eos_dir,
        param_names=param_names,
        tidal=args.tidal,
        sol_pts=args.sol_pts,
        reg=args.reg,
        length_scale=args.length_scale,
    )

    print(f"Training on {len(train_params)} {eos_label} EOS parameter points...")
    emulator.train(train_params)
    emulator.save(results_dir / f"emulator_{args.eos}.npz")
    print(
        f"Banach-GRIM selected {len(emulator._grim.selected_idx)}/{len(train_params)} "
        f"training points\n"
        f"one-time cost: training data {emulator.train_data_time_:.1f} s, "
        f"fit {emulator.fit_time_ * 1e3:.1f} ms"
    )

    # --- benchmark over the held-out test set ---
    test_params = build_test_set(args, grids)
    names = emulator.quantity_names
    rows = []
    curves = []  # (X_true, X_pred) per test point, for the worst-case plot

    for p in test_params:
        t0 = perf_counter()
        X_true = emulator.solve_log(p, tag="_test")
        t_tov = perf_counter() - t0

        t0 = perf_counter()
        X_pred = emulator.predict_log(p)
        t_grim = perf_counter() - t0

        errs = relative_errors(X_pred, X_true)
        rows.append([*p, *errs, t_tov, t_grim, t_tov / t_grim])
        curves.append((X_true, X_pred))

    table = np.asarray(rows)
    n_p = len(param_names)
    err_block = table[:, n_p : n_p + len(names)]
    worst_idx = int(np.argmax(err_block.max(axis=1)))
    worst_err = float(err_block[worst_idx].max())
    worst_p = test_params[worst_idx]
    X_true, X_pred = curves[worst_idx]

    # --- print the accuracy/timing table ---
    p_cols = "".join(f"{n[:10]:>11}" for n in param_names)
    err_cols = "".join(f"{('err ' + n)[:14]:>15}" for n in names)
    print(
        f"\n{p_cols}{err_cols}{'t_TOV [s]':>12}{'t_emu [ms]':>12}{'speedup':>10}"
    )
    for row in table.tolist():
        p_vals, rest = row[:n_p], row[n_p:]
        errs, t_tov, t_grim, speedup = rest[:-3], rest[-3], rest[-2], rest[-1]
        p_str = "".join(f"{v:>11.4g}" for v in p_vals)
        err_str = "".join(f"{e:>15.3e}" for e in errs)
        print(f"{p_str}{err_str}{t_tov:>12.2f}{t_grim * 1e3:>12.3f}{speedup:>10.0f}x")

    err_summary = "  ".join(
        f"{n}: mean {err_block[:, j].mean():.2e} / max {err_block[:, j].max():.2e}"
        for j, n in enumerate(names)
    )
    print(f"\nerror over {len(table)} test points:  {err_summary}")

    mean_t_tov = table[:, -3].mean()
    mean_t_grim = table[:, -2].mean()
    print(
        f"mean per prediction: TOV solver {mean_t_tov:.2f} s, "
        f"emulator {mean_t_grim * 1e3:.3f} ms "
        f"(~{mean_t_tov / mean_t_grim:.0f}x speedup)"
    )

    # --- save table and a plot of the worst-error test point ---
    table_path = results_dir / f"pSLM_accuracy_timing_{args.eos}.txt"
    np.savetxt(
        table_path,
        table,
        header=(
            "  ".join(param_names)
            + "  "
            + "  ".join(f"max_rel_err_{n.replace(' ', '_')}" for n in names)
            + "  t_tov_s  t_emulator_s  speedup"
        ),
        fmt="%.6e",
    )

    stem = ("pSLM_" + args.eos + "_" + "_".join(f"{v:.2e}" for v in worst_p)).replace(
        ".", "_"
    )
    header = "log " + "  log ".join(names)
    np.savetxt(tests_dir / f"{stem}_true.txt", X_true.T, header=header, delimiter="\t")
    np.savetxt(tests_dir / f"{stem}_emu.txt", X_pred.T, header=header, delimiter="\t")

    ylabels = [r"Central Pressure $({\rm{MeV/fm}}^3)$", r"Mass $({\rm{M}}_{\odot})$"]
    if args.tidal:
        ylabels.append(r"Tidal Deformability $k_2$")
    plot_slm_rad(
        X_true,
        X_pred,
        fileNames_for_labels=[stem],
        plots_output_base_path=plots_dir,
        ylabels=ylabels,
    )
    print(f"table: {table_path}\nworst-point curves/plots under {tests_dir} and {plots_dir}")

    # --- verdict ---
    worst_p_str = ", ".join(f"{n}={v:g}" for n, v in zip(param_names, worst_p))
    if worst_err > args.tol:
        print(
            f"FAIL: worst emulator relative error {worst_err:.3e} > "
            f"tol {args.tol:.1e} at ({worst_p_str})"
        )
        return 1
    print(f"PASS: worst emulator relative error {worst_err:.3e} <= tol {args.tol:.1e}")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos", choices=["quarkyonia", "mseos"], default="quarkyonia",
                        help="EOS family (mseos uses the testSLM.py Ls/Lv/zeta/xi grids)")
    parser.add_argument("--grid", type=int, default=4,
                        help="quarkyonia: training grid points per parameter "
                             "(2 reproduces the testSLM.py lamVal/kappaVal sets)")
    parser.add_argument("--lam-range", type=float, nargs=2, default=(300.0, 500.0))
    parser.add_argument("--kappa-range", type=float, nargs=2, default=(0.1, 0.3))
    parser.add_argument("--test-lam", type=float, default=None,
                        help="single held-out lambda (default: use grid-cell midpoints)")
    parser.add_argument("--test-kappa", type=float, default=None,
                        help="single held-out kappa (default: use grid-cell midpoints)")
    parser.add_argument("--n-test", type=int, default=None,
                        help="subsample the midpoint test set to this many points")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the test-point subsampling")
    tidal = parser.add_mutually_exclusive_group()
    tidal.add_argument("--tidal", dest="tidal", action="store_true", default=True)
    tidal.add_argument("--no-tidal", dest="tidal", action="store_false")
    parser.add_argument("--sol-pts", type=int, default=4000,
                        help="TOV integration steps (default 4000)")
    parser.add_argument("--reg", type=float, default=1e-10,
                        help="Banach-GRIM regularization (default 1e-10)")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="RBF length scale in normalized parameter units "
                             "(default: median pairwise distance)")
    parser.add_argument("--tol", type=float, default=0.15,
                        help="max allowed relative error for PASS (default 0.15; "
                             "the k2 curve near grid edges is the hardest quantity)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_pslm_test(parse_args(sys.argv[1:])))
