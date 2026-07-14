"""End-to-end SLM driver: generate (or load) an EOS, solve the TOV equations,
run the SLM/DMD emulator and save results and plots.

Run non-interactively with flags, e.g.:
    python testSLM.py --general --eos apr_eos.table
    python testSLM.py --parametric --quarkyonia
Run without arguments for the interactive prompts.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from slmemulator import (
    MSEOS,
    SLM,
    TOV,
    clean_directory,
    generate_quarkyonia_eos,
    get_paths,
    plot_eigs,
    plot_slm_rad,
)

# parameter grids for the parametric runs
lamVal = np.linspace(300, 500, 2)
kappaVal = np.linspace(0.1, 0.3, 2)
Ls = np.linspace(0.0, 3e-3, 4)
Lv = np.linspace(0.0, 3e-2, 4)
zetaVal = np.linspace(1e-4, 2e-4, 2)
xiVal = np.linspace(0.0, 1.0, 2)


def _get_boolean_input(prompt: str, default_value: bool) -> bool:
    """Helper function to handle boolean user input with robust parsing."""
    user_input = input(f"{prompt} (True/False)[{default_value}]: ").strip().lower()
    if not user_input:
        return default_value
    return user_input in ("true", "t", "1")


def make_folders(parametric: bool, mseos: bool, eos_name: str | None = None):
    """
    Creates the necessary output directories and returns (paths, eos_name).
    """
    if parametric:
        eos_name = "MSEOS" if mseos else "Quarkyonia"
    elif eos_name is None:
        eos_name = (
            input("Insert EOS file name (e.g., MSEOS.dat, QEOS.txt, apr_eos.table): ")
            or "apr_eos.table"
        )

    paths = get_paths(
        output_base_dir=Path.cwd(), eos_name=eos_name, is_parametric_run=parametric
    )

    dirs_to_create = [
        paths["results_dir"],
        paths["current_eos_input_dir"],
        paths["user_tov_data_dir"],
        paths["plots_dir"],
        paths["current_slm_results_dir"],
        paths["current_tov_data_dir"],
        paths["current_slm_plots_dir"],
    ]
    if parametric:
        dirs_to_create.append(paths["tests_dir"])
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    return paths, eos_name


def run_slm_for_eos(eos_path: Path, label: str, tidal: bool, paths: dict) -> None:
    """
    Solve the TOV equations for one EOS file, run the SLM emulator on the
    log-transformed results, and save the MR data, SLM reconstruction and
    plots.

    Parameters:
        eos_path: Path to the EOS data file.
        label: Base name (with extension) used for the output files.
        tidal: Whether to include the tidal Love number k2.
        paths: Path dictionary from get_paths().
    """
    tov = TOV(eos_path, tidal=tidal)
    results = tov.tov_routine()

    # results columns: radius (km), central pressure (MeV/fm^3), mass (M_sun)
    # and, if tidal, the Love number k2
    n_quantities = 4 if tidal else 3
    mr_results = np.asarray(results[:n_quantities], dtype=np.float64)
    header = "Radius (km) Central Pressure (MeV/fm^3) Mass (M_sun)"
    ylabels = [
        r"Central Pressure $({\rm{MeV/fm}}^3)$",
        r"Mass $({\rm{M}}_{\odot})$",
    ]
    if tidal:
        header += " k2"
        ylabels.append(r"Tidal Deformability $k_2$")

    stem = label.removesuffix(".txt")
    np.savetxt(
        paths["current_tov_data_dir"] / f"MR_{stem}.txt",
        mr_results.T,
        header=header,
        delimiter="\t",
    )

    X = np.log(mr_results)
    slm_results = SLM(X, dt=1, error_threshold=1e-6)
    Xdmd = slm_results[4]
    eigs = slm_results[2]

    results_path = paths["current_slm_results_dir"] / f"SLM_{stem}.txt"
    np.savetxt(
        results_path,
        np.exp(Xdmd.real).T,
        header=header,
        delimiter="\t",
    )

    plot_eigs(eigs, output_path=paths["current_slm_plots_dir"] / f"SLM_{label}.png")
    plot_slm_rad(
        X,
        Xdmd,
        fileNames_for_labels=[f"SLM_{stem}", f"Eigs_{stem}"],
        plots_output_base_path=paths["current_slm_plots_dir"],
        ylabels=ylabels,
    )


def main(parametric: bool, tidal: bool, mseos: bool, paths: dict, eos_name: str):
    """Main function to orchestrate the process."""
    eos_output_dir = paths["current_eos_input_dir"]

    if parametric:
        print("Running Parametric DMD...")
        if mseos:
            mseos_instance = MSEOS()
            for ls in Ls:
                for lv in Lv:
                    for zeta in zetaVal:
                        for xi in xiVal:
                            print(ls, lv, zeta, xi)
                            file_name = f"mseos_{ls:.2e}_{lv:.2e}_{zeta:.2e}_{xi:.2e}.txt"
                            eos_path = eos_output_dir / file_name
                            mseos_instance.main(ls, lv, zeta, xi, eos_path)
                            run_slm_for_eos(eos_path, file_name, tidal, paths)
        else:
            for lam in lamVal:
                for kappa in kappaVal:
                    file_name = (
                        f"QEOS_{f'{lam:.2e}'.replace('.', '_')}"
                        f"_{f'{kappa:.2e}'.replace('.', '_')}.txt"
                    )
                    eos_path = eos_output_dir / file_name
                    generate_quarkyonia_eos(lam, kappa, eos_path)
                    print(file_name)
                    run_slm_for_eos(eos_path, file_name, tidal, paths)
    else:
        print("Running General SLM for file:", eos_name)
        eos_path = paths["package_eos_data_dir"] / eos_name
        run_slm_for_eos(eos_path, eos_name, tidal, paths)
        print(f"File Name: {eos_name}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--parametric", action="store_true", help="run the parametric (pSLM) pipeline"
    )
    mode.add_argument(
        "--general", action="store_true", help="run the general SLM pipeline"
    )
    eos = parser.add_mutually_exclusive_group()
    eos.add_argument(
        "--mseos", action="store_true", help="parametric runs: use the MSEOS EOS"
    )
    eos.add_argument(
        "--quarkyonia",
        action="store_true",
        help="parametric runs: use the Quarkyonia EOS",
    )
    parser.add_argument(
        "--eos",
        default=None,
        help="general runs: EOS file name from the packaged EOS_Data "
        "(e.g. apr_eos.table)",
    )
    tidal = parser.add_mutually_exclusive_group()
    tidal.add_argument(
        "--tidal", dest="tidal", action="store_true", default=None,
        help="include tidal deformability (default)",
    )
    tidal.add_argument("--no-tidal", dest="tidal", action="store_false")
    parser.add_argument(
        "--clean", action="store_true", help="clean output directories first"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    interactive = len(sys.argv) == 1

    if interactive:
        cleanup = _get_boolean_input("Clean up the directories?", default_value=False)
        parametric = _get_boolean_input("Parametric or General DMD?", default_value=True)
        tidal = _get_boolean_input("Tidal or Non-Tidal?", default_value=True)
        mseos = (
            _get_boolean_input("MSEOS or Quarkyonia?", default_value=True)
            if parametric
            else False
        )
        eos_name = None
    else:
        cleanup = args.clean
        parametric = not args.general
        tidal = True if args.tidal is None else args.tidal
        mseos = not args.quarkyonia
        eos_name = args.eos

    if cleanup:
        clean_directory()

    all_paths, eos_name = make_folders(parametric, mseos, eos_name)
    main(parametric, tidal, mseos, all_paths, eos_name)
