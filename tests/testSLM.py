import os
import numpy as np
from pathlib import Path
from slmemulator import get_paths, clean_directory, generate_quarkyonia_eos, MSEOS
from slmemulator import TOV, SLM
from slmemulator import plot_eigs, plot_slm_rad
from slmemulator import ParametricSLM

# set parameters for lambda and kappa
lamVal = np.linspace(300, 500, 2)
kappaVal = np.linspace(0.1, 0.3, 2)
Ls = np.linspace(0.0, 3e-3, 4)
Lv = np.linspace(0.0, 3e-2, 4)
zetaVal = np.linspace(1e-4, 2e-4, 2)
xiVal = np.linspace(0.0, 1.0, 2)


def _get_boolean_input(prompt, default_value):
    """Helper function to handle boolean user input with robust parsing."""
    user_input = input(f"{prompt} (True/False)[{default_value}]: ").strip().lower()
    if not user_input:
        return default_value
    return user_input in ("true", "t", "1")

def make_folders(parametric: bool, initial_mseos_flag: bool) -> dict:
    """
    Creates necessary directories and returns the paths dictionary.
    """
    if parametric:
        # For parametric runs, the EOS name is fixed based on the user's choice (MSEOS or Quarkyonia)
        eos_name = "MSEOS" if initial_mseos_flag else "Quarkyonia"
    else:
        # For non-parametric (General) runs, ask the user for the specific EOS file name
        # We assume the user inputs the actual file name (e.g., "apr_eos.table")
        eos_name = input("Insert EOS file name (e.g., MSEOS.dat, QEOS.txt, apr_eos.table): ") or "apr_eos.table"
    paths = get_paths(output_base_dir=Path(os.getcwd()), eos_name=eos_name, is_parametric_run=parametric)

    # Correct unpacking of paths dictionary
    RESULTS_PATH = paths["results_dir"]
    EOS_DATA_DIR = paths["package_eos_data_dir"]  # EOS_data directory
    TOV_DATA_DIR = paths["user_tov_data_dir"]
    # TOV_DATA_DIR = paths["tov_data_dir"]  # Renamed for clarity from TOV_PATH
    PLOTS_PATH = paths["plots_dir"]
    CURRENT_SLM_RESULTS_DIR = paths["current_slm_results_dir"]
    CURRENT_TOV_DATA_DIR = paths["current_tov_data_dir"]
    CURRENT_SLM_PLOTS_DIR = paths["current_slm_plots_dir"]

    # Crucial paths to pass to functions that need them
    CURRENT_EOS_OUTPUT_DIR = paths[
        "current_eos_input_dir"
    ]  # This is the specific QEOS/MSEOS output folder

    # Create necessary directories
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    CURRENT_EOS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # EOS_FILES_DIR.mkdir(parents=True, exist_ok=True)
    TOV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    CURRENT_SLM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_TOV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_SLM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if parametric:
        tests_dir = paths["tests_dir"]
        tests_dir.mkdir(parents=True, exist_ok=True)

    # CURRENT_EOS_OUTPUT_DIR will be created within _generate_and_run_slm
    return paths, eos_name

def main(parametric, tidal, mseos, paths:dict, eos_name: str):
    """Main function to orchestrate the process."""
    # --- Path Assignment from the passed dictionary ---
    CURRENT_EOS_OUTPUT_DIR = paths["current_eos_input_dir"] # Specific EOS_files output folder
    CURRENT_TOV_DATA_DIR = paths["current_tov_data_dir"]   # Specific TOV_data output folder
    CURRENT_SLM_RESULTS_DIR = paths["current_slm_results_dir"] # Specific Results output folder
    CURRENT_SLM_PLOTS_DIR = paths["current_slm_plots_dir"]     # Specific Plots output folder
    # Assuming the non-parametric EOS input file is stored under EOS_DATA_DIR (project default)
    EOS_DATA_DIR = paths["package_eos_data_dir"]

    # Initialize list to store paths of generated TOV data files for ParametricSLM
    tov_data_files = []

    if parametric:
        print("Running Parametric DMD...")
        # Define the base command for EOS generation scripts
        if mseos:
            mseos_instance = MSEOS()
            for ls in Ls:
                for lv in Lv:
                    for zeta in zetaVal:
                        for xi in xiVal:
                            # Generate MSEOS with the current parameters
                            print(ls, lv, zeta, xi)
                            mseos_file_name = (
                                f"mseos_{ls:.2e}_{lv:.2e}_{zeta:.2e}_{xi:.2e}.txt"
                            )
                            mseos_output_path = CURRENT_EOS_OUTPUT_DIR / mseos_file_name
                            mseos_instance.main(ls, lv, zeta, xi, mseos_output_path)
                            tov = TOV(mseos_output_path, tidal=tidal)
                            results = tov.tov_routine()
                            tov_file_name = (
                                f"MR_{ls:.2e}_{lv:.2e}_{zeta:.2e}_{xi:.2e}.txt"
                            )
                            mr_results = np.asarray(
                                [
                                    results[0],  # Radius (km)
                                    results[1],  # Central Pressure (MeV/fm^3)
                                    results[2],  # Mass (M_sun)
                                    results[3],  # Tidal Deformability k2
                                ],
                                dtype=np.float64,
                            )
                            # Save results to specific directories
                            np.savetxt(
                                CURRENT_TOV_DATA_DIR / tov_file_name,
                                mr_results.T,
                                header="Radius (km) Central Pressure (MeV/fm^3) Mass (M_sun) k2",
                                delimiter="\t",
                            )
                            X = np.asarray(
                                [
                                    np.log(results[0]),
                                    np.log(results[1]),
                                    np.log(results[2]),
                                    np.log(results[3]),
                                ],
                                dtype=np.float64,
                            )
                            slm = SLM(X, dt=1, error_threshold=1e-6)
                            Xdmd = slm[4]
                            eigs = slm[2]
                            # Save results to specific directories
                            mseos_results_path = (
                                CURRENT_SLM_RESULTS_DIR / f"SLM_{mseos_file_name}"
                            )
                            mseos_results_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savetxt(
                                mseos_results_path,
                                np.exp(Xdmd.real).T,
                                header="Radius (km) Mass (M_sun) Central Pressure (MeV/fm^3) Tidal Deformability",
                                delimiter="\t",
                            )
                            mseos_plot_path = (
                                CURRENT_SLM_PLOTS_DIR / f"SLM_{mseos_file_name}.png"
                            )
                            mseos_plot_path.parent.mkdir(parents=True, exist_ok=True)
                            plot_eigs(eigs, output_path=mseos_plot_path)
                            ylabels = [
                                r"Mass $(M_sun)$",
                                r"Central Pressure $(MeV/fm^3)$",
                                r"Tidal Deformability $k_2$",
                            ]
                            fileNames = [
                                f"SLM_{mseos_file_name}".strip(".txt"),
                                f"Eigs_{mseos_file_name}".strip(".txt"),
                            ]
                            plot_slm_rad(
                                X,
                                Xdmd,
                                fileNames_for_labels=fileNames,
                                plots_output_base_path=CURRENT_SLM_PLOTS_DIR,
                                ylabels=ylabels,
                            )
        else:
            # Loop through parameters and generate/run
            for lam in lamVal:
                for kappa in kappaVal:
                    qeos_file_name = f"QEOS_{f'{lam:.2e}'.replace('.', '_')}_{f'{kappa:.2e}'.replace('.', '_')}.txt"
                    fileName = CURRENT_EOS_OUTPUT_DIR / qeos_file_name
                    generate_quarkyonia_eos(lam, kappa, fileName)
                    tov = TOV(fileName, tidal=tidal)
                    results = tov.tov_routine()
                    tov_file_name = f"MR_{lam:.2e}_{kappa:.2e}.txt"
                    mr_results = np.asarray(
                        [
                            results[0],  # Radius (km)
                            results[1],  # Central Pressure (MeV/fm^3)
                            results[2],  # Mass (M_sun)
                            results[3],  # Tidal Deformability k2
                        ],
                        dtype=np.float64,
                    )
                    # tov_data_files.append(tov_data_path)
                    # Save results to specific directories
                    np.savetxt(
                        CURRENT_TOV_DATA_DIR / tov_file_name,
                        mr_results.T,
                        header="Radius (km) Central Pressure (MeV/fm^3) Mass (M_sun) k2",
                        delimiter="\t",
                    )
                    X = np.asarray(
                        [
                            np.log(results[0]),
                            np.log(results[1]),
                            np.log(results[2]),
                            np.log(results[3]),
                        ],
                        dtype=np.float64,
                    )
                    slm = SLM(X, dt=1, error_threshold=1e-6)
                    Xdmd = slm[4]
                    # Save results to specific directories
                    qeos_results_path = (
                        CURRENT_SLM_RESULTS_DIR / f"SLM_{qeos_file_name}"
                    )
                    qeos_results_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(
                        qeos_results_path,
                        np.exp(Xdmd.real).T,
                        header="Radius (km) Mass (M_sun) Central Pressure (MeV/fm^3) Tidal Deformability",
                        delimiter="\t",
                    )
                    qeos_plot_path = (
                        CURRENT_SLM_PLOTS_DIR / f"SLM_{qeos_file_name}.png"
                    )
                    qeos_plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_eigs(slm[2], output_path=qeos_plot_path)
                    ylabels = [
                        r"Mass $(M_sun)$",
                        r"Central Pressure $(MeV/fm^3)$",
                        r"Tidal Deformability $k_2$",
                    ]
                    print(qeos_file_name)
                    fileNames = [
                        f"SLM_{qeos_file_name}".strip(".txt"),
                        f"Eigs_{qeos_file_name}".strip(".txt"),
                    ]
                    plot_slm_rad(
                        X,
                        Xdmd,
                        fileNames_for_labels=fileNames,
                        plots_output_base_path=CURRENT_SLM_PLOTS_DIR,
                        ylabels=ylabels,
                    )

    else:
        # fileName = input("Insert filename of EOS data: ") or "apr_eos.table"
        filePath = EOS_DATA_DIR / eos_name
        print("Running General SLM for file:", eos_name)
        tov = TOV(filePath, tidal=tidal)
        results = tov.tov_routine()
        tov_file_name = f"MR_{eos_name}.txt"
        mr_results = np.asarray(
            [
                results[0],  # Radius (km)
                results[1],  # Central Pressure (MeV/fm^3)
                results[2],  # Mass (M_sun)
                results[3],  # Tidal Deformability k2
            ],
            dtype=np.float64,
        )
        # tov_data_files.append(tov_data_path)
        # Save results to specific directories
        np.savetxt(
            CURRENT_TOV_DATA_DIR / tov_file_name,
            mr_results.T,
            header="Radius (km) Central Pressure (MeV/fm^3) Mass (M_sun) k2",
            delimiter="\t",
        )
        X = np.asarray(
            [
                np.log(results[0]),
                np.log(results[1]),
                np.log(results[2]),
                np.log(results[3]),
            ],
            dtype=np.float64,
        )
        slm = SLM(X, dt=1, error_threshold=1e-6)
        Xdmd = slm[4]
        # Save results to specific directories
        results_path = (
            CURRENT_SLM_RESULTS_DIR / f"SLM_{eos_name}.txt"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            results_path,
            np.exp(Xdmd.real).T,
            header="Radius (km) Mass (M_sun) Central Pressure (MeV/fm^3) Tidal Deformability",
            delimiter="\t",
        )
        plot_path = (
            CURRENT_SLM_PLOTS_DIR / f"SLM_{eos_name}.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_eigs(slm[2], output_path=plot_path)
        ylabels = [
            r"Central Pressure $({\rm{MeV/fm}}^3)$",
            r"Mass $({\rm{M}}_{\odot})$",
            r"Tidal Deformability $k_2$",
        ]
        print(f"File Name: {eos_name}")
        fileNames = [
            f"SLM_{eos_name}".strip(".txt"),
            f"Eigs_{eos_name}".strip(".txt"),
        ]
        plot_slm_rad(
            X,
            Xdmd,
            fileNames_for_labels=fileNames,
            plots_output_base_path=CURRENT_SLM_PLOTS_DIR,
            ylabels=ylabels,
        )


def run_pSLM():
    """Function to run parametric SLM."""
    pslm = ParametricSLM()
    pslm.fit()
    pslm.predict()


if __name__ == "__main__":
    cleanup = _get_boolean_input("Clean up the directories?", default_value=False)
    if cleanup is True:
        clean_directory()

    parametric = _get_boolean_input("Parametric or General DMD?", default_value=True)
    tidal = _get_boolean_input("Tidal or Non-Tidal?", default_value=True)

    mseos = False  # Default value
    if parametric is True:
        mseos = _get_boolean_input("MSEOS or Quarkyonia?", default_value=True)

    all_paths, eos_name  = make_folders(parametric, mseos)

    main(parametric, tidal, mseos, all_paths, eos_name)
