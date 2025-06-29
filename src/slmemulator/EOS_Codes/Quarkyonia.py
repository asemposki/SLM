###########################################
# Quarkyonia Equation of State (PNM)
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################

import sys
import os
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from ..config import get_paths


# Constants
pi2 = np.pi**2
Nc = 3.0
hc = 197.33
hc2 = hc**2
mnmev = 939
n0 = 0.16
a = -28.8
b = 10.0
mn = mnmev / hc  # Nucleon mass in MeV


# Function to calculate eps(k, m)
# def eps(k, m):
#     pre = 1.0 / (2.0 * pi2)
#     t1 = np.sqrt(k**2 + m**2)
#     t = 0.25 * k * t1**3 - (m**2 / 8.0) * k * t1 - (m**4 / 8.0) * np.log(t1 + k)
#     return pre * t


def eps(k, m):
    pre = 1.0 / (2.0 * pi2)
    t1 = np.sqrt(k**2 + m**2)
    t = (
        0.25 * k * t1**3
        - (m**2 / 8.0) * k * t1
        + (m**4 / 16.0) * np.log((t1 - k) / (t1 + k))
    )
    return pre * t


def generate_quarkyonia_eos(lamInput, kappa, output_filepath):
    """
    Generates Quarkyonia EOS data and writes it to a specified file.
    """
    lam = lamInput / hc
    kap = kappa
    kfmin = 0.25
    kfmax = 5.0
    kArray = np.linspace(kfmin, kfmax, 400)
    # set paths
    paths = get_paths()
    eos_data_dir = paths["eos_data_dir"]
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    # Initialize arrays
    epst = np.zeros(len(kArray))
    nbq = np.zeros(len(kArray))
    nQuarks = []

    for i in range(len(kArray)):
        kfb = kArray[i]
        delta = lam**3 / kfb**2 + kap * lam / Nc**2

        # Neutrons
        if kfb <= delta:
            theta = 0
        else:
            theta = 1
            nQuarks.append(i)
        kfq = abs(kfb - delta) / Nc * theta
        ll = Nc * kfq
        ul = kfb
        nn = (ul**3 - ll**3) / (3 * pi2)
        u = nn / n0
        vn = (a * u + b * u**2) * nn
        Eul = eps(ul, mn)
        Ell = eps(ll, mn)
        epskb = 2 * (Eul - Ell) * hc
        epstb = epskb + vn

        # Add quarks
        kfd = abs(kfb - delta) / 3.0 * theta
        kfu = 2 ** (-1 / 3.0) * kfd
        nq = (kfd**3 + kfu**3) / (3 * pi2)
        nu = kfu**3 / pi2
        nd = kfd**3 / pi2

        # Total baryon density
        nb = nn + nq
        ebk = epskb / nb - mnmev
        ebi = vn / nb
        eb = ebk + ebi

        # Quarks only
        mq = mn / 3.0
        eulu = eps(kfu, mq)
        ellu = eps(0.0, mq)
        epsku = 2 * Nc * (eulu - ellu) * hc
        euld = eps(kfd, mq)
        elld = eps(0.0, mq)
        epskd = 2 * Nc * (euld - elld) * hc
        epskq = epsku + epskd

        # Total including quarks and nucleons
        epstot = epskb + vn + epskq  # (optional if quark energy is included)
        epst[i] = epstot  # * 8.956e-7 (conversion factor if needed)
        nbq[i] = nb

        # Output results for each step (replacing write statement)
        # print(f"nb: {nb}, nn/nb: {nn/nb}, nq/nb: {nq/nb}, nu: {nu}, nd: {nd}")

    interpErho = CubicSpline(nbq, epst)
    indices = np.where(nbq >= 0.08)
    nbq = nbq[indices]
    epst = epst[indices]

    press = -epst + interpErho(nbq, 1) * nbq
    interpPrho = CubicSpline(nbq, press)
    cs2 = interpPrho(nbq, 1) / interpErho(nbq, 1)

    # Write where quarks begin to appear
    filename = "quarkyonia_params.txt"
    if os.path.exists(filename):
        with open(filename, "a") as f:
            f.write(f"{kap} {lamInput} {nbq[nQuarks[0]]} \n")
    else:
        with open(filename, "w") as f:
            f.write(f"{kap} {lamInput} {nbq[nQuarks[0]]} \n")

    # Add lowden part
    # Read the lowdensity file
    mft_path = eos_data_dir / "MFT_ns6p.dat"
    lowden = np.loadtxt(mft_path)  # Skip header if present
    eLow, pLow, nbLow = lowden.T
    pLow = pLow[::-1]
    eLow = eLow[::-1]
    nbLow = nbLow[::-1]
    eLowDer = CubicSpline(nbLow, eLow).derivative()
    pLowDer = CubicSpline(nbLow, pLow).derivative()
    cs2Low = pLowDer(nbLow) / eLowDer(nbLow)

    # Combine the two parts
    nbq = np.concatenate((nbLow, nbq))
    epst = np.concatenate((eLow, epst))
    press = np.concatenate((pLow, press))
    cs2 = np.concatenate((cs2Low, cs2))

    nameList = "nb (fm^-1)   E (MeV)     P (MeV/fm^3)    cs2"

    # Ensure the parent directory for the output file exists
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving Quarkyonia EOS to {output_filepath}...")
    np.savetxt(
        output_filepath,
        np.array([nbq, epst, press, cs2]).T,  # Transpose to save columns
        header=nameList,
        delimiter="   ",
        fmt="%.6e",
    )
    # shutil.move(fileName, QEOS_PATH)
    print(f"Quarkonia EOS saved to {output_filepath}")


if __name__ == "__main__":
    # Expects 3 command-line arguments: lambda_val, kappa_val, output_directory
    if len(sys.argv) < 4:
        print("Usage: python Quarkyonia.py <lambda_val> <kappa_val> <output_directory>")
        sys.exit(1)

    try:
        lamVal = float(sys.argv[1])
        kappaVal = float(sys.argv[2])
        # The output_directory argument is passed as a string from testSLM.py
        output_dir = Path(sys.argv[3])
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        print("Usage: python Quarkyonia.py <lambda_val> <kappa_val> <output_directory>")
        sys.exit(1)

    # Construct the full output file path within the specified output directory
    output_file_name = f"MR_{lamVal:.2f}_{kappaVal:.2f}.txt"  # Consistent naming
    full_output_filepath = output_dir / output_file_name

    generate_quarkyonia_eos(lamVal, kappaVal, full_output_filepath)
