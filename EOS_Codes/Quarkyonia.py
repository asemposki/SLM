""" 
Quarkonia EOS using Mclaren-Reddy Paper
Author: Sudhanva Lalit
Started: 09/14/2024
Latest Update: 09/14/2024
"""

import numpy as np
import sys
import os
from scipy.interpolate import CubicSpline

# Constants
pi2 = np.pi**2
Nc = 3.0
hc = 197.33
hc2 = hc**2
mnmev = 939
# lam = 380.0 / hc
# kap = 0.3
n0 = 0.16
a = -28.8
b = 10.0
mn = mnmev / hc  # Nucleon mass in MeV


# Function to calculate eps(k, m)
def eps(k, m):
    pre = 1.0 / (2.0 * pi2)
    t1 = np.sqrt(k**2 + m**2)
    t = 0.25 * k * t1**3 - (m**2 / 8.0) * k * t1 - (m**4 / 8.0) * np.log(t1 + k)
    return pre * t


def main(kap, lamInput):
    lam = lamInput / hc
    kfmin = 0.025
    kfmax = 6.0
    kArray = np.linspace(kfmin, kfmax, 100)
    # Initialize arrays
    epst = np.zeros(len(kArray))
    nbq = np.zeros(len(kArray))

    for i in range(len(kArray)):
        kfb = kArray[i]
        delta = lam**3 / kfb**2 + kap * lam / Nc**2

        # Neutrons
        if kfb < delta:
            theta = 0
        else:
            theta = 1
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

    press = -epst + interpErho(nbq, 1) * nbq
    interpPrho = CubicSpline(nbq, press)
    cs2 = interpPrho(nbq, 1) / interpErho(nbq, 1)

    # Add lowden part
    # Read the lowdensity file
    lowden = np.loadtxt("../EOS_Data/MFT_ns6p.dat")
    eLow, pLow, nbLow = lowden.T
    pLow = pLow[::-1]
    eLow = eLow[::-1]
    nbLow = nbLow[::-1]
    eLowDer = CubicSpline(nbLow, eLow).derivative()
    pLowDer = CubicSpline(nbLow, pLow).derivative()
    cs2Low = pLowDer(nbLow) / eLowDer(nbLow)

    # Combine the two parts
    # nbq = np.concatenate((nbLow, nbq))
    # epst = np.concatenate((eLow, epst))
    # press = np.concatenate((pLow, press))
    # cs2 = np.concatenate((cs2Low, cs2))

    nameList = "nb (fm^-1)   E (MeV)     P (MeV/fm^3)    cs2"
    fileName = f"EOS_Quarkyonia_{lamInput:.2f}_{kap:.2f}.dat"
    np.savetxt(
        fileName,
        np.array([nbq, epst, press, cs2], dtype=np.float64).T,
        header=nameList,
        fmt="%.6e",
    )
    os.system(f"mv {fileName} ../EOS_files/Quarkies/")


if __name__ == "__main__":
    argv = sys.argv
    kap, lam = argv[1:]
    main(float(kap), float(lam))
