####################################################
# Muller Serot Equation of State (Beta Equilibrium)
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
####################################################

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import sys
from ..config import get_paths


class MSEOS:
    def __init__(self):
        # Constants
        self.hc = 197.33
        self.pi = np.pi
        self.pi2 = self.pi * self.pi
        self.n0 = 0.16
        self.msom = 0.6
        self.knm = 230 / self.hc
        self.eoa = -16.0 / self.hc
        self.esym = 30.0 / self.hc
        # Masses
        self.mst, self.mn, self.ms, self.mr, self.mw, self.mmu = (
            0.4814,
            4.7585,
            3.3446,
            3.9021,
            3.9679,
            0.53720,
        )
        self.me = 2.5897e-3
        self.ms2 = self.ms * self.ms
        self.mr2 = self.mr * self.mr
        self.mw2 = self.mw * self.mw
        self.mmu2 = self.mmu * self.mmu

        # Get the paths using the get_paths function from config.py
        # By default, get_paths returns paths relevant to MSEOS if use_mseos is True, which is its default.
        self.paths = get_paths()
        self.mseos_output_dir = self.paths["mseos_path_specific"]

    # Define functions for solving integrals
    def _fg(self, data):
        gam, g, ms, kf, efs = data
        r = (kf + efs) / ms
        return g * gam / (2.0 * self.pi2) * 0.5 * ms * (kf * efs - ms**2 * np.log(r))

    def _feden(self, gam, kf, ms):
        if kf > 1e-12:
            if ms == 0.0:
                return gam / (2.0 * self.pi) * kf**4 / 4.0
            else:
                ef = np.sqrt(kf * kf + ms * ms)
                r = (kf + ef) / ms
                term = 2.0 * kf * ef**3 - kf * ef * ms**2 - ms**4 * np.log(r)
                return gam / (2.0 * self.pi * self.pi * 8.0) * term
        return 0.0

    def _fpres(self, gam, kf, ms):
        if kf > 1e-12:
            if ms == 0.0:
                return gam / (2.0 * self.pi) * kf**4 / 12.0
            else:
                ef = np.sqrt(kf * kf + ms * ms)
                r = (kf + ef) / ms
                term = (
                    2.0 * kf**3 * ef - 3.0 * kf * ef * ms**2 + 3.0 * ms**4 * np.log(r)
                )
                return gam / (2.0 * self.pi * self.pi * 24.0) * term
        else:
            return 0.0

    def _fcn(self, xf, *args):
        zeta, xi = args
        (cs2, cw2, b, c) = xf
        obt = 1.0 / 3.0
        ome = cw2 * self.n0
        dome = cw2
        phi = self.mn * (1.0 - self.msom)
        mns = self.mn * self.msom

        gam = 4.0
        kf = (1.5 * self.pi * self.pi * self.n0) ** (1 / 3)
        efs = np.sqrt(kf**2 + mns**2)
        rhos = (
            gam
            / (4.0 * self.pi * self.pi)
            * mns
            * (kf * efs - mns**2 * np.log((kf + efs) / mns))
        )

        if zeta != 0.0:
            cq = 8.0 / (9.0 * cw2**3 * self.n0**2 * zeta)
            lr = 3 * self.n0 / zeta
            lr13 = lr ** (obt)
            sqt = np.sqrt(1 + cq)
            fir = (1 + sqt) ** obt
            sec = abs(1 - sqt) ** obt
            ome = lr13 * (fir - sec)
            dome = 1 / (1 / cw2 + zeta / 2 * ome**2)

        kbar = 2.0 * b * self.mn
        lbar = 6.0 * c
        us = kbar / 6.0 * phi**3 + lbar / 24.0 * phi**4
        duds = kbar / 2.0 * phi**2 + lbar / 6.0 * phi**3
        uv = -zeta / 24.0 * ome**4

        eq1 = phi / cs2 + duds - rhos
        eps = (
            phi**2 / (2.0 * cs2)
            + ome * self.n0
            - ome**2 / (2.0 * cw2)
            + us
            + uv
            + self._feden(gam, kf, mns)
        )
        eq2 = eps / self.n0 - self.mn - self.eoa
        eq3 = (
            -(phi**2) / (2.0 * cs2)
            + ome**2 / (2.0 * cw2)
            - us
            - uv
            + self._fpres(gam, kf, mns)
        )

        one = 1.0 / cs2 + kbar * phi + lbar / 2.0 * phi**2
        two = 3.0 / mns * rhos
        tri = 3.0 * self.n0 / efs
        denr = one + two - tri
        aknm = (
            9.0
            * self.n0
            * (dome + kf**2 / (3.0 * efs * self.n0) - (mns / efs) ** 2 / denr)
        )
        eq4 = aknm - self.knm
        return np.array([eq1, eq2, eq3, eq4], dtype=np.float64)

    def _fcns(self, x, *args):
        (sig, ome) = x
        barn, gs, gw, b, c, gam1, zeta, xi = args
        gw2 = gw * gw
        kf = (1.5 * self.pi2 * barn) ** (1 / 3)
        mns = self.mn - gs * sig
        efs = np.sqrt(kf**2 + mns**2)
        fgar = [gam1, gs, mns, kf, efs]
        fgs = self._fg(fgar)

        duds = b * self.mn * gs**3 * sig**2 + c * gs**4 * sig**3
        eq1 = self.ms2 * sig + duds - fgs
        eq2 = self.mw2 * ome - gw * barn + zeta * gw2**2 * ome**3 / 6.0
        return np.array([eq1, eq2], dtype=np.float64)

    def _grcalc(self, data, fields):
        gs, gw, Ls, Lv = data
        sign0, omen0, mnsn0 = fields
        kf = (1.5 * self.pi2 * self.n0) ** (1 / 3)
        kf2 = kf * kf
        efs = np.sqrt(kf2 + mnsn0**2)
        t1 = Ls * (gs * sign0) ** 2
        t2 = Lv * (gw * omen0) ** 2
        fsfw = t1 + t2
        gomr2 = 1.0 / (self.n0 / 8.0 / (self.esym - kf2 / (6.0 * efs)) - 2.0 * fsfw)
        gr2 = gomr2 * self.mr2
        gr = np.sqrt(gr2)
        return gr

    def _SNM_EOS(self, data):
        rhomin, rhomax, gs, gw, b, c, gam, zeta, xi = data
        nb = np.linspace(rhomin, rhomax, 150)
        eps = np.zeros_like(nb)
        pres = np.zeros_like(nb)
        kf = (1.5 * self.pi2 * self.n0) ** (1 / 3)
        sig = 0
        ome = 0
        sign0 = 0
        omen0 = 0
        for i in range(len(nb)):
            ls = (nb[i], gs, gw, b, c, gam, zeta, xi)
            if i == 0:
                xs = (0.01 * self.mn / gs, gw * nb[i] / self.mw2)
            xs = fsolve(self._fcns, xs, args=ls)
            sig = xs[0]
            ome = xs[1]
            mns = self.mn - gs * sig
            if nb[i] == 0.16:
                sign0 = sig
                omen0 = ome
                mnsn0 = mns
            us = b / 3.0 * self.mn * (gs * sig) ** 3 + c / 4.0 * (gs * sig) ** 4
            uv = zeta / 8.0 * (gw * ome) ** 4
            edenb = (
                0.5 * (self.ms * sig) ** 2
                + 0.5 * (self.mw * ome) ** 2
                + us
                + uv
                + self._feden(gam, kf, mns)
            )
            eps[i] = edenb * self.hc
            presb = (
                -0.5 * (self.ms * sig) ** 2
                + 0.5 * (self.mw * ome) ** 2
                - us
                + uv / 3.0
                + self._fpres(gam, kf, mns)
            )
            pres[i] = presb * self.hc
        presInterp = CubicSpline(nb, pres)
        epsInterp = CubicSpline(nb, eps)
        cs2 = presInterp(nb, 1) / epsInterp(nb, 1)
        return np.array([nb, eps, pres, cs2]), np.array([sign0, omen0, mnsn0])

    def _MS_EOS(self, data, fields):
        rhomin, rhomax, gs, gw, b, c, gam1, Ls, Lv, zeta, xi = data
        nb = np.linspace(rhomin, rhomax, 150)
        eps = np.zeros_like(nb)
        pres = np.zeros_like(nb)
        coups = (gs, gw, Ls, Lv)
        gr = self._grcalc(coups, fields)
        for i in range(len(nb)):
            ls = (nb[i], gs, gw, gr, b, c, gam1, Ls, Lv, zeta, xi)
            if i == 0:
                xs = (
                    0.01 * self.mn / gs,
                    gw * nb[i] / self.mw2,
                    -gr * nb[i] / (2.0 * self.mr2),
                    (3.0 * self.pi2 * nb[i]) ** (1 / 2),
                    1e-5,
                    1e-5,
                    1e-5,
                )
            xs = fsolve(self._EoS_func, xs, args=ls)
            sig = xs[0]
            ome = xs[1]
            rho = xs[2]
            kfn = xs[3]
            kfp = xs[4]
            kfe = xs[5]
            kfmu = xs[6]

            mns = self.mn - gs * sig
            efns = np.sqrt(kfn * kfn + mns * mns)
            efps = np.sqrt(kfp * kfp + mns * mns)
            mun = gw * ome - 0.5 * gr * rho + efns
            mup = gw * ome - 0.5 * gr * rho + efps
            mue = mun - mup

            nmu = kfmu**3 / (3.0 * self.pi2)
            ne = kfe**3 / (3.0 * self.pi2)
            nn = kfn**3 / (3.0 * self.pi2)
            npr = kfp**3 / (3.0 * self.pi2)

            us = b / 3.0 * self.mn * (gs * sig) ** 3 + c / 4.0 * (gs * sig) ** 4
            uv = zeta / 8.0 * (gw * ome) ** 4
            ur = xi / 8.0 * (gr * rho) ** 4
            epsmix1 = gr**2 * rho**2 * gw**2 * ome**2 * Lv
            epsmix2 = gr**2 * rho**2 * (Ls * gs**2 * sig**2 + Lv * gw**2 * ome**2)
            epsmix = epsmix1 + 2.0 * epsmix2
            edenb = (
                0.5 * (self.ms * sig) ** 2
                + 0.5 * (self.mw * ome) ** 2
                + 0.5 * (self.mr * rho) ** 2
                + us
                + uv
                + ur
                + epsmix
                + self._feden(gam1, kfn, mns)
                + self._feden(gam1, kfp, mns)
            )
            edenl = self._feden(gam1, kfe, 0.0) + self._feden(gam1, kfmu, self.mmu)
            eps[i] = (edenb + edenl) * self.hc
            presb = (
                -0.5 * (self.ms * sig) ** 2
                + 0.5 * (self.mw * ome) ** 2
                + 0.5 * (self.mr * rho) ** 2
                - us
                + uv / 3.0
                + ur / 3.0
                + self._fpres(gam1, kfn, mns)
                + self._fpres(gam1, kfp, mns)
                + self._fpres(gam1, kfe, 0.0)
                + self._fpres(gam1, kfmu, self.mmu)
                + epsmix2
            )
            pres[i] = presb * self.hc
        presDeriv = np.gradient(pres, nb)
        epsDeriv = np.gradient(eps, nb)
        cs2 = presDeriv / epsDeriv
        print("Success, MS EOS file created")
        return np.array([nb, eps, pres, cs2])

    def _EoS_func(self, x, *data):
        for i in range(3, 7):
            if x[i] < 0:
                x[i] = 0.0
        (sig, ome, rho, kfn, kfp, kfe, kfmu) = x
        barn, gs, gw, gr, b, c, gam1, Ls, Lv, zeta, xi = data
        gw2 = gw * gw
        tpi2 = 3 * self.pi**2
        nn = kfn**3 / tpi2
        npr = kfp**3 / tpi2
        ne = kfe**3 / tpi2
        nmu = kfmu**3 / tpi2

        mns = self.mn - gs * sig
        efns = np.sqrt(kfn * kfn + mns * mns)
        efps = np.sqrt(kfp * kfp + mns * mns)
        mun = gw * ome - 0.5 * gr * rho + efns
        mup = gw * ome + 0.5 * gr * rho + efps
        mue = kfe
        mumu = np.sqrt(kfmu * kfmu + self.mmu**2)

        fgn = self._fg((gam1, gs, mns, kfn, efns))
        fgp = self._fg((gam1, gs, mns, kfp, efps))

        duds = b * self.mn * gs**3 * sig**2 + c * gs**4 * sig**3
        dvds = zeta / 2 * gw2**2 * ome**3

        mixs = 2.0 * gr**2 * rho**2 * Ls * gs**2 * sig
        mixw = 2.0 * gr**2 * rho**2 * Lv * gw**2 * ome
        mixr = 2.0 * gr**2 * rho * (Ls * gs**2 * sig**2 + Lv * gw**2 * ome**2)

        eq1 = self.ms2 * sig + duds - (fgn + fgp) - mixs
        eq2 = self.mw2 * ome - gw * (nn + npr) + zeta * gw**4 * ome**3 / 6.0 + mixw
        eq3 = self.mr2 * rho - 0.5 * gr * (npr - nn) + xi * gr**4 * rho**3 / 6 + mixr

        eq4 = mun - mup - mue  # charge neutrality
        eq5 = barn - (nn + npr)
        eq6 = npr - (ne + nmu)
        eq7 = kfmu
        if mue > self.mmu:
            eq7 = mue - mumu
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7], dtype=np.float64)

    def main(self, Ls, Lv, zeta, xi, out_file=None):
        gam = 4.0
        gam1 = 2.0
        xf = (10.0, 10.0, 2e-3, 2e-3)
        ls = (zeta, xi)
        paths = get_paths()
        eos_data_dir = paths["eos_data_dir"]

        mft_path = eos_data_dir / "MFT_ns6p.dat"
        lowden = np.loadtxt(mft_path)

        [cs2, cw2, b, c] = fsolve(self._fcn, xf, args=ls)
        gs2 = cs2 * self.ms2
        gw2 = cw2 * self.mw2
        gs = np.sqrt(gs2)
        gw = np.sqrt(gw2)
        rhomin = np.float64(0.02)
        rhomax = np.float64(3.0)
        scvein = (rhomin, rhomax, gs, gw, b, c, gam, zeta, xi)
        snmEOS, fields = self._SNM_EOS(scvein)
        nameList = "nb (fm^-1)   E (MeV)     P (MeV/fm^3)    cs2"

        nb, eps, pres, speedOfSound = snmEOS
        scvein = (rhomin, rhomax, gs, gw, b, c, gam1, Ls, Lv, zeta, xi)
        fullEOS = self._MS_EOS(scvein, fields)
        fullEOS = fullEOS[:, 4:]

        lowden = np.loadtxt(mft_path)
        eLow, pLow, nbLow = lowden.T
        pLow = pLow[::-1]
        eLow = eLow[::-1]
        nbLow = nbLow[::-1]
        eLowDer = CubicSpline(nbLow, eLow).derivative()
        pLowDer = CubicSpline(nbLow, pLow).derivative()
        cs2Low = pLowDer(nbLow) / eLowDer(nbLow)
        nb2, eps2, pres2, speedOfSound2 = fullEOS
        nb2 = np.hstack([nbLow, nb2])
        eps2 = np.hstack([eLow, eps2])
        pres2 = np.hstack([pLow, pres2])
        speedOfSound2 = np.hstack([cs2Low, speedOfSound2])
        fullEOS = np.array([nb2, eps2, pres2, speedOfSound2])

        # Use the class attribute for the MSEOS output directory and Path object concatenation
        fixed_path = (
            self.mseos_output_dir / f"EOS_MS_{Ls:.4f}_{Lv:.3f}_{zeta:.4f}_{xi:.1f}.txt"
        )
        fileName = out_file if out_file else fixed_path

        np.savetxt(fileName, fullEOS.T, fmt="%.6e", header=nameList)
        print(f"MS EOS saved to {fileName}")


if __name__ == "__main__":
    mseos_calculator = MSEOS()
    argv = sys.argv
    if len(argv) == 6:
        Ls, Lv, zeta, xi, out_file = argv[1:]
        print("Running MSEOS code now with parameters:")
        print(f"Ls: {Ls}, Lv: {Lv}, zeta: {zeta}, xi: {xi}, fileName: {out_file}")
        mseos_calculator.main(
            float(Ls), float(Lv), float(zeta), float(xi), out_file=out_file
        )
    elif len(argv) == 5:
        Ls, Lv, zeta, xi = argv[1:]
        print("Running MSEOS code now with parameters:")
        print(f"Ls: {Ls}, Lv: {Lv}, zeta: {zeta}, xi: {xi}")
        mseos_calculator.main(float(Ls), float(Lv), float(zeta), float(xi))
    else:
        print("Usage: python MSEOS.py <Ls> <Lv> <zeta> <xi> [<output_file>]")
        sys.exit(1)
