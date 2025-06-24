#!/usr/bin/env python
# coding: utf-8
# Python TOV solver
# Sudhanva Lalit: Date 06/14/2024

"""Information about the code:
This code solves TOV equations for mass radius relations. This can also plot the mass-radius curve.

The code solves dr/dp and dm/dp instead of the regular way.

USE: To use the code, here are the steps:
1) Include the file in your main code e.g. import tov_class as tc
2) Load the EoS using the ToV loader, tc.ToV(filename, arraysize)
3) call the solver as tc.ToV.mass_radius(min_pressure, max_pressure)
4) To plot, follow the code in main() on creating the dictionary of inputs

Updates: Version 0.0.1-1
Solves ToV, can only take inputs of pressure (MeV/fm^3), energy density in MeV, baryon density in fm^-3
in ascending order.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import pylab

# constants
r0 = 8.378  # km
m0 = 2.837  # solar mass
alphaG = 5.922e-39  # constant
lambda_n = 2.10e-19  # km
rSun = 2.953  # Schwarzschild Radius of sun in km
eps0 = 1.285e3  # MeV/fm^3
pres0 = 1.285e3  # MeV/fm^3


class TOV:
    """Solves TOV equations and gives data-table, mass-radius plot and max. mass, central pressure
    and central density by loading an EoS datafile."""

    def __init__(self, filename, imax, tidal=False):
        self.file = np.loadtxt(filename)
        self.tidal = tidal
        self.e_in = self.file[:, 1] / eps0  # Scaled Energy density
        self.p_in = self.file[:, 2] / pres0  # Scaled pressure
        self.nb_in = self.file[:, 0]  # Scaled baryon density
        if self.tidal:
            self.cs2_in = self.file[:, 3]  # sound speed
        self.imax = imax
        self.radius = np.empty(self.imax)
        self.mass = np.empty(self.imax)

    def pressure_from_nb(self, nb):
        """Evaluate pressure from number density using interpolation"""
        p1 = interp1d(
            self.nb_in, self.p_in, axis=0, kind="linear", fill_value="extrapolate"
        )
        return p1(nb)

    def energy_from_pressure(self, pressure):
        """Evaluate energy density from pressure using interpolation"""
        plow = 1e-10 / pres0
        if pressure < plow:
            return 2.6e-310
        else:
            e1 = interp1d(
                self.p_in, self.e_in, axis=0, kind="linear", fill_value="extrapolate"
            )
            return e1(pressure)

    def pressure_from_energy(self, energy):
        """Evaluate pressure from energy density using interpolation"""
        p1 = interp1d(
            self.e_in, self.p_in, axis=0, kind="linear", fill_value="extrapolate"
        )
        return p1(energy)

    def baryon_from_energy(self, energy):
        """Evaluate number density from energy using interpolation"""
        n1 = interp1d(
            self.e_in, self.nb_in, axis=0, kind="linear", fill_value="extrapolate"
        )
        return n1(energy)

    def f_x(self, x, mass, pres, eps):
        one = 1.0 - (3.0 / 2.0) * x**2 * (eps - pres)
        two = 1.0 - mass / x
        return one / two

    def q_x(self, x, mass, pres, eps, cs2):
        pre = (3.0 / 2.0) * x**2 / (1.0 - mass / x)
        one = 5.0 * eps + 9 * pres + ((eps + pres) / cs2) - 4.0 / x**2
        two = (mass / x) + 3.0 * x**2 * pres
        three = 1 - mass / x
        return pre * one - (two / three) ** 2

    def tov_rhs(self, initial, pres):
        if self.tidal:
            x, mass, y = initial
        else:
            x, mass = initial

        edn = self.energy_from_pressure(pres)
        # Equations one: dr/dp, 2: dm/dp and 3: dy/dps
        one = x * (mass - x) / (0.5 * (edn + pres) * (mass + 3.0 * x**3 * pres))
        two = 3.0 * x**2 * edn * one
        if self.tidal:
            f = self.f_x(x, mass, pres, edn)
            q = self.q_x(x, mass, pres, edn, y)
            three = -(1.0 / x) * (y**2 + f * y + q) * one
            return np.asarray([one, two, three], dtype=np.float64)
        return np.asarray([one, two], dtype=np.float64)

    def tovsolve(self, pcent):
        dx = 1e-3
        # x = np.arange(pcent, dx, -dx)
        x = np.geomspace(pcent, 1e-12, num=100, endpoint=False)
        initial = 1e-3, 1e-12
        if self.tidal:
            initial = 1e-3, 1e-12, 2.0
        psol = odeint(self.tov_rhs, initial, x, rtol=1e-8, atol=1e-8)
        rstar = 0.0
        mstar = 0.0
        count = 0
        rData = []

        count = np.where(x > 1e-10)[0][-1]
        # print(count)
        # Calculate tidal deformability
        # if self.tidal:
        #     yR = psol[: count - 1, 2]
        #     mass = psol[: count - 1, 1] * m0
        #     radius = psol[: count - 1, 0] * r0
        #     tidal_deform, k2 = self.tidal_def(yR, mass, radius)
        #     # print(f"Tidal Deformability: {tidal_deform}, Love number: {k2}")
        return (
            # np.array(rData[: count - 1]) * r0,
            psol[: count - 1, 0] * r0,
            np.array(x[: count - 1]),
            psol[: count - 1, 1] * m0,
            psol[: count - 1, 2] if self.tidal else None,
        )
        # return rstar * r0, mstar * m0

    def tidal_def(self, yR, mass, radius):
        # love number calculation
        beta = mass / radius
        k2 = (
            (8.0 / 5.0)
            * beta**5.0
            * (1.0 - 2.0 * beta) ** 2.0
            * (2.0 - yR + 2.0 * beta * (yR - 1.0))
            * (
                2.0 * beta * (6.0 - 3.0 * yR + 3.0 * beta * (5.0 * yR - 8.0))
                + 4.0
                * beta**3.0
                * (
                    13.0
                    - 11.0 * yR
                    + beta * (3.0 * yR - 2.0)
                    + 2.0 * beta**2.0 * (1.0 + yR)
                )
                + 3.0
                * (1.0 - 2.0 * beta) ** 2.0
                * (2.0 - yR + 2.0 * beta * (yR - 1.0))
                * np.log(1.0 - 2.0 * beta)
            )
            ** (-1.0)
        )

        # tidal deformability calculation
        tidal_deform = (2.0 / 3.0) * k2 / (beta**5.0)

        return tidal_deform, k2

    def mass_radius(self, pmin, pmax):
        pc = np.zeros(self.imax)
        mass = np.zeros(self.imax)
        radius = np.zeros(self.imax)
        pc = np.geomspace(pmax, pmin, num=50)
        for i, pcent in enumerate(pc):
            # pc[i] = pmin + (pmax - pmin) * i / self.imax
            radius[i], mass[i] = self.tovsolve(pcent)
        # print(radius, mass)
        self.radius = np.ma.masked_equal(radius, 0.0).compressed()
        self.mass = np.ma.masked_equal(mass, 0.0).compressed()
        return self.radius, self.mass, pc

    def rad14(self):
        n1 = interp1d(
            self.mass, self.radius, axis=0, kind="linear", fill_value="extrapolate"
        )
        r14 = n1(1.4)
        Max_mass = np.max(self.mass)
        Max_radius = n1(Max_mass)
        return print(
            "Radius of 1.4 M_sun star : {} \n Max_mass : {}, Max_radius: {}".format(
                r14, Max_mass, Max_radius
            )
        )

    def plot(self, data, **kwargs):
        xl = kwargs["xlabel"]
        yl = kwargs["ylabel"]
        fl = kwargs["filename"]
        ttl = kwargs["title"]
        fig = pylab.figure(figsize=(11, 11), dpi=600)
        ax1 = fig.add_subplot(111)
        [ax1.plot(data[0], data[i + 1], label=ttl) for i in range(len(data) - 1)]
        # ax1.plot(data[0], data[1], '-b', label=ttl)

        pylab.xlabel(xl, fontsize=24)
        pylab.ylabel(yl, fontsize=24)
        ax1.tick_params(
            direction="inout",
            length=10,
            width=2,
            colors="k",
            grid_color="k",
            labelsize=24,
        )
        pylab.legend(loc="upper right", fontsize=24)
        pylab.ylim(auto=True)
        pylab.xlim(auto=True)
        pylab.savefig(fl)


def main():
    imax = 100
    # Replace the filename and run the code
    fileName = "EOS_files/NJL_EOS.dat"
    file = TOV(fileName, imax)
    print(file.pressure_from_energy(1.0), file.energy_from_pressure(1.0))
    radius = np.empty(imax)
    mass = np.empty(imax)
    radius, mass, pcentral = file.mass_radius(1e-3, 2.5)
    print(f"Radius: {radius}, Mass: {mass}")
    radius = np.ma.masked_equal(radius, 0.0).compressed()
    mass = np.ma.masked_equal(mass, 0.0).compressed()
    data1 = np.array([radius, mass])
    file.rad14()
    data = np.array(data1)
    labels = {
        "xlabel": "radius (km)",
        "ylabel": r"Mass (M$_{\odot}$)",
        "filename": "Mass-Rad.pdf",
        "title": "Mass-Radius",
    }
    file.plot(data, **labels)
    labels = {
        "xlabel": "radius (km)",
        "ylabel": r"Pressure ",
        "filename": "Pres-Rad.pdf",
        "title": "Pressure-Radius",
    }
    data2 = np.array([radius, pcentral])
    file.plot(data2, **labels)


if __name__ == "__main__":
    main()
