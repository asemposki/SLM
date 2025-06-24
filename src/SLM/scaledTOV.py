#!/usr/bin/env python
# coding: utf-8
# Python TOV solver
# Sudhanva Lalit: Date 02/14/2023

""" Information about the code:
This code solves TOV equations for mass radius relations. This can also plot the mass-radius curve.

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
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import pylab
import os

# constants
r0 = 8.378  # km
m0 = 2.837  # solar mass
alphaG = 5.922e-39  # constant
lambda_n = 2.10e-19  # km
rSun = 2.953  # Schwarzschild Radius of sun in km
eps0 = 1.285e3  # MeV/fm^3
pres0 = 1.285e3  # MeV/fm^3
msol = 1.116e60  # Mass of sun in MeV
Ggrav = 1.324e-42  # Mev^-1*fm
rsol = 2.954e18
# Schwarrzschild radius of the sun in fm
rhosol = msol * 3 / (4.0 * np.pi * rsol**3)


class TOV:
    """Solves TOV equations and gives data-table, mass-radius plot and max. mass, central pressure
    and central density by loading an EoS datafile."""

    def __init__(self, filename, imax):
        self.file = np.loadtxt(filename, skiprows=3)
        self.e_in = self.file[:, 1] / eps0  # Scaled Energy density
        self.p_in = self.file[:, 2] / pres0  # Scaled pressure
        self.nb_in = self.file[:, 0]
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

    # include the RK4 handwritten solver
    def RK4(self, f, x0, t0, te, N):
        r"""
        A simple RK4 solver to avoid overhead of
        calculating with solve_ivp or any other
        adaptive step-size function.

        Example:
            tov.RK4(f=func, x0=1., t0=1., te=10., N=100)

        Parameters:
            f (func): A Python function for the ODE(s) to be solved.
                Able to solve N coupled ODEs.

            x0 (float): Guess for the function(s) to be solved.

            t0 (float): Initial point of the grid.

            te (float): End point of the grid.

            N (int): The number of steps to take in the range (te-t0).

        Returns:
            times (array): The grid of solution steps.

            solution (array): The solutions of each function
                at each point in the grid.
        """

        h = (te - t0) / N
        times = np.arange(t0, te + h, h)
        solution = []
        x = x0

        for t in times:
            solution.append(np.array(x).T)
            k1 = h * f(t, x)
            k2 = h * f(t + 0.5 * h, x + 0.5 * k1)
            k3 = h * f(t + 0.5 * h, x + 0.5 * k2)
            k4 = h * f(t + h, x + k3)
            x += (k1 + 2 * (k2 + k3) + k4) / 6

        solution = np.asarray(solution, dtype=np.float64).T

        return times, solution

    def tov_rhs(self, x, initial):
        pres, mass = initial
        # mass = initial[1]
        edn = self.energy_from_pressure(pres)
        # Equations one: pressure, 2: mass
        if pres > 0.0:
            one = 0.5 * (edn + pres) * (mass + 3.0 * x**3 * pres) / (x * mass - x**2)
            two = 3.0 * x**2 * edn
        else:
            one = 0.0
            two = 0.0
        return np.asarray([one, two], dtype=np.float64)

    def tovsolve(self, pcent, xfinal):
        dx = 1e-3
        # x = np.arange(dx, xfinal, dx / 2)
        # x = np.geomspace(1e-3, xfinal, num=1000, endpoint=False)
        # x = np.geomspace(dx, xfinal, num=100, endpoint=False)
        initial = pcent, 0.0
        # initial = pcent, 4 * np.pi * dx**3 / (3.0)
        # psol = odeint(self.tov_rhs, initial, x, rtol=1e-12, atol=1e-12)
        psol = solve_ivp(
            self.tov_rhs,
            [1e-3, xfinal],
            initial,
            max_step=1,
            dense_output=True,
            rtol=1e-10,
            atol=1e-10,
        )
        # xval, psol = self.RK4(self.tov_rhs, initial, 1e-3, xfinal, 2500)
        # print(psol)
        rstar = 0.0
        mstar = 0.0
        count = 0
        rData = []

        index_mass = np.where([psol.y[0] > 1e-11])[1][-1]
        # print(index_mass)
        rstar = psol.t[index_mass]  # x[index_mass]
        mstar = psol.y[1][index_mass]

        # mstar = psol[count - 1, 1]
        # pcentral = psol[:, 0][0]
        # print(
        #     psol[: count - 1, 1].shape,
        #     np.array(rData[: count - 1]).shape,
        #     psol[: count - 1, 1].shape,
        # )
        eps = []
        for i in range(index_mass):
            eps.append(self.energy_from_pressure(psol.y[0][i]))
        return (
            np.array(psol.t[:index_mass]) * r0,
            psol.y[0, :index_mass],
            psol.y[1, :index_mass] * m0,
            np.array(eps) * eps0,
        )
        # return rstar * r0, pcentral, mstar * m0

    def mass_radius(self, pmin, pmax):
        pc = np.zeros(self.imax)
        mass = np.zeros(self.imax)
        radius = np.zeros(self.imax)
        pcentral = np.zeros(self.imax)
        maxp = max(self.p_in)
        pmax = min(pmax, maxp)
        pmin = max(pmin, 1e-3)
        pc = np.geomspace(pmin, pmax)
        for i, pcent in enumerate(pc):
            # pc[i] = pmin + (pmax - pmin) * i / self.imax
            radius[i], pcentral[i], mass[i] = self.tovsolve(pcent, 10)
        self.radius = np.ma.masked_equal(radius, 0.0).compressed()
        self.mass = np.ma.masked_equal(mass, 0.0).compressed()
        self.pcentral = np.ma.masked_equal(pcentral, 0.0).compressed()
        return self.radius, self.pcentral, self.mass

    # @staticmethod
    def rad14(self):
        n1 = interp1d(
            self.mass, self.radius, axis=0, kind="linear", fill_value="extrapolate"
        )
        r14 = n1(1.4)
        # f = interp1d(self.radius, self.mass, axis=0, kind='cubic', fill_value='extrapolate')
        Max_mass = np.max(self.mass)
        Max_radius = n1(Max_mass)
        # nidx = np.where(self.mass == Max_mass)
        # Max_radius = self.radius[nidx][0]
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
    os.getcwd()
    fileName = "neos.dat"
    file = TOV(fileName, imax)
    print(file.pressure_from_energy(1.0), file.energy_from_pressure(1.0))
    # print(type(file))
    radius = np.empty(imax)
    mass = np.empty(imax)
    radius, pcentral, mass = file.mass_radius(1e-3, 1.5)
    radius = np.ma.masked_equal(radius, 0.0).compressed()
    mass = np.ma.masked_equal(mass, 0.0).compressed()
    print(f"Radius: {radius} \n Mass: {mass}")
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
