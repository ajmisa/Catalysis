# -*- coding: utf-8 -*-

# Reaction-diffusion in a spherical particle with arbitrary order kinetics
#
# * Reaction: A + B -> C
# * A and B are explicitly being modeled
# * Steady-state solution is found via time-integration
# * Reaction order set to a = 1 and b = 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.integrate import solve_ivp

def main():
    N = 40          # number of control volumes
    Da = 1e-5       # Effective diffusion coefficient A [m^2/s]
    Db = 1e-5       # Effective diffusion coefficient B [m^2/s]
    k = 5           # Reaction rate constant [(mol/m^3)/s]
    R = 1e-2        # Radius of the particle [m]
    ca0 = 1         # Concentration of A at the surface [mol/m^3]
    cb0 = 2         # Concentration of B at the surface [mol/m^3]
    dr = R / N      # radial interval

    # Geometry information
    r = np.linspace(dr / 2, R - dr / 2, N)
    rw = r - dr / 2
    re = r + dr / 2
    V = (re**3 - rw**3) / 3

    # Construct a dictionary that contains all system variables
    var = {
        'N': N,
        'Da': Da,
        'Db': Db,
        'k': k,
        'R': R,
        'ca0': ca0,
        'cb0': cb0,
        'dr': dr,
        'r': r,
        'rw': rw,
        're': re,
        'V': V
    }

    # Construct matrix for diffusion
    A = build_matrix(var)

    # Initial conditions
    c0 = np.ones(2 * N)  # Initial guess for concentrations

    # Time-integrate the ODE system
    res = solve_ivp(ode, t_span=[0, 1000], y0=c0, args=(A, var), method='LSODA', rtol=1e-4, atol=1e-4)

    # Extract steady-state concentrations
    ca = res.y[0:N, -1]
    cb = res.y[N:2 * N, -1]

    # Plot numerical results
    plt.figure(dpi=144, figsize=(6, 4))
    plt.plot(r, ca, 'o-', label='A numerical', zorder=5)
    plt.plot(r, cb, '^-', label='B numerical', zorder=5)
    plt.ylim(0, max(ca0, cb0))
    plt.grid(linestyle='--')
    plt.xlabel('$r$ [m]')
    plt.ylabel('$c$ [mol/m$^{3}$]')
    plt.title('Concentration Profiles for A + B -> C')
    plt.legend()
    plt.show()

def ode(t, c, A, var):
    """
    dc/dt part of the ordinary differential equation
    """
    V = var['V']

    # Ensure concentrations are non-negative
    c = np.maximum(c, 1e-12)

    # Reaction term and boundary conditions
    b = build_vector(c, var)
    return (A.dot(c) - b) / np.hstack([V, V])

def build_matrix(var):
    """
    Matrix that describes the diffusion part of the system
    """
    N = var['N']
    Da = var['Da']
    Db = var['Db']
    re = var['re']
    rw = var['rw']
    dr = var['dr']

    A = np.zeros((2 * N, 2 * N))

    # Central elements
    for i in range(1, N - 1):
        # For A
        aw = Da * rw[i]**2 / dr
        ae = Da * re[i]**2 / dr
        ap = -aw - ae

        A[i, i - 1] = aw
        A[i, i] = ap
        A[i, i + 1] = ae

        # For B
        aw = Db * rw[i]**2 / dr
        ae = Db * re[i]**2 / dr
        ap = -aw - ae

        A[N + i, N + i - 1] = aw
        A[N + i, N + i] = ap
        A[N + i, N + i + 1] = ae

    # Left boundary (center of particle)
    A[0, 1] = Da * re[0]**2 / dr
    A[0, 0] = -A[0, 1]
    A[N, N + 1] = Db * re[0]**2 / dr
    A[N, N] = -A[N, N + 1]

    # Right boundary (edge of particle)
    A[N - 1, N - 2] = Da * rw[-1]**2 / dr
    A[N - 1, N - 1] = -A[N - 1, N - 2] - 2 * Da * re[-1]**2 / dr
    A[-1, -2] = Db * rw[-1]**2 / dr
    A[-1, -1] = -A[-1, -2] - 2 * Db * re[-1]**2 / dr

    return A

def build_vector(c, var):
    """
    All non-linear terms of the reaction-diffusion system
    """
    N = var['N']
    k = var['k']
    V = var['V']
    Da = var['Da']
    Db = var['Db']
    R = var['R']
    dr = var['dr']
    ca0 = var['ca0']
    cb0 = var['cb0']

    # Calculate reaction term
    ca = c[0:N]
    cb = c[N:2 * N]
    reac = k * ca * cb * V

    # Construct vector
    b = np.zeros(2 * N)

    # Set reaction term
    b[0:N] = reac
    b[N:2 * N] = reac

    # Set boundary conditions
    b[N - 1] -= 2 * Da * ca0 * R**2 / dr
    b[2 * N - 1] -= 2 * Db * cb0 * R**2 / dr

    return b

if __name__ == '__main__':
    main()
