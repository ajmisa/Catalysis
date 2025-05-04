import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv  # Importing analytical solution

# Function to compute numerical solution
def compute_numerical_solution(Phi, N):
    delta_eta = 1.0 / N  # Step size
    eta = np.linspace(0, 1, N + 1)  # Grid points

    # Initialize coefficient matrix A and RHS vector b
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Apply boundary condition at eta = 0 (Symmetry)
    A[0, 0] = 1
    A[0, 1] = -1  # Symmetry condition dC/dη = 0
    b[0] = 0

    # Apply boundary condition at eta = 1 (Surface concentration)
    A[-1, -1] = 1
    b[-1] = 1

    # Fill matrix A for interior points
    for i in range(1, N):
        eta_p = eta[i] + 0.5 * delta_eta
        eta_m = eta[i] - 0.5 * delta_eta
        A[i, i - 1] = eta_m / (eta[i] * delta_eta**2)
        A[i, i] = - (eta_p + eta_m) / (eta[i] * delta_eta**2) - Phi**2
        A[i, i + 1] = eta_p / (eta[i] * delta_eta**2)

    # Solve linear system
    C_A_star_numerical = np.linalg.solve(A, b)
    return eta, C_A_star_numerical

# Plot Error vs N
def plot_error_vs_N(Phi, N_values):
    errors_L1, errors_L2, errors_Lmax = [], [], []

    for N in N_values:
        eta, C_A_star_numerical = compute_numerical_solution(Phi, N)
        C_A_star_analytical = iv(0, Phi * eta) / iv(0, Phi)

        # Compute errors
        L1_error = np.mean(np.abs(C_A_star_numerical - C_A_star_analytical))
        L2_error = np.sqrt(np.mean((C_A_star_numerical - C_A_star_analytical)**2))
        Lmax_error = np.max(np.abs(C_A_star_numerical - C_A_star_analytical))

        errors_L1.append(L1_error)
        errors_L2.append(L2_error)
        errors_Lmax.append(Lmax_error)

    # Plot errors
    plt.figure()
    plt.plot(N_values, errors_L1, label='$L_1$ Error', marker='o')
    plt.plot(N_values, errors_L2, label='$L_2$ Error', marker='s')
    plt.plot(N_values, errors_Lmax, label='$L_{max}$ Error', marker='^')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Grid Points (N)')
    plt.ylabel('Error')
    plt.title(f'Error Metrics for Different Grid Sizes (Phi={Phi})')
    plt.legend()
    plt.grid()
    plt.show()

# Plot Error vs Thiele Modulus
def plot_error_vs_Phi(N, Phi_values):
    errors_L1, errors_L2, errors_Lmax = [], [], []

    for Phi in Phi_values:
        eta, C_A_star_numerical = compute_numerical_solution(Phi, N)
        C_A_star_analytical = iv(0, Phi * eta) / iv(0, Phi)

        # Compute errors
        L1_error = np.mean(np.abs(C_A_star_numerical - C_A_star_analytical))
        L2_error = np.sqrt(np.mean((C_A_star_numerical - C_A_star_analytical)**2))
        Lmax_error = np.max(np.abs(C_A_star_numerical - C_A_star_analytical))

        errors_L1.append(L1_error)
        errors_L2.append(L2_error)
        errors_Lmax.append(Lmax_error)

    # Plot errors
    plt.figure()
    plt.plot(Phi_values, errors_L1, label='$L_1$ Error', marker='o')
    plt.plot(Phi_values, errors_L2, label='$L_2$ Error', marker='s')
    plt.plot(Phi_values, errors_Lmax, label='$L_{max}$ Error', marker='^')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Thiele Modulus ($\\Phi$)')
    plt.ylabel('Error')
    plt.title('Error Metrics for Different Thiele Modulus Values')
    plt.legend()
    plt.grid()
    plt.show()

# Plot Numerical vs Analytical for Different N
def plot_numerical_vs_analytical_N(Phi, N_values):
    eta_fine = np.linspace(0, 1, 1000)
    C_A_star_analytical = iv(0, Phi * eta_fine) / iv(0, Phi)

    plt.figure()
    for N in N_values:
        eta, C_A_star_numerical = compute_numerical_solution(Phi, N)
        plt.plot(eta, C_A_star_numerical, label=f'Numerical (N = {N})')

    plt.plot(eta_fine, C_A_star_analytical, linestyle='--', label='Analytical Solution', color='black')
    plt.xlabel('Dimensionless Radius ($\\eta$)')
    plt.ylabel('Dimensionless Concentration ($C_A^*$)')
    plt.title(f'Numerical vs Analytical Solution (Phi={Phi})')
    plt.legend()
    plt.grid()
    plt.show()

# Plot Numerical vs Analytical for Different Phi
def plot_numerical_vs_analytical_Phi(N, Phi_values):
    eta_fine = np.linspace(0, 1, 1000)

    plt.figure()
    for Phi in Phi_values:
        eta, C_A_star_numerical = compute_numerical_solution(Phi, N)
        C_A_star_analytical = iv(0, Phi * eta_fine) / iv(0, Phi)
        plt.plot(eta, C_A_star_numerical, label=f'Numerical (Phi = {Phi:.0e})')
        plt.plot(eta_fine, C_A_star_analytical, linestyle='--', label=f'Analytical (Phi = {Phi:.0e})')

    plt.xlabel('Dimensionless Radius ($\\eta$)')
    plt.ylabel('Dimensionless Concentration ($C_A^*$)')
    plt.title(f'Numerical vs Analytical Solution for Different Phi (N={N})')
    plt.legend()
    plt.grid()
    plt.show()

# Parameters
Phi_values = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
N_values = [10, 20, 50, 100, 200]

# Run all plots
plot_error_vs_N(Phi=1.0, N_values=N_values)
plot_error_vs_Phi(N=100, Phi_values=Phi_values)
plot_numerical_vs_analytical_N(Phi=1.0, N_values=N_values)
plot_numerical_vs_analytical_Phi(N=100, Phi_values=Phi_values)
