import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv  # Importing analytical solution

# Function to compute numerical solution
def compute_numerical_solution(Phi, N, boundary_order):
    delta_eta = 1.0 / N  # Step size
    eta = np.linspace(0, 1, N + 1)  # Grid points

    # Initialize coefficient matrix A and RHS vector b
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Apply boundary condition at eta = 0 (Symmetry)
    if boundary_order == 1:
        A[0, 0] = 1
        A[0, 1] = -1  # First-order boundary condition
        b[0] = 0
    elif boundary_order == 2:
        A[0, 0] = -3
        A[0, 1] = 4
        A[0, 2] = -1  # Second-order boundary condition
        b[0] = 0
    elif boundary_order == 3:
        A[0, 0] = 11
        A[0, 1] = -18
        A[0, 2] = 9
        A[0, 3] = -2  # Third-order boundary condition
        b[0] = 0

    # Apply boundary condition at eta = 1 (Surface concentration)
    A[-1, -1] = 1
    b[-1] = 1

    # Fill matrix A for interior points
    for i in range(1, N):
        eta_p = eta[i] + 0.5 * delta_eta  # η_(i+1/2)
        eta_m = eta[i] - 0.5 * delta_eta  # η_(i-1/2)

        # Coefficients for finite difference method
        A[i, i - 1] = eta_m / (eta[i] * delta_eta**2)
        A[i, i] = - (eta_p + eta_m) / (eta[i] * delta_eta**2) - Phi**2
        A[i, i + 1] = eta_p / (eta[i] * delta_eta**2)

    # Solve linear system
    C_A_star_numerical = np.linalg.solve(A, b)
    return eta, C_A_star_numerical

# Plot Error vs N
def plot_error_vs_N(Phi, N_values):
    errors_L1_1st, errors_L2_1st, errors_Lmax_1st = [], [], []
    errors_L1_2nd, errors_L2_2nd, errors_Lmax_2nd = [], [], []
    errors_L1_3rd, errors_L2_3rd, errors_Lmax_3rd = [], [], []

    for N in N_values:
        eta_1, C_A_star_1st = compute_numerical_solution(Phi, N, boundary_order=1)
        eta_2, C_A_star_2nd = compute_numerical_solution(Phi, N, boundary_order=2)
        eta_3, C_A_star_3rd = compute_numerical_solution(Phi, N, boundary_order=3)
        C_A_star_analytical = iv(0, Phi * eta_1) / iv(0, Phi)

        # Compute errors for 1st order
        L1_error_1st = np.mean(np.abs(C_A_star_1st - C_A_star_analytical))
        L2_error_1st = np.sqrt(np.mean((C_A_star_1st - C_A_star_analytical)**2))
        Lmax_error_1st = np.max(np.abs(C_A_star_1st - C_A_star_analytical))

        # Compute errors for 2nd order
        L1_error_2nd = np.mean(np.abs(C_A_star_2nd - C_A_star_analytical))
        L2_error_2nd = np.sqrt(np.mean((C_A_star_2nd - C_A_star_analytical)**2))
        Lmax_error_2nd = np.max(np.abs(C_A_star_2nd - C_A_star_analytical))

        # Compute errors for 3rd order
        L1_error_3rd = np.mean(np.abs(C_A_star_3rd - C_A_star_analytical))
        L2_error_3rd = np.sqrt(np.mean((C_A_star_3rd - C_A_star_analytical)**2))
        Lmax_error_3rd = np.max(np.abs(C_A_star_3rd - C_A_star_analytical))

        errors_L1_1st.append(L1_error_1st)
        errors_L2_1st.append(L2_error_1st)
        errors_Lmax_1st.append(Lmax_error_1st)

        errors_L1_2nd.append(L1_error_2nd)
        errors_L2_2nd.append(L2_error_2nd)
        errors_Lmax_2nd.append(Lmax_error_2nd)

        errors_L1_3rd.append(L1_error_3rd)
        errors_L2_3rd.append(L2_error_3rd)
        errors_Lmax_3rd.append(Lmax_error_3rd)

    # Plot errors
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, errors_L1_1st, label='$L_1$ Error (1st Order)', marker='o')
    plt.plot(N_values, errors_L2_1st, label='$L_2$ Error (1st Order)', marker='s')
    plt.plot(N_values, errors_Lmax_1st, label='$L_{max}$ Error (1st Order)', marker='^')
    plt.plot(N_values, errors_L1_2nd, label='$L_1$ Error (2nd Order)', linestyle='--', marker='o')
    plt.plot(N_values, errors_L2_2nd, label='$L_2$ Error (2nd Order)', linestyle='--', marker='s')
    plt.plot(N_values, errors_Lmax_2nd, label='$L_{max}$ Error (2nd Order)', linestyle='--', marker='^')
    plt.plot(N_values, errors_L1_3rd, label='$L_1$ Error (3rd Order)', linestyle='-.', marker='o')
    plt.plot(N_values, errors_L2_3rd, label='$L_2$ Error (3rd Order)', linestyle='-.', marker='s')
    plt.plot(N_values, errors_Lmax_3rd, label='$L_{max}$ Error (3rd Order)', linestyle='-.', marker='^')
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
    errors_L1_1st, errors_L2_1st, errors_Lmax_1st = [], [], []
    errors_L1_2nd, errors_L2_2nd, errors_Lmax_2nd = [], [], []
    errors_L1_3rd, errors_L2_3rd, errors_Lmax_3rd = [], [], []

    for Phi in Phi_values:
        eta_1, C_A_star_1st = compute_numerical_solution(Phi, N, boundary_order=1)
        eta_2, C_A_star_2nd = compute_numerical_solution(Phi, N, boundary_order=2)
        eta_3, C_A_star_3rd = compute_numerical_solution(Phi, N, boundary_order=3)
        C_A_star_analytical = iv(0, Phi * eta_1) / iv(0, Phi)

        # Compute errors for 1st order
        L1_error_1st = np.mean(np.abs(C_A_star_1st - C_A_star_analytical))
        L2_error_1st = np.sqrt(np.mean((C_A_star_1st - C_A_star_analytical)**2))
        Lmax_error_1st = np.max(np.abs(C_A_star_1st - C_A_star_analytical))

        # Compute errors for 2nd order
        L1_error_2nd = np.mean(np.abs(C_A_star_2nd - C_A_star_analytical))
        L2_error_2nd = np.sqrt(np.mean((C_A_star_2nd - C_A_star_analytical)**2))
        Lmax_error_2nd = np.max(np.abs(C_A_star_2nd - C_A_star_analytical))

        # Compute errors for 3rd order
        L1_error_3rd = np.mean(np.abs(C_A_star_3rd - C_A_star_analytical))
        L2_error_3rd = np.sqrt(np.mean((C_A_star_3rd - C_A_star_analytical)**2))
        Lmax_error_3rd = np.max(np.abs(C_A_star_3rd - C_A_star_analytical))

        errors_L1_1st.append(L1_error_1st)
        errors_L2_1st.append(L2_error_1st)
        errors_Lmax_1st.append(Lmax_error_1st)

        errors_L1_2nd.append(L1_error_2nd)
        errors_L2_2nd.append(L2_error_2nd)
        errors_Lmax_2nd.append(Lmax_error_2nd)

        errors_L1_3rd.append(L1_error_3rd)
        errors_L2_3rd.append(L2_error_3rd)
        errors_Lmax_3rd.append(Lmax_error_3rd)

    # Plot errors
    plt.figure(figsize=(10, 6))
    plt.plot(Phi_values, errors_L1_1st, label='$L_1$ Error (1st Order)', marker='o')
    plt.plot(Phi_values, errors_L2_1st, label='$L_2$ Error (1st Order)', marker='s')
    plt.plot(Phi_values, errors_Lmax_1st, label='$L_{max}$ Error (1st Order)', marker='^')
    plt.plot(Phi_values, errors_L1_2nd, label='$L_1$ Error (2nd Order)', linestyle='--', marker='o')
    plt.plot(Phi_values, errors_L2_2nd, label='$L_2$ Error (2nd Order)', linestyle='--', marker='s')
    plt.plot(Phi_values, errors_Lmax_2nd, label='$L_{max}$ Error (2nd Order)', linestyle='--', marker='^')
    plt.plot(Phi_values, errors_L1_3rd, label='$L_1$ Error (3rd Order)', linestyle='-.', marker='o')
    plt.plot(Phi_values, errors_L2_3rd, label='$L_2$ Error (3rd Order)', linestyle='-.', marker='s')
    plt.plot(Phi_values, errors_Lmax_3rd, label='$L_{max}$ Error (3rd Order)', linestyle='-.', marker='^')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Thiele Modulus ($\Phi$)')
    plt.ylabel('Error')
    plt.title('Error Metrics for Different Thiele Modulus Values')
    plt.legend()
    plt.grid()
    plt.show()

# Parameters
Phi_values = [10**-1, 10**0, 10**1, 10**2]
N_values = [10, 20, 50, 100, 200]

# Run all plots
plot_error_vs_N(Phi=1.0, N_values=N_values)
plot_error_vs_Phi(N=100, Phi_values=Phi_values)
