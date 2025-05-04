import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Load the data file
data = pd.read_csv('Group A1.txt', sep='\t', skiprows=1)
data.columns = ['T', 'P_CO2', 'P_H2', 'R']

# Convert columns to the appropriate data types
data['T'] = data['T'].astype(float)  # Temperature in Kelvin
data['P_CO2'] = data['P_CO2'].astype(float)  # Partial pressure of CO2 in bar
data['P_H2'] = data['P_H2'].astype(float)  # Partial pressure of H2 in bar
data['R'] = data['R'].astype(float)  # Reaction rate in mol/s

# Define constants
R_gas = 8.314  # Universal gas constant, J/(mol·K)

# Define the modified rate law function
def modified_rate_law(T, P_CO2, P_H2, A, b, E_a, m, n):
    k0 = A * (T ** b)
    return k0 * np.exp(-E_a / (R_gas * T)) * (P_CO2 ** m) * (P_H2 ** n)

# Extract data from the DataFrame
T_data = data['T'].values
P_CO2_data = data['P_CO2'].values
P_H2_data = data['P_H2'].values
R_exp_data = data['R'].values

# Define the updated cost function for Differential Evolution
def log_cost_function(params):
    A, b, E_a, m, n = params
    R_fit = modified_rate_law(T_data, P_CO2_data, P_H2_data, A, b, E_a, m, n)
    log_residuals = np.log(R_exp_data) - np.log(R_fit)
    return np.sum(log_residuals**2)  # RSS on logarithmic scale

# Define bounds for parameters: [A, b, E_a (J/mol), m, n]
param_bounds = [
    (1e-5, 1e6),   # A: Prefactor scaling constant
    (-3, 3),       # b: Temperature exponent
    (1e3, 1e5),    # E_a: Activation Energy
    (0, 3),        # m: Reaction order for P_CO2
    (0, 3)         # n: Reaction order for P_H2
]

# Perform Differential Evolution optimization with the new cost function
result = differential_evolution(log_cost_function, bounds=param_bounds, strategy='best1bin', maxiter=1000, tol=1e-6)

# Extract optimized parameters
A_opt, b_opt, E_a_opt, m_opt, n_opt = result.x

# Print results
print("Optimized Parameters with Temperature-Dependent Prefactor:")
print(f"A (Prefactor Scaling Constant): {A_opt}")
print(f"b (Temperature Exponent): {b_opt}")
print(f"Activation Energy (E_a, J/mol): {E_a_opt}")
print(f"Reaction Order with respect to P_CO2 (m): {m_opt}")
print(f"Reaction Order with respect to P_H2 (n): {n_opt}")

# Calculate the fitted reaction rates using optimized parameters
R_fit = modified_rate_law(T_data, P_CO2_data, P_H2_data, A_opt, b_opt, E_a_opt, m_opt, n_opt)

# Calculate residuals
log_residuals = np.log(R_exp_data) - np.log(R_fit)

# Calculate error metrics
log_mae = np.mean(np.abs(log_residuals))  # Mean Absolute Error on log scale
log_rmse = np.sqrt(np.mean(log_residuals**2))  # Root Mean Squared Error on log scale

print("\nError Metrics on Logarithmic Scale:")
print(f"Mean Absolute Error (MAE): {log_mae}")
print(f"Root Mean Squared Error (RMSE): {log_rmse}")

# Plot residuals vs Temperature
plt.figure()
plt.scatter(T_data, log_residuals, c='blue', label='Logarithmic Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('Logarithmic Residuals vs Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals (ln(R_experimental) - ln(R_fit))')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals vs P_CO2
plt.figure()
plt.scatter(P_CO2_data, log_residuals, c='green', label='Logarithmic Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('Logarithmic Residuals vs P_CO2')
plt.xlabel('P_CO2 (bar)')
plt.ylabel('Residuals (ln(R_experimental) - ln(R_fit))')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals vs P_H2
plt.figure()
plt.scatter(P_H2_data, log_residuals, c='purple', label='Logarithmic Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('Logarithmic Residuals vs P_H2')
plt.xlabel('P_H2 (bar)')
plt.ylabel('Residuals (ln(R_experimental) - ln(R_fit))')
plt.legend()
plt.grid(True)
plt.show()
