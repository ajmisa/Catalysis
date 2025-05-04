import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, curve_fit

# Load the data file
data = pd.read_csv('Group A1.txt', sep='\t', skiprows=1)
data.columns = ['T', 'P_CO2', 'P_H2', 'R']

# Convert columns to appropriate data types
data['T'] = data['T'].astype(float)  # Temperature in Kelvin
data['P_CO2'] = data['P_CO2'].astype(float)  # Partial pressure of CO2 in bar
data['P_H2'] = data['P_H2'].astype(float)  # Partial pressure of H2 in bar
data['R'] = data['R'].astype(float)  # Reaction rate in mol/s

# Define constants
R_gas = 8.314  # Universal gas constant, J/(mol·K)

# Modified rate law with second-order temperature dependency
def modified_rate_law(T, P_CO2, P_H2, k0, E_a, beta, m, n):
    k_T = k0 * np.exp(-E_a / (R_gas * T) + beta / T**2)
    return k_T * (P_CO2 ** m) * (P_H2 ** n)

# Extract data from the DataFrame
T_data = data['T'].values
P_CO2_data = data['P_CO2'].values
P_H2_data = data['P_H2'].values
R_exp_data = data['R'].values

# Define the cost function for fitting
def cost_function(params):
    k0, E_a, beta, m, n = params
    R_fit = modified_rate_law(T_data, P_CO2_data, P_H2_data, k0, E_a, beta, m, n)
    residuals = R_exp_data - R_fit
    return np.sum(residuals**2)  # RSS: Residual Sum of Squares

# Define bounds for parameters: [k0, E_a (J/mol), beta, m, n]
param_bounds = [
    (1e-5, 1e6),   # k0: Prefactor
    (1e3, 1e5),    # E_a: Activation Energy
    (-1e5, 1e5),   # beta: Second-order temperature dependency
    (0, 3),        # m: Reaction order for P_CO2
    (0, 3)         # n: Reaction order for P_H2
]

# Perform Differential Evolution optimization
result = differential_evolution(cost_function, bounds=param_bounds, strategy='best1bin', maxiter=2000, tol=1e-6)

# Extract optimized parameters
k0_opt, E_a_opt, beta_opt, m_opt, n_opt = result.x

# Print results
print("Optimized Parameters with Second-Order Temperature Dependency:")
print(f"Rate Prefactor (k0): {k0_opt}")
print(f"Activation Energy (E_a, J/mol): {E_a_opt}")
print(f"Second-Order Temperature Coefficient (beta): {beta_opt}")
print(f"Reaction Order with respect to P_CO2 (m): {m_opt}")
print(f"Reaction Order with respect to P_H2 (n): {n_opt}")

# Calculate the fitted reaction rates using optimized parameters
R_fit = modified_rate_law(T_data, P_CO2_data, P_H2_data, k0_opt, E_a_opt, beta_opt, m_opt, n_opt)

# Calculate residuals
residuals = R_exp_data - R_fit

# Calculate error metrics
mae = np.mean(np.abs(residuals))  # Mean Absolute Error
rmse = np.sqrt(np.mean(residuals**2))  # Root Mean Squared Error

print("\nError Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot residuals vs Temperature
plt.figure()
plt.scatter(T_data, residuals, c='blue', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('Residuals vs Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals (R_experimental - R_fit)')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals vs P_CO2
plt.figure()
plt.scatter(P_CO2_data, residuals, c='green', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('Residuals vs P_CO2')
plt.xlabel('P_CO2 (bar)')
plt.ylabel('Residuals (R_experimental - R_fit)')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals vs P_H2
plt.figure()
plt.scatter(P_H2_data, residuals, c='purple', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('Residuals vs P_H2')
plt.xlabel('P_H2 (bar)')
plt.ylabel('Residuals (R_experimental - R_fit)')
plt.legend()
plt.grid(True)
plt.show()
