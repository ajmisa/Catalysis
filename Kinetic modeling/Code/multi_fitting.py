import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the data file
data = pd.read_csv('Group A1.txt', sep='\t', skiprows=1)
data.columns = ['T', 'P_CO2', 'P_H2', 'R']

# Convert columns to the appropriate data types
data['T'] = data['T'].astype(float)  # Temperature in Kelvin
data['P_CO2'] = data['P_CO2'].astype(float)  # Partial pressure of CO2 in bar
data['P_H2'] = data['P_H2'].astype(float)  # Partial pressure of H2 in bar
data['R'] = data['R'].astype(float)  # Reaction rate in mol/s

# Filter data for temperatures below 500 K
data = data[data['T'] < 500]

# Define constants
R_gas = 8.314  # Universal gas constant, J/(mol·K)

# Define the multi-parameter model
def multi_parameter_model(X, A, B, C, D, E, F, G):
    T, P_CO2, P_H2 = X
    k_T = A * np.exp(-B / T)  # Temperature dependency (Arrhenius-like)
    f_P = C * P_CO2**D + E * P_H2**F + G * P_CO2 * P_H2  # Pressure dependency
    return k_T * f_P

# Extract data from the DataFrame
T_data = data['T'].values
P_CO2_data = data['P_CO2'].values
P_H2_data = data['P_H2'].values
R_exp_data = data['R'].values

# Prepare input arrays for curve_fit
X_data = (T_data, P_CO2_data, P_H2_data)

# Initial guesses for parameters: A, B, C, D, E, F, G
initial_guesses = [1e3, 5000, 1.0, 1.0, 1.0, 1.0, 1.0]

# Use curve_fit to optimize parameters
optimal_params, covariance = curve_fit(
    multi_parameter_model,
    X_data,
    R_exp_data,
    p0=initial_guesses,
    bounds=(0, np.inf)
)

# Extract optimized parameters
A_opt, B_opt, C_opt, D_opt, E_opt, F_opt, G_opt = optimal_params

# Display results
print("Optimized Parameters:")
print(f"A (prefactor): {A_opt}")
print(f"B (activation energy factor): {B_opt}")
print(f"C (P_CO2 coefficient): {C_opt}")
print(f"D (P_CO2 order): {D_opt}")
print(f"E (P_H2 coefficient): {E_opt}")
print(f"F (P_H2 order): {F_opt}")
print(f"G (interaction coefficient): {G_opt}")

# Calculate the fitted reaction rates using the optimized parameters
R_fit = multi_parameter_model((T_data, P_CO2_data, P_H2_data), A_opt, B_opt, C_opt, D_opt, E_opt, F_opt, G_opt)

# Calculate residuals
residuals = R_exp_data - R_fit

# Calculate R-squared
ss_res = np.sum(residuals**2)
ss_tot = np.sum((R_exp_data - np.mean(R_exp_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared}")

# Calculate the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
mae = np.mean(np.abs(residuals))  # Mean Absolute Error
rmse = np.sqrt(np.mean(residuals**2))  # Root Mean Squared Error
print("Fit Errors Between Experimental and Fitted Reaction Rates:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot experimental vs fitted rates
plt.figure()
plt.scatter(R_exp_data, R_fit, color='blue', label='Data')
plt.plot([R_exp_data.min(), R_exp_data.max()], [R_exp_data.min(), R_exp_data.max()], color='red', linestyle='--', label='Ideal Fit')
plt.title('Experimental vs Fitted Reaction Rates')
plt.xlabel('Experimental Reaction Rates (R)')
plt.ylabel('Fitted Reaction Rates (R_fit)')
plt.legend()
plt.grid(True)
plt.show()

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
