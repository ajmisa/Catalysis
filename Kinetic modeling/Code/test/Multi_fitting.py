import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Update the file path to the correct local file path
file_path = "Group A1.txt"  # Replace with the correct file path if needed

# Load the data
data = pd.read_csv(file_path, sep=r'\s+', engine='python')

# Clean the data
data['R'] = pd.to_numeric(data['R'], errors='coerce')  # Convert to numeric, setting invalid values to NaN
data = data.dropna()  # Drop rows with NaN values

# Extract columns
T = data['T'].values.astype(float)  # Temperature (K)
P_CO2 = data['P_CO2'].values.astype(float)  # Partial pressure of CO2 (bar)
P_H2 = data['P_H2'].values.astype(float)  # Partial pressure of H2 (bar)
R = data['R'].values.astype(float)  # Reaction rate (mol/s)

# Define the enhanced multi-parameter model
def enhanced_model(variables, A, B, C, D, E, F, G, H, I):
    T, P_CO2, P_H2 = variables
    k_T = A * np.exp(-B / T)  # Arrhenius-like dependence on T
    f_P = (C * P_CO2**D + E * P_H2**F + G * P_CO2 * P_H2 +
           H * P_CO2**2 + I * P_H2**2)  # Added higher-order and interaction terms
    return k_T * f_P

# Initial guesses for the parameters
initial_guess = [1e3, 1000, 1, 1, 1, 1, 1, 0.1, 0.1]

# Perform the curve fitting
params, covariance = curve_fit(
    enhanced_model,
    (T, P_CO2, P_H2),
    R,
    p0=initial_guess,
    maxfev=20000
)

# Extract the fitted parameters
A, B, C, D, E, F, G, H, I = params
print(f"Fitted Parameters:\nA = {A}\nB = {B}\nC = {C}\nD = {D}\nE = {E}\nF = {F}\nG = {G}\nH = {H}\nI = {I}")

# Assess fit quality
residuals = R - enhanced_model((T, P_CO2, P_H2), *params)
rmse = np.sqrt(np.mean(residuals**2))
print(f"Root Mean Square Error (RMSE): {rmse}")

# Plot the experimental vs. predicted data
R_pred = enhanced_model((T, P_CO2, P_H2), *params)
plt.figure(figsize=(8, 6))
plt.scatter(T, R, label='Experimental Data', color='blue')
plt.scatter(T, R_pred, label='Fitted Model', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Reaction Rate (mol/s)')
plt.title('Enhanced Multi-parameter Model Fit')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(T, residuals, label='Residuals', color='green')
plt.axhline(0, linestyle='--', color='black')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.legend()
plt.show()
