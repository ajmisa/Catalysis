import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Update the file path to the correct local file path
file_path = "Group A1.txt"  # Replace with the correct file path if needed

# Load the data, skipping the header row with units
data = pd.read_csv(file_path, sep=r'\s+', engine='python')

# Clean the data by ensuring numeric values
try:
    data['R'] = pd.to_numeric(data['R'], errors='coerce')  # Convert to numeric, setting invalid values to NaN
    data = data.dropna()  # Drop rows with NaN values
except KeyError:
    print("Error: 'R' column not found. Check your file format.")
    exit()

# Extract columns
T = data['T'].values.astype(float)  # Temperature (K)
P_CO2 = data['P_CO2'].values.astype(float)  # Partial pressure of CO2 (bar)
P_H2 = data['P_H2'].values.astype(float)  # Partial pressure of H2 (bar)
R = data['R'].values.astype(float)  # Reaction rate (mol/s)

# Define the modified Langmuir-Hinshelwood model
def langmuir_hinshelwood(variables, A, Ea, m, n, K_CO2, K_H2):
    T, P_CO2, P_H2 = variables
    R_gas = 8.314  # Universal gas constant, J/(mol*K)
    k = A * np.exp(-Ea / (R_gas * T))  # Arrhenius equation for k(T)
    numerator = k * (P_CO2**m) * (P_H2**n)
    denominator = 1 + K_CO2 * P_CO2 + K_H2 * P_H2
    return numerator / denominator

# Initial parameter guesses
initial_guess = [1e7, 80000, 1, 1, 0.01, 0.01]

# Perform the curve fitting
params, covariance = curve_fit(
    langmuir_hinshelwood,
    (T, P_CO2, P_H2),
    R,
    p0=initial_guess,
    maxfev=10000
)

# Extract the fitted parameters
A, Ea, m, n, K_CO2, K_H2 = params
print(f"Fitted Parameters:\nA = {A}\nEa = {Ea}\nm = {m}\nn = {n}\nK_CO2 = {K_CO2}\nK_H2 = {K_H2}")

# Assess fit quality
residuals = R - langmuir_hinshelwood((T, P_CO2, P_H2), *params)
rmse = np.sqrt(np.mean(residuals**2))
print(f"Root Mean Square Error (RMSE): {rmse}")

# Plot the experimental vs. predicted data
R_pred = langmuir_hinshelwood((T, P_CO2, P_H2), *params)
plt.figure(figsize=(8, 6))
plt.scatter(T, R, label='Experimental Data', color='blue')
plt.scatter(T, R_pred, label='Fitted Model', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Reaction Rate (mol/s)')
plt.title('Langmuir-Hinshelwood Model Fit')
plt.legend()
plt.show()
