import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load your experimental data
# Load the data file
data = pd.read_csv('Group A1.txt', sep='\t', skiprows=1)
data.columns = ['T', 'P_CO2', 'P_H2', 'R']
data = data.astype(float)

# Constants
R = 8.314  # Universal gas constant, J/(mol·K)

# Mechanistic rate law based on Exercise 3
def methanation_rate(X, k, K_CO2, K_H2):
    """
    Rate law for the methanation reaction.
    Args:
        X: Tuple of (T, P_CO2, P_H2)
        k: Rate constant
        K_CO2: Adsorption equilibrium constant for CO2
        K_H2: Adsorption equilibrium constant for H2
    Returns:
        Reaction rate (r)
    """
    T, P_CO2, P_H2 = X
    # Rate expression
    return k * (K_CO2 * P_CO2 / (1 + K_CO2 * P_CO2 + K_H2 * P_H2)) * \
           ((K_H2 * P_H2 / (1 + K_CO2 * P_CO2 + K_H2 * P_H2))**2)

# Prepare data
T = data['T'].values
P_CO2 = data['P_CO2'].values
P_H2 = data['P_H2'].values
R_exp = data['R'].values

# Combine independent variables into a single array for curve_fit
X_data = (T, P_CO2, P_H2)

# Initial parameter guesses
initial_guess = [1e3, 0.1, 0.1]  # k, K_CO2, K_H2

# Fit the model to the experimental data
popt, pcov = curve_fit(methanation_rate, X_data, R_exp, p0=initial_guess, bounds=(0, np.inf))

# Extract fitted parameters
k_fit, K_CO2_fit, K_H2_fit = popt

# Compute residuals and R-squared
R_fit = methanation_rate(X_data, *popt)
residuals = R_exp - R_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((R_exp - np.mean(R_exp))**2)
r_squared = 1 - (ss_res / ss_tot)

# Print results
print("Fitted Parameters:")
print(f"k: {k_fit:.2e}")
print(f"K_CO2: {K_CO2_fit:.2e}")
print(f"K_H2: {K_H2_fit:.2e}")
print(f"R-squared: {r_squared:.4f}")

# Plot experimental vs fitted reaction rates
plt.figure()
plt.scatter(R_exp, R_fit, label='Fitted Data')
plt.plot([min(R_exp), max(R_exp)], [min(R_exp), max(R_exp)], 'r--', label='Ideal Fit')
plt.xlabel('Experimental Rate (R_exp)')
plt.ylabel('Fitted Rate (R_fit)')
plt.legend()
plt.title('Experimental vs Fitted Reaction Rates')
plt.grid(True)
plt.show()

# Plot residuals
plt.figure()
plt.scatter(T, residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residuals vs Temperature')
plt.grid(True)
plt.show()
