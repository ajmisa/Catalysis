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

# Define the rate law function
def rate_law_curve_fit(T, P_CO2, P_H2, k0, E_a, m, n):
    return k0 * np.exp(-E_a / (R_gas * T)) * (P_CO2 ** m) * (P_H2 ** n)

# Extract data from the DataFrame
T_data = data['T'].values
P_CO2_data = data['P_CO2'].values
P_H2_data = data['P_H2'].values
R_exp_data = data['R'].values

# Prepare input arrays for curve_fit
X_data = (T_data, P_CO2_data, P_H2_data)

# Initial guesses for k0, E_a, m, n
initial_guesses = [1e4, 50e3, 1.0, 1.0]  # k0, E_a (in J/mol), m, n

# Use curve_fit to optimize parameters
optimal_params, covariance = curve_fit(
    lambda X, k0, E_a, m, n: rate_law_curve_fit(X[0], X[1], X[2], k0, E_a, m, n),
    X_data,
    R_exp_data,
    p0=initial_guesses,
    bounds=(0, np.inf)
)

# Extract optimized parameters
k0_opt, E_a_opt, m_opt, n_opt = optimal_params

# Display results
print("Optimized Parameters:")
print(f"Rate Prefactor (k0): {k0_opt}")
print(f"Activation Energy (E_a, J/mol): {E_a_opt}")
print(f"Reaction Order with respect to P_CO2 (m): {m_opt}")
print(f"Reaction Order with respect to P_H2 (n): {n_opt}")

# Calculate the fitted reaction rates using the optimized parameters
R_fit = rate_law_curve_fit(T_data, P_CO2_data, P_H2_data, k0_opt, E_a_opt, m_opt, n_opt)

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

# Plot log-scale residuals vs Temperature
plt.figure()
plt.scatter(T_data, np.abs(residuals), c='blue', label='Log Residuals')
plt.yscale('log')
plt.axhline(1e-6, color='red', linestyle='--', label='Zero Line')
plt.title('Log-Scale Residuals vs Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Log Residuals')
plt.legend()
plt.grid(True)
plt.show()

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

# Compute necessary transformations
data['1/T'] = 1 / data['T']  # Inverse temperature
data['ln(R)'] = np.log(data['R'])  # Log of reaction rate
data['ln(P_CO2)'] = np.log(data['P_CO2'])  # Log of P_CO2
data['ln(P_H2)'] = np.log(data['P_H2'])  # Log of P_H2

# Define the independent variables and the dependent variable
X = data[['1/T', 'ln(P_CO2)', 'ln(P_H2)']].values  # Independent variables
y = data['ln(R)'].values  # Dependent variable (ln(R))

# Perform least squares regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# Extract coefficients
intercept = model.intercept_  # ln(k0)
coefficients = model.coef_  # [E_a/R, m, n]

# Compute fitted parameters
ln_k0 = intercept
E_a = coefficients[0] * R_gas  # Activation energy
m = coefficients[1]  # Reaction order with respect to P_CO2
n = coefficients[2]  # Reaction order with respect to P_H2

# Display results
print("Fitted Parameters (Least Squares on ln(R)):")
print(f"ln(k0): {ln_k0}")
print(f"k0: {np.exp(ln_k0)}")
print(f"Activation Energy (E_a, J/mol): {E_a}")
print(f"Reaction Order with respect to P_CO2 (m): {m}")
print(f"Reaction Order with respect to P_H2 (n): {n}")

# Predicted ln(R)
ln_R_fit = model.predict(X)

# Residuals
ln_residuals = y - ln_R_fit

# Plot ln(R) residuals vs Temperature
plt.figure()
plt.scatter(data['T'], ln_residuals, c='blue', label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.title('ln(R) Residuals vs Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals (ln(R_exp) - ln(R_fit))')
plt.legend()
plt.grid(True)
plt.show()

# Experimental vs fitted ln(R)
plt.figure()
plt.scatter(y, ln_R_fit, color='blue', label='Data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')
plt.title('Experimental vs Fitted ln(R)')
plt.xlabel('Experimental ln(R)')
plt.ylabel('Fitted ln(R)')
plt.legend()
plt.grid(True)
plt.show()
