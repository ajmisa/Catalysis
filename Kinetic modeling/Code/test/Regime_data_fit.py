import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Update the file path to the correct local file path
file_path = "Group A1.txt"  # Replace with the correct file path if needed

# Load the data
data = pd.read_csv(file_path, sep=r'\s+', engine='python')

# Clean the data
data['R'] = pd.to_numeric(data['R'], errors='coerce')  # Convert to numeric, setting invalid values to NaN
data = data.dropna()  # Drop rows with NaN values

# Extract columns
T = data['T'].values.astype(float)  # Temperature (K)
R = data['R'].values.astype(float)  # Reaction rate (mol/s)

# Calculate 1/T and ln(R)
inverse_T = 1 / T  # 1/T
ln_R = np.log(R)   # Natural logarithm of R

# Exclude data points where 1/T < 0.002
mask = inverse_T >= 0.002
inverse_T_filtered = inverse_T[mask]
ln_R_filtered = ln_R[mask]

# Perform linear regression on the filtered data
slope, intercept, r_value, p_value, std_err = linregress(inverse_T_filtered, ln_R_filtered)

# Calculate R-squared
r_squared = r_value**2
print(f"R-squared: {r_squared:.4f}")

# Residuals
ln_R_predicted = slope * inverse_T_filtered + intercept
residuals = ln_R_filtered - ln_R_predicted

# Plot ln(R) vs. 1/T with the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(inverse_T_filtered, ln_R_filtered, label='Filtered Data', color='blue')
plt.plot(inverse_T_filtered, ln_R_predicted, label=f'Fit: y={slope:.2f}x + {intercept:.2f}', color='red')
plt.xlabel('1/T (1/K)')
plt.ylabel('ln(R)')
plt.title('ln(R) vs. 1/T with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()

# Residual analysis
plt.figure(figsize=(8, 6))
plt.scatter(inverse_T_filtered, residuals, color='green', label='Residuals')
plt.axhline(0, linestyle='--', color='black')
plt.xlabel('1/T (1/K)')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.legend()
plt.grid(True)
plt.show()
