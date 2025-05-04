import numpy as np
import pandas as pd
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
R = data['R'].values.astype(float)  # Reaction rate (mol/s)

# Calculate 1/T and ln(R)
inverse_T = 1 / T  # 1/T
ln_R = np.log(R)   # Natural logarithm of R

# Plot ln(R) vs. 1/T
plt.figure(figsize=(8, 6))
plt.scatter(inverse_T, ln_R, color='blue', label='Data Points')
plt.xlabel('1/T (1/K)')
plt.ylabel('ln(R)')
plt.title('ln(R) vs. 1/T')
plt.grid(True)
plt.legend()
plt.show()
