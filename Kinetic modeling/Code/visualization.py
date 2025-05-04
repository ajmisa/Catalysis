import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data file, skipping the first row with units
data = pd.read_csv('Group A1.txt', sep='\t', skiprows=1)

# Rename columns for clarity
data.columns = ['T', 'P_CO2', 'P_H2', 'R']

# Convert columns to the appropriate data types
data['T'] = data['T'].astype(float)  # Temperature in Kelvin
data['P_CO2'] = data['P_CO2'].astype(float)  # Partial pressure of CO2 in bar
data['P_H2'] = data['P_H2'].astype(float)  # Partial pressure of H2 in bar
data['R'] = data['R'].astype(float)  # Reaction rate in mol/s

# Plot 1: Reaction rate vs Temperature for constant pressures
plt.figure()
for pressure_group, group_data in data.groupby(['P_CO2', 'P_H2']):
    plt.scatter(group_data['T'], group_data['R'], label=f"P_CO2={pressure_group[0]}, P_H2={pressure_group[1]}")
plt.title('Reaction Rate vs Temperature for Different Pressures')
plt.xlabel('Temperature (K)')
plt.ylabel('Reaction Rate (mol/s)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Reaction rate vs P_CO2 for constant temperature and P_H2
plt.figure()
for temp_group, group_data in data.groupby(['T']):
    subset = group_data.groupby('P_CO2').mean()  # Average over P_H2
    plt.scatter(subset.index, subset['R'], label=f"T={temp_group} K")
plt.title('Reaction Rate vs P_CO2 at Different Temperatures')
plt.xlabel('P_CO2 (bar)')
plt.ylabel('Reaction Rate (mol/s)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Reaction rate vs P_H2 for constant temperature and P_CO2
plt.figure()
for temp_group, group_data in data.groupby(['T']):
    subset = group_data.groupby('P_H2').mean()  # Average over P_CO2
    plt.scatter(subset.index, subset['R'], label=f"T={temp_group} K")
plt.title('Reaction Rate vs P_H2 at Different Temperatures')
plt.xlabel('P_H2 (bar)')
plt.ylabel('Reaction Rate (mol/s)')
plt.legend()
plt.grid(True)
plt.show()

# Filter data for temperatures below 500 K
data = data[data['T'] < 500]

# Calculate 1/T and ln(R)
data['1/T'] = 1 / data['T']  # Inverse of temperature (1/K)
data['ln(R)'] = np.log(data['R'])  # Natural log of reaction rate

# Plot ln(R) vs 1/T
plt.figure()
plt.scatter(data['1/T'], data['ln(R)'], color='blue', label='Data Points')
plt.title('ln(R) vs 1/T')
plt.xlabel('1/T (1/K)')
plt.ylabel('ln(R)')
plt.legend()
plt.grid(True)
plt.show()