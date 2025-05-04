import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from scipy.optimize import curve_fit

# Load the data file
data = pd.read_csv('Group A1.txt', sep='\t', skiprows=1)
data.columns = ['T', 'P_CO2', 'P_H2', 'R']

# Convert columns to appropriate data types
data['T'] = data['T'].astype(float)
data['P_CO2'] = data['P_CO2'].astype(float)
data['P_H2'] = data['P_H2'].astype(float)
data['R'] = data['R'].astype(float)

# Log-transform the reaction rate
data['ln_R'] = np.log(data['R'])

# Features and target
X = data[['T', 'P_CO2', 'P_H2']]
y = data['ln_R']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Feature importance
print("Feature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Partial Dependency Analysis
# Temperature dependency: Keep P_CO2 and P_H2 constant
P_CO2_const = 10  # Arbitrary constant value for P_CO2
P_H2_const = 30   # Arbitrary constant value for P_H2
T_range = np.linspace(data['T'].min(), data['T'].max(), 100)

# Generate predictions for varying temperature
X_temp = pd.DataFrame({'T': T_range, 'P_CO2': P_CO2_const, 'P_H2': P_H2_const})
ln_R_pred = model.predict(X_temp)

# Fit to Arrhenius equation
def arrhenius(T, ln_k0, Ea):
    return ln_k0 - Ea / (8.314 * T)  # R_gas = 8.314 J/(mol*K)

params, _ = curve_fit(arrhenius, T_range, ln_R_pred)
ln_k0, Ea = params

# Reaction order analysis: Keep T constant and vary P_CO2 and P_H2
T_const = 500  # Arbitrary constant value for T
P_CO2_range = np.linspace(data['P_CO2'].min(), data['P_CO2'].max(), 100)
P_H2_range = np.linspace(data['P_H2'].min(), data['P_H2'].max(), 100)

# Generate predictions for varying P_CO2
X_CO2 = pd.DataFrame({'T': T_const, 'P_CO2': P_CO2_range, 'P_H2': P_H2_const})
ln_R_CO2 = model.predict(X_CO2)

# Generate predictions for varying P_H2
X_H2 = pd.DataFrame({'T': T_const, 'P_CO2': P_CO2_const, 'P_H2': P_H2_range})
ln_R_H2 = model.predict(X_H2)

# Fit power laws for P_CO2 and P_H2
def power_law(P, ln_k, n):
    return ln_k + n * np.log(P)

params_CO2, _ = curve_fit(power_law, P_CO2_range, ln_R_CO2)
params_H2, _ = curve_fit(power_law, P_H2_range, ln_R_H2)

ln_k_CO2, m = params_CO2
ln_k_H2, n = params_H2

# Print results
print("\nExtracted Parameters:")
print(f"Rate Prefactor (ln_k0): {ln_k0}")
print(f"Activation Energy (Ea): {Ea} J/mol")
print(f"Reaction Order with respect to P_CO2 (m): {m}")
print(f"Reaction Order with respect to P_H2 (n): {n}")

# Plot temperature dependency fit
plt.figure()
plt.plot(T_range, ln_R_pred, label='XGBoost Predictions')
plt.plot(T_range, arrhenius(T_range, ln_k0, Ea), '--', label='Arrhenius Fit')
plt.title('Temperature Dependency')
plt.xlabel('Temperature (K)')
plt.ylabel('ln(R)')
plt.legend()
plt.grid(True)
plt.show()
