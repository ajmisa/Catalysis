# Kinetics of a Heterogeneously Catalyzed Reaction – Part 1

This project investigates the kinetics of a heterogeneously catalyzed industrial reaction, focusing on kinetic model development, parameter fitting, and reaction mechanism identification. The work is part of the 6CPT20 course: Catalysis, Science and Technology (Q2 2024) at Eindhoven University of Technology.

## Objective

To analyze batch reactor experimental data for CO₂ and H₂ reacting over a solid catalyst. The aim is to:
- Determine kinetic parameters (rate constant, activation energy, reaction orders)
- Assess model fit quality through residual analysis
- Propose and validate a plausible reaction mechanism

## Key Results

- **Rate law model**: Power-law form  
- **Reaction orders**:  
  - \( n_{\mathrm{H_2}} = 1.389 \)  
  - \( m_{\mathrm{CO_2}} = 0.829 \)
- **Activation energy**: 41.8 kJ/mol  
- **Rate constant**: \( k_0 = 1974.36 \)
- **Model quality**:  
  - R² = 0.993  
  - Residuals show good random distribution under kinetically controlled conditions

## Methodology

- Fitting performed using:
  - Linearized Arrhenius plots
  - Mean squared error (MSE) minimization on \( r \) and \( \ln(r) \)
- Distinction made between kinetically controlled and mass-transfer-limited regimes
- Surface coverage and Langmuir-Hinshelwood kinetics used to derive analytical expressions for rate

## Proposed Mechanism

- Based on methanol synthesis:
  - Stepwise hydrogenation of adsorbed CO₂ species
  - Langmuir-Hinshelwood type adsorption
  - RDS: hydrogenation of H₂COO* to form methanol intermediate
- Derived rate expression:
  \[
  r = k_r \cdot \frac{K_{\mathrm{CO_2}} K_{\mathrm{H_2}}^3 P_{\mathrm{CO_2}} P_{\mathrm{H_2}}^{3/2}}{(1 + K_{\mathrm{CO_2}} P_{\mathrm{CO_2}} + K_{\mathrm{H_2}} P_{\mathrm{H_2}}^{1/2})^4}
  \]

## Reaction Order Ranges (Analytical)

- Hydrogen: \( n_{\mathrm{H_2}} \in [-0.5, 1.5] \)
- CO₂: \( m_{\mathrm{CO_2}} \in [-3, 1] \)

The experimentally fitted orders fall within these bounds, confirming the model’s consistency.

## Files Included

- `Catalysis_assignment_Part_1.pdf`: Report containing full analysis, model derivations, and figures
- `6CPT20_assignment_part_1.pdf`: Assignment description and requirements

## Authors

- Adam Jordani Misa (2208512)  
- Artemis Angelopoulou (2192977)  
Tutor: Emiel Hensen

## Course Info

6CPT20 – Catalysis, Science and Technology  
Eindhoven University of Technology  
Q2 2024
