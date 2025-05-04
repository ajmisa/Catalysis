# Finite Volume Modeling of Catalyst Pellets – Part 2

This project develops numerical models for the steady-state concentration profiles in cylindrical catalyst pellets using the finite volume method (FVM). The focus is on simulating reaction-diffusion systems for unimolecular and bimolecular reactions under isothermal and diffusion-dominated assumptions.

## Objective

- Solve the steady-state reaction-diffusion equation in cylindrical geometry
- Compare numerical solutions to analytical results using Bessel functions
- Evaluate convergence behavior and numerical accuracy for various grid sizes and Thiele modulus values
- Extend the model to bimolecular systems (A + B → C) and analyze differences from unimolecular cases

## Reactions Studied

- **Unimolecular Reaction:** A → C  
  Modeled with both analytical (Bessel function) and numerical finite volume approaches

- **Bimolecular Reaction:** A + B → C  
  Modeled numerically using finite volume discretization. Comparisons made to the A → C case under conditions of excess B.

## Key Features

- Finite volume discretization in radial dimension with proper boundary conditions:
  - Symmetry at pellet center (dC/dξ = 0)
  - Fixed concentration at pellet surface (C = Cₐ,₀)

- Numerical implementation in Python using:
  - Linear system solvers (NumPy)
  - Modified Bessel function (SciPy) for analytical solution
  - Error metrics: L₁, L₂, and L∞ norms

- Visualization and analysis of:
  - Convergence with increasing grid resolution
  - Effect of Thiele modulus (Φ) on solution behavior
  - Concentration profiles for species A and B
  - Sensitivity to surface concentration, reaction rate, and particle radius

## Repository Contents

- `C_pt2_Q1.py` – Python script implementing all numerical models and visualizations
- `Catalysis_Assignment_part_2.pdf` – Full report with derivations, figures, and interpretation
- `6CPT20_assignment_part_2.pdf` – Assignment brief

## Summary of Findings

- The finite volume method yields excellent agreement with analytical solutions for unimolecular reactions
- Error metrics decrease with increasing grid size, confirming convergence
- Bimolecular models show that as B becomes increasingly in excess, the A + B → C system approaches the A → C behavior
- Error sensitivity increases with Thiele modulus, surface concentration, reaction rate, and particle radius

