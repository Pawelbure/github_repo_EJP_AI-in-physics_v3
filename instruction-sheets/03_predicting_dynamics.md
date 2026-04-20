# Instruction Sheet 3: Predicting Radioactive Decay Dynamics (Supervised + PINN)

## Learning goal
Model chained radioactive decay using two approaches:
1) a direct supervised ANN fit, and
2) a physics-informed neural network (PINN) constrained by decay ODEs.

## Where to work in this repo
- Notebook (3-isotope chain): `Predicting-dynamics/Supervised_ML-chained-radioactive-decay.ipynb`
- Notebook (many isotopes): `Predicting-dynamics/Supervised_ML-chained-radioactive-decay-many-isotopes.ipynb`

## Physics context
For isotopes in a chain, populations satisfy coupled first-order differential equations with decay constants \(\lambda_i\). PINNs enforce this structure while fitting data.

## Step-by-step workflow
1. **Generate or load decay data** from the notebook simulation cells.
2. **Define ANN architecture** and verify output dimension matches isotope count.
3. **Train the direct model** using supervised loss on sampled time points.
4. **Train the PINN model** using:
   - data-matching term
   - ODE residual/physics term from automatic differentiation
5. **Evaluate both models** on full time grid:
   - trajectory error (MSE)
   - qualitative shape agreement
   - physically valid behavior (non-negativity/trends)
6. **Compare simple vs many-isotope case** and discuss scaling difficulty.

## What to report in your solution
- Side-by-side plots of predicted vs reference trajectories.
- Quantitative comparison (MSE or equivalent) for direct ANN vs PINN.
- One case where physics constraints improve extrapolation/stability.
- One practical challenge in training PINNs (loss weighting, stiffness, optimization).

## Suggested checks before submission
- Confirm units and time range are consistent.
- Ensure gradient-based derivative terms are computed with `requires_grad=True` time tensors.
- Verify reported metrics use the same evaluation grid for both models.
