# Instruction Sheet 4: Detecting Conservation Laws (Unsupervised Learning)

## Learning goal
Use an unsupervised neural-network workflow to detect low-dimensional invariant structure associated with conservation laws from trajectory/state data.

## Where to work in this repo
- Notebook: `Detecting-conservation-laws/Unsupervised_ML_Detecting-conservation-laws.ipynb`
- Input data: `Detecting-conservation-laws/harmonic_1d.txt`

## Physics context
Conserved quantities constrain motion to lower-dimensional manifolds in phase space. If a model learns this structure robustly, it indicates hidden or known constants of motion.

## Step-by-step workflow
1. **Load and inspect the dataset** (`harmonic_1d.txt`) and identify state variables.
2. **Normalize and preprocess** features as done in the notebook.
3. **Apply PCA/rotations** to examine dominant variance directions.
4. **Train the neural network setup** over the notebook’s noise settings.
5. **Track diagnostic quantities** (loss, explained variance, effective-rank style metrics).
6. **Interpret outcomes as physics statements:**
   - evidence for conserved structure
   - robustness to added noise
   - limits of inference from finite samples

## What to report in your solution
- The key diagnostic plots and their interpretation.
- A short statement of the inferred conservation behavior.
- How noise changed confidence in that inference.
- One potential extension (different potential, dimensionality, or dynamical system).

## Suggested checks before submission
- Data normalization and PCA steps are applied in the documented order.
- Hyperparameters are recorded for reproducibility.
- Claims about conservation are tied to model diagnostics, not just visual impressions.
