# Instruction Sheet 2: Clustering Phases of Matter (Unsupervised Learning)

## Learning goal
Use Monte Carlo Ising-model samples to discover phase structure with unsupervised learning (autoencoder + clustering), without using explicit phase labels during training.

## Where to work in this repo
- Notebook: `Clustering-phases/Unsupervised_ML_Clustering-phases_ANN.ipynb`
- Input data: `Clustering-phases/mc_samples_80.pickle`

## Physics context
At low temperature, spin configurations are ordered; at high temperature, they are disordered. A learned low-dimensional representation should separate these regimes and help identify the transition region.

## Step-by-step workflow
1. **Load Monte Carlo samples** and confirm temperature keys and sample counts.
2. **Build data matrix** where each lattice snapshot is flattened into a feature vector.
3. **Run dimensionality reduction** using the ANN autoencoder in the notebook.
4. **Cluster in latent space** (e.g., HDBSCAN in the notebook).
5. **Visualize latent space** and color points by temperature to inspect phase separation.
6. **Interpret clusters physically:**
   - ordered vs disordered phases
   - ambiguous/transition region near criticality

## What to report in your solution
- A latent-space plot with temperature coloring.
- Cluster labels (or noise points) and their temperature ranges.
- A short argument for where the phase transition occurs.
- A note on uncertainty (finite-size effects, cluster instability, random initialization).

## Suggested checks before submission
- Re-run with the same seed to verify consistency.
- Confirm cluster conclusions are not from one visualization alone.
- Compare at least one alternative latent embedding or clustering parameter choice.
