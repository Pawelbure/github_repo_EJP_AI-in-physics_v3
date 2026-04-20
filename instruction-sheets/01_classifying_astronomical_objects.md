# Instruction Sheet 1: Classifying Astronomical Objects (Supervised Learning)

## Learning goal
Use a labeled astrophysics dataset to train an artificial neural network (ANN) that classifies each source as a **STAR**, **GALAXY**, or **QSO**.

## Where to work in this repo
- Notebook: `Classifying-astronomical-objects/Supervised_ML_Classifying-astronomical-objects_ANN.ipynb`
- Input data: `Classifying-astronomical-objects/Dataset_galaxies_quasars_stars/subdf_unsupervised.pickle`

## Physics context
Photometric and survey-derived features contain enough information to separate source classes statistically. Your ANN learns a nonlinear decision boundary in this feature space.

## Step-by-step workflow
1. **Open the notebook and run imports/setup cells first.**
2. **Load the dataset** from the `.pickle` file and inspect:
   - feature columns (input `X`)
   - class labels (target `y`)
   - class balance (`STAR`, `GALAXY`, `QSO` counts)
3. **Create train/test split** (keep `random_state` fixed for reproducibility).
4. **Train the MLP classifier** in the notebook.
5. **Evaluate with physics-aware interpretation:**
   - confusion matrix
   - precision/recall/F1
   - which classes are most often confused and why
6. **Run the width/depth experiments** (double-descent section) and compare train/test error.

## What to report in your solution
- Final test accuracy and macro-F1.
- Confusion matrix with one short interpretation paragraph.
- One model limitation (e.g., class imbalance, domain shift, feature uncertainty).
- One concrete improvement (feature engineering, balancing, regularization, calibration).

## Suggested checks before submission
- Same split seed used throughout.
- No test data used in training or hyperparameter tuning.
- Metrics and plots generated directly from notebook outputs.
