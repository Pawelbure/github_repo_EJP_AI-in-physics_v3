# Instruction Sheets for the AI-in-Physics Notebooks

These instruction sheets accompany four Jupyter notebooks on artificial neural networks (ANNs), unsupervised representation learning, and physics-informed neural networks (PINNs) in physics contexts.

The activities are ordered to match the manuscript structure:

1. **Unsupervised ML:** clustering magnetic phases with an autoencoder.
2. **Unsupervised ML:** detecting conservation-law structure from phase-space data.
3. **Supervised ML:** classifying astronomical objects.
4. **Physics-informed supervised modeling:** modeling a radioactive decay chain with a PINN.

The instructions are written as a single Markdown file so that they can be read directly in **GitHub**, **Binder**, **JupyterLab**, or **Jupyter Notebook**.

---

## Notebook overview and recommended order

| Activity | Topic | Notebook |
|---:|---|---|
| 1 | Unsupervised ML for clustering Ising phases | `Unsupervised_ML_clustering-phases_AE_train_test_validation.ipynb` |
| 2 | Conservation-law structure with unsupervised ML | `Unsupervised_ML_Detecting-conservation-laws_validated_corrected.ipynb` |
| 3 | Supervised ML for classifying astronomical objects | `Supervised_ML_Classifying-astronomical-objects_ANN_validated.ipynb` |
| 4 | Radioactive decay chain modeling with a PINN | `Supervised_ML-chained-radioactive-decay_validated.ipynb` |

---

## General instructions for students

Work through the notebooks in the order above. For each activity:

1. Read the case-study focus and learning goals before opening the notebook.
2. Answer the “Before running the notebook” questions in your own notes.
3. Open the linked notebook and run the cells from top to bottom.
4. Fill in the requested tables using the outputs printed or plotted by the notebook.
5. Finish with the synthesis task.

The goal is not only to obtain a high numerical score or a visually appealing plot. In every activity, the central question is:

> What has the neural network learned, how can we validate that interpretation, and where are the limits of the model?

---

## Data and repository requirements

The notebooks assume that the required data files are available in the same repository or in the relative paths used by the notebooks.

| Notebook | Required data |
|---|---|
| Ising autoencoder | `mc_samples_80.pickle` |
| Conservation-law notebook | `harmonic_1d.txt` |
| Astronomical classification | `Dataset_galaxies_quasars_stars/subdf_unsupervised.pickle` |
| Radioactive decay PINN | no external data file; synthetic data are generated in the notebook |

Generated figures are saved by the notebooks in the current working directory or in an `img/` subfolder.

---

# Activity 1: Unsupervised ML for Clustering Ising Phases

**Notebook:** `Unsupervised_ML_clustering-phases_AE_train_test_validation.ipynb`

## Case-study focus

In this activity, you use an unsupervised ANN autoencoder to compress two-dimensional Ising spin configurations into a low-dimensional latent representation. The autoencoder is trained to reconstruct spin configurations. It is not given temperature, magnetization, Hamiltonian values, or phase labels as prediction targets during training.

The central question is therefore:

> Can the learned autoencoder bottleneck be interpreted physically after training?

The physical system is the finite two-dimensional square-lattice Ising model. Each configuration consists of spins $s_k=\pm 1$ on an $80\times80$ lattice. The conventional order parameter is the magnetization per spin,

$m=\frac{1}{d_x}\sum_{k=1}^{d_x}s_k.$

where $d_x=N^2$. Because the two ferromagnetic branches $m>0$ and $m<0$ are symmetry-related, the order-parameter magnitude $\lvert m\rvert$ is often more useful for identifying ordered versus disordered regimes.

## Learning goals

After completing this activity, you should be able to:

- explain how an autoencoder performs unsupervised representation learning by minimizing reconstruction error;
- distinguish compression quality from physical interpretability;
- interpret a latent space as a candidate representation of physical structure rather than as automatic proof of a phase transition;
- relate the autoencoder bottleneck to magnetization, spin-flip symmetry, and finite-size phase behavior;
- use post-hoc validation to compare learned latent coordinates with conventional thermodynamic observables;
- explain why finite-size systems show rounded and shifted transition indicators rather than an exact singularity.

## Before running the notebook

Answer the following questions:

1. What distinguishes the ferromagnetic and paramagnetic regimes in the Ising model?
2. Why is $\lvert m\rvert$, rather than $m$, useful when comparing the two ordered ferromagnetic branches?
3. What does it mean for an autoencoder to be unsupervised in this notebook?
4. Why would low reconstruction error alone not prove that the autoencoder has learned the phase structure?

## Notebook workflow

### Step 1: Set up packages and reproducibility

Run the notebook section **“Set up packages and reproducibility.”**

Record:

| Quantity | Value |
|---|---|
| PyTorch version | |
| Training device, CPU or GPU | |
| Random seed | |

### Step 2: Load and flatten the spin configurations

Run the section **“Load and flatten the spin configurations.”** The notebook loads `mc_samples_80.pickle`, flattens each $80\times80$ configuration into a vector of length $6400$, and stores the corresponding temperature values for post-hoc interpretation.

Record:

| Quantity | Value |
|---|---|
| Lattice size N | |
| Number of lattice sites D = N² | |
| Number of temperature batches | |
| Shape of `lattice_bank` | |
| Shape of `temperature_bank` | |

### Step 3: Prepare spin values and split the data

Run the section **“Prepare spin values and split into train, validation, and test data.”**

The autoencoder input is the spin configuration itself. The autoencoder target is the same spin configuration. The split has three parts:

- **training set:** used to update the network weights;
- **validation set:** used for early stopping and learning-rate scheduling;
- **test set:** kept separate until final evaluation and post-hoc probing.

Record:

| Split | Shape |
|---|---|
| Full data | |
| Training data | |
| Validation data | |
| Test data | |

Answer:

1. Why is the target identical to the input in an autoencoder?
2. Why is a validation split useful even though the learning task is unsupervised?
3. Why should the test set remain separate until the final evaluation?

### Step 4: Define and train the dense autoencoder

Run the sections **“Define the dense autoencoder”** and **“Train the autoencoder.”**

The model has:

- a dense encoder;
- a five-dimensional bottleneck;
- a dense decoder;
- a `tanh` output layer;
- reconstruction mean-squared-error loss;
- early stopping based on validation loss.

Answer:

1. What are the input and output dimensions of the autoencoder?
2. What is the role of the five-dimensional bottleneck?
3. Why is a `tanh` output physically reasonable for spin-valued data?
4. Does low reconstruction MSE alone prove that the autoencoder has learned phases? Explain.

Record:

| Quantity | Value |
|---|---|
| Latent dimension | |
| Final training MSE | |
| Best validation MSE | |
| Number of epochs trained | |

### Step 5: Inspect the latent representation

Run the post-hoc latent-space section and the 2-D PCA visualization of the five-dimensional bottleneck.

The notebook projects the five-dimensional latent representation to two principal components and plots train and test data colored by:

- temperature;
- signed magnetization.

Answer:

1. Do low- and high-temperature configurations occupy visibly different regions?
2. Do the colors suggest a relation between latent position and magnetization?
3. Does the ferromagnetic regime appear as one compact cluster or as two symmetry-related branches?
4. Why is this visualization only post-hoc validation rather than part of the autoencoder training?

### Step 6: Post-hoc validation by predicting the order-parameter magnitude

Run the section **“Regression probe: predict the order-parameter magnitude $\lvert M\rvert$.”**

The notebook trains a simple supervised readout from the frozen autoencoder bottleneck to the normalized order-parameter magnitude $\lvert m\rvert$. This does not change the autoencoder weights. It asks whether the learned representation contains information about a conventional physical observable.

Record:

| Split | $R²$ | MAE($|m|$) | RMSE($|m|$) |
|---|---:|---:|---:|
| Train | | | |
| Validation | | | |
| Test | | | |

Answer:

1. Does the bottleneck contain enough information to recover $\lvert m\rvert$?
2. Why does the readout use quadratic latent features?
3. Does a high $R^2$ mean that the autoencoder discovered the Ising Hamiltonian? Explain.
4. What is the difference between “the latent space is physically useful” and “the model has learned the physics”?

## Synthesis task

Write a 300–500 word explanation of what the autoencoder did and did not learn. Your explanation should explicitly distinguish:

- reconstruction;
- representation learning;
- post-hoc physical validation;
- conventional statistical-physics interpretation.

## Optional extension

Group the encoded test data by temperature and compute the average $\langle \lvert m\rvert \rangle_T$. Compare this conventional order-parameter trend with the latent-space visualization. Does the autoencoder representation sharpen, blur, or distort the finite-size crossover?

---

# Activity 2: Conservation-Law Structure with Unsupervised ML

**Notebook:** `Unsupervised_ML_Detecting-conservation-laws_validated_corrected.ipynb`

## Case-study focus

In this activity, you use an unsupervised denoising ANN to learn a correction field around the phase-space trajectory of a one-dimensional harmonic oscillator. The network is trained to remove artificial noise from data points. It is not given the oscillator energy, Hamiltonian, or equation of motion during training.

The central question is:

> Is the learned pullback field consistent with the constant-energy manifold of the oscillator?

For a harmonic oscillator with mass $m$ and spring constant $c$, using the notebook notation of position $x$ and velocity $v$, the energy is

$H(x,v)=\frac{1}{2}cx^2+\frac{1}{2}mv^2.$

A trajectory at fixed energy forms a one-dimensional curve in the two-dimensional phase space. The ML task is therefore connected to the physical idea that conservation laws constrain the accessible states of a system to a lower-dimensional manifold.

## Learning goals

After completing this activity, you should be able to:

- explain how a denoising ANN can learn a local correction field from perturbed data;
- relate the learned pullback field to the idea of a lower-dimensional physical manifold;
- distinguish unsupervised manifold learning from explicit discovery of a conservation law;
- interpret PCA-based explained-variance diagnostics and effective dimensionality;
- validate the learned representation by checking intrinsic dimension and energy consistency after training;
- explain why physical validation must be performed separately when the training loss contains no explicit physics.

## Before running the notebook

Answer:

1. What is conserved in an undamped harmonic oscillator?
2. What is the shape of a fixed-energy oscillator trajectory in phase space?
3. Why is a fixed-energy trajectory lower-dimensional than the full phase space?
4. What would it mean for a learned vector field to “point back” toward the physical trajectory?

## Notebook workflow

### Step 1: Set up the notebook and load the phase-space data

Run the setup and data-loading cells. The notebook loads `harmonic_1d.txt`, which contains samples of position and velocity along an oscillator trajectory.

Record:

| Quantity | Value |
|---|---|
| Number of samples | |
| Input dimension | |
| Qualitative shape of the raw phase-space plot | |

### Step 2: Preprocess the data

The notebook normalizes each coordinate to zero mean and unit variance and then applies PCA as a rotation. It also stores the inverse transformations for later physics validation.

Answer:

1. Why is preprocessing useful for ANN training?
2. Why does the notebook store the inverse transformations?
3. Why should physical validation be performed in the original physical coordinates when possible?

### Step 3: Train the denoising network across noise levels

Run the training section. The network receives noisy inputs and is trained to output the negative of the artificial perturbation. In other words, the network learns a local correction vector.

Answer:

1. What is the input to the network during training?
2. What is the target output?
3. Why does this training objective not explicitly use energy conservation?
4. How does the noise scale $\sigma$ affect the physical meaning of the learned correction?

### Step 4: Interpret explained variance and effective dimensionality

After training, the notebook performs stochastic walks and computes explained-variance ratios with PCA. These diagnostics ask whether the corrected samples concentrate along a low-dimensional structure.

Complete:

| Noise scale | Dominant PCA behavior | Interpretation |
|---|---|---|
| Small sigma | | |
| Intermediate sigma | | |
| Large sigma | | |

Answer:

1. For which noise scales does the learned representation appear most one-dimensional?
2. Why might very small or very large noise scales be less informative?
3. What does effective rank tell us that a phase-space plot alone does not?

### Step 5: Inspect the learned vector field

Run the vector-field visualization. The arrows show the correction predicted by the trained network in processed phase space.

Answer:

1. Do the arrows point toward the trajectory manifold?
2. Are there regions where the correction field is unreliable?
3. Why should a visual vector field be treated as evidence, but not proof, of conservation-law structure?
4. If noisy training samples are shown in the figure, how do they help you judge where the network was actually trained?

### Step 6: Validation procedure 1 — local intrinsic dimension after pullback

Run the section **“Validation 1: local intrinsic dimension after pullback.”**

The notebook samples local noisy point clouds, applies the pullback network, and computes local PCA. A successful result should make the corrected cloud approximately line-like.

Record:

| Diagnostic | Value |
|---|---:|
| Mean participation-ratio dimension | |
| Median number of PCs for 95% variance | |
| Mean variance explained by first PC | |

Interpret whether the corrected local clouds support a one-dimensional manifold interpretation.

### Step 7: Validation procedure 2 — energy consistency of corrected states

Run the section **“Validation 2: energy consistency of corrected states.”**

Although the network was not trained on $H(x,v)$, corrected points should lie closer to the constant-energy curve if the learned pullback is physically meaningful.

Record:

| Diagnostic | Value |
|---|---:|
| Mean &vert;H(noisy) - E0&vert; | |
| Mean &vert;H(corrected) - E0&vert; | |
| Relative energy-error reduction | |
| Mean correction/energy-normal alignment | |

Answer:

1. Does the pullback reduce the energy error?
2. Does the correction direction align with the expected normal direction to the energy contour?
3. Does this prove that the network has discovered the Hamiltonian? Why or why not?

## Synthesis task

Write a 300–500 word interpretation addressing the following claim:

> The ANN has discovered the conservation law of the harmonic oscillator.

Argue for a more careful version of this claim. Your answer should mention the training loss, the learned correction field, the constant-energy manifold, and the two validation procedures.

## Optional extension

Change the oscillator data, noise scale, or network width. Investigate when the pullback field becomes physically misleading. Report one failure mode.

---

# Activity 3: Supervised ML for Classifying Astronomical Objects

**Notebook:** `Supervised_ML_Classifying-astronomical-objects_ANN_validated.ipynb`

## Case-study focus

In this activity, you use a supervised artificial neural network classifier to classify astronomical objects as `STAR`, `GALAXY`, or `QSO`. The aim is not only to obtain a high classification score. The central question is what kind of astrophysical information the ANN uses and how we can validate whether the numerical output has a physically meaningful interpretation.

The notebook uses photometric and morphology-related input features, including SDSS optical bands `u, g, r, i, z`, WISE infrared bands `w1, w2, w3, w4`, and the morphology proxy `resolved`. The classifier is therefore a data-driven model of astronomical source type: it learns a mapping from measured source properties to class labels.

## Learning goals

After completing this activity, you should be able to:

- describe an ANN classifier as a parameterized function $f_\theta$ that maps measured astronomical features to class labels;
- distinguish training, validation, and test data and explain why the test set must remain independent of model development;
- interpret classification accuracy, balanced accuracy, macro F1-score, and the confusion matrix as complementary validation tools;
- connect classification errors to astrophysical source properties, such as point-like versus extended morphology and optical/infrared color information;
- use feature ablation to ask whether the ANN relies on physically interpretable feature groups;
- identify limitations of the model, including class imbalance, feature correlations, and the risk of interpreting predictive success as physical explanation.

## Before running the notebook

Briefly answer the following questions in your notes:

1. Which of the three classes do you expect to be easiest to identify from morphology alone? Explain your reasoning.
2. Why might quasars be confused with stars, even though they are physically very different objects?
3. What kind of information is added by infrared WISE bands compared with optical SDSS bands?
4. Why is overall accuracy alone insufficient for judging a classifier trained on an imbalanced astronomical catalogue?

## Notebook workflow

### Step 1: Load and inspect the data

Run the setup cells and inspect the loaded arrays `tsne_subsample` and `tsne_subsample_classes`.

Record:

| Quantity | Value |
|---|---|
| Number of samples | |
| Number of input features | |
| Three class labels | |
| Class counts or class balance | |

### Step 2: Construct the supervised learning problem

Run the train/validation/test split cell. In your notes, draw a simple diagram showing which data are used for training and which data are reserved for validation and final evaluation.

Answer:

1. Why does using test data during model tuning lead to an overly optimistic assessment?
2. In the notebook, where are `X_train`, `X_val`, and `X_test` defined?
3. What would a strict validation workflow use for monitoring during training?

### Step 3: Train the ANN classifier

Run the ANN training cell. The model is a feed-forward multilayer perceptron with two hidden layers. While the model is training, monitor the train and test error curves shown by the notebook.

Answer:

1. Does the training error decrease?
2. Does the held-out error decrease in the same way?
3. Is there evidence of overfitting?
4. What does the ANN learn mathematically: a physical law, a statistical association, or both? Explain.

### Step 4: Evaluate final test-set performance

Run the cell **“Final test-set predictions.”** It prints accuracy, balanced accuracy, macro F1-score, and the classification report.

Record:

| Metric | Value |
|---|---:|
| Accuracy | |
| Balanced accuracy | |
| Macro F1-score | |
| Class with lowest recall | |
| Class with lowest precision | |

Interpret the table. A good result is not simply a high overall accuracy; it should also perform adequately for each class.

### Step 5: Validation procedure 1 — confusion matrix and class-wise errors

Run **“Validation procedure 1: confusion matrix and class-wise performance.”**

Read the normalized confusion matrix row-wise: each row corresponds to a true class, and each entry gives the fraction predicted as each class.

Answer:

1. Which class is classified most reliably?
2. Which pair of classes is confused most often?
3. Give a possible astrophysical explanation for this confusion.
4. Does the confusion matrix support the claim that the model has learned physically meaningful distinctions, or only that it has learned statistical separations?

### Step 6: Validation procedure 2 — feature ablation

Run **“Validation procedure 2: feature-ablation analysis.”**

The notebook compares the full model with models trained using different feature sets, including:

- all features;
- without `resolved`;
- without WISE bands;
- optical bands plus `resolved` only.

Complete:

| Feature setting | Balanced accuracy | Macro F1 | Main interpretation |
|---|---:|---:|---|
| All features | | | |
| Without `resolved` | | | |
| Without WISE bands | | | |
| Optical + `resolved` only | | | |

Answer:

1. Which feature group seems most important for the classifier?
2. Does removing a feature group prove that the removed feature causes the classification decision? Why or why not?
3. How do feature correlations complicate the interpretation of the ablation study?

## Synthesis task

Write a short interpretation of the model in 300–500 words. Your interpretation should include:

- the supervised learning task;
- the physical meaning of the input feature groups;
- the strongest and weakest validation results;
- one limitation of the ANN classifier;
- one way in which traditional astrophysical reasoning is still needed.

## Optional extension

Run the model-complexity or double-descent section. Compare how changing width and depth changes train and test error. Discuss whether larger models necessarily produce better physical understanding.

---

# Activity 4: Radioactive Decay Chain Modeling with a PINN

**Notebook:** `Supervised_ML-chained-radioactive-decay_validated.ipynb`

## Case-study focus

In this activity, you model a radioactive decay chain

$N_1 \rightarrow N_2 \rightarrow N_3$

with a direct supervised ANN and a physics-informed neural network (PINN). The system is governed by

$\frac{dN_1}{dt}=-\lambda_1N_1,\qquad \frac{dN_2}{dt}=\lambda_1N_1-\lambda_2N_2,\qquad \frac{dN_3}{dt}=\lambda_2N_2.$

The direct ANN learns from a small number of data points. The PINN additionally uses the differential-equation residuals as part of the loss. The central question is whether encoding physical structure in the learning objective improves the model’s interpolation, extrapolation, and physical plausibility.

## Learning goals

After completing this activity, you should be able to:

- describe the decay-chain equations as a physics-based model of coupled time evolution;
- explain how a direct ANN represents a mapping $t\mapsto (N_1,N_2,N_3)$;
- explain how a PINN adds physical constraints through differential-equation residuals;
- compare data-driven fitting with physics-informed fitting under sparse-data conditions;
- evaluate model outputs using both numerical error and physical criteria such as residual size, extrapolation behavior, non-negativity, and total-population consistency;
- identify limitations of PINNs, including sensitivity to loss balancing, collocation-point choice, optimizer behavior, and the difference between satisfying a residual approximately and understanding the physics.

## Before running the notebook

Answer:

1. What physical process is represented by each term in the three differential equations?
2. What should happen to $N_1(t)$, $N_2(t)$, and $N_3(t)$ over time?
3. Is the total population $N_1+N_2+N_3$ conserved in this closed decay chain? Show this from the equations.
4. Why might a direct ANN trained on only a few points extrapolate poorly?

## Notebook workflow

### Step 1: Simulate reference data

Run **“Simulate data.”** The notebook uses

$\lambda_1=0.5,\qquad \lambda_2=0.3.$

with initial values $N_1(0)=0.8$, $N_2(0)=0.1$, and $N_3(0)=0.5$. The reference solution is generated by numerical Euler integration over the training interval.

Record:

| Quantity | Value |
|---|---:|
| Training time interval | |
| Extrapolation interval | |
| Number of sparse observations | |
| Number of physics collocation points | |
| Initial total population N1(0) + N2(0) + N3(0) | |

### Step 2: Inspect the ANN architecture

Run **“Define ANN.”** The model maps one input, time $t$, to three outputs, $(N_1,N_2,N_3)$. The hidden layers use `tanh` activations.

Answer:

1. Why is the output dimension three?
2. What does it mean to treat the ANN as a parameterized function $f_\theta(t)$?
3. Does the architecture itself enforce the decay equations, non-negativity, or conservation of total population?

### Step 3: Train the direct ANN

Run **“Direct solution model.”** This model is trained only on the sparse observed data values.

Answer:

1. How many data points are used for training?
2. Does a low training loss at the observed points guarantee accurate behavior between the points?
3. What kind of information is missing from the direct ANN loss?

### Step 4: Train the PINN

Run **“PINN.”** The PINN loss contains two parts:

$\mathcal{L}_{\mathrm{PINN}} = \mathcal{L}_{\mathrm{data}} + \mathcal{L}_{\mathrm{ODE}}.$

where $\mathcal{L}_{\mathrm{ODE}}$ penalizes violations of the decay-chain equations.

Answer:

1. How are automatic derivatives used to compute $dN_i/dt$?
2. What does each residual term $r_1,r_2,r_3$ measure physically?
3. Why can an ODE residual act as a learning bias?
4. Does the PINN use physics as data, as architecture, or as a loss constraint?

### Step 5: Compare interpolation and extrapolation

Run **“Evaluate model and predict time series”** and the plotting cell. The notebook compares the direct ANN and the PINN both within the training interval and in an extrapolation interval.

Complete:

| Diagnostic | Direct ANN | PINN |
|---|---:|---:|
| Training-region/interpolation MSE | | |
| ODE residual MSE | | |
| Extrapolation MSE | | |
| Qualitative extrapolation behavior | | |

Answer:

1. Which model follows the reference solution better in the extrapolation region?
2. Does the PINN improvement come from more observed data, more physics, or both?
3. Are there time regions where either model gives physically implausible predictions?

### Step 6: Add physical consistency checks

Use the printed quantitative validation and plotted predictions to check:

- whether $N_1,N_2,N_3$ remain non-negative;
- whether $N_1+N_2+N_3$ remains approximately constant;
- whether $N_1$ decreases monotonically;
- whether $N_3$ increases monotonically in the modeled interval.

These checks are not merely numerical. They test whether the learned function respects qualitative physics.

Record:

| Physical diagnostic | Direct ANN | PINN |
|---|---:|---:|
| Minimum predicted population | | |
| Number of negative predicted values | | |
| Mean absolute total-population error | | |
| Maximum absolute total-population error | | |
| Final total-population error | | |

## Synthesis task

Write a 300–500 word comparison of the direct ANN and the PINN. Your explanation should include:

- what information each model receives during training;
- how the loss functions differ;
- which model generalizes better and why;
- one physical diagnostic beyond MSE;
- one limitation of the PINN approach.

## Optional extension

Vary the number and location of observed data points or physics collocation points. Investigate whether the PINN still outperforms the direct ANN when the observed points are clustered in one part of the time interval. Explain the result in terms of data coverage and physics-based constraints.
