# Mean-Covariance Modeling of Macaque Foraging Activity

This project models neural population activity recorded from the dorsolateral prefrontal cortex (dlPFC) of freely moving macaques during a foraging task. It uses data associated with *Population coding of strategic variables during foraging in freely moving macaques* to ask how behavioral and spatial variables relate to both predicted neural activity and shared population variability. [Shahidi et al., 2024](https://doi.org/10.1038/s41593-024-01575-w)

---

## Model

The framework separates the predicted population response into a **conditional mean** and **shared residual covariance**:

$$
\mathbf{y}_i \mid \mathbf{x}_i
\sim
\mathcal{N}\!\left(\boldsymbol{\mu}_{\theta}(\mathbf{x}_i),\boldsymbol{\Sigma}_{\phi}\right).
$$

Here, $$\mathbf{x}_i$$ contains task and time-varying spatial variables for trial $$i$$, while $$\mathbf{y}_i$$ contains the neural responses across units and time bins. A transformer-based mean model predicts activity from the covariates. After fitting the mean model, its parameters are frozen while a low-rank latent model learns covariance in the remaining population activity.

The fitted covariance combines shared latent structure with unit- and time-specific residual variance:

$$
\{\Sigma}_{\phi} = \sum_{k=1}^{K}\mathbf{C}_{k}+\mathbf{D},
$$

where $$\mathbf{C}_{k}$$ represents the contribution of latent component $$k$$ and $$\mathbf{D}$$ is diagonal residual noise. Temporal kernels allow the shared components to vary smoothly across the peri-press window.

**The covariance in this implementation is learned across trials; it is not yet conditioned on each trial’s behavioral input.** The mean and covariance models are fitted sequentially rather than jointly.

---

## Data and Preprocessing

The source study recorded dlPFC population activity while unrestrained macaques made self-paced foraging decisions. This repository uses trial-aligned neural responses, strategic task variables, and time-varying position and movement variables. [Shahidi et al., 2024](https://doi.org/10.1038/s41593-024-01575-w)

Neural responses are analyzed in 200 ms bins around button presses. The preprocessing workflow selects foraging events, removes invalid trials and unreliable units, transforms spike counts for Gaussian modeling, and standardizes continuous covariates using training-set statistics. See [`filter_data.m`](filter_data.m) and [`mean-cov-model.ipynb`](mean-cov-model.ipynb) for the implemented workflow.

---

## Model Evaluation

The repository compares three stages:

1. **Baseline:** A time-bin- and unit-specific response template without task covariates.
2. **Mean model:** A transformer-based prediction from task and spatial covariates.
3. **Mean-covariance model:** The fitted mean plus low-rank, temporally structured residual covariance.

The joint covariance also permits **conditional prediction**: observed activity in selected units or earlier time bins can update predictions for other units or bins. A gain from conditioning indicates predictive statistical dependence under the model; it does **not** by itself establish causal or directed neural influence.

---

## Model Interpretation

Permutation-based SHAP analysis attributes **mean-model predictions** to behavioral and spatial inputs. Shuffle-based controls are used to assess whether attribution patterns exceed those expected when trial-level covariate–response alignment is disrupted. These attributions explain the fitted model, not biological causation; correlated covariates and the choice of background data can affect their values.

<p align="center">
  <img src="assets/1.png" alt="Mean-model training and evaluation alongside shared-covariance parameter estimates" width="800">
</p>

**Figure 1.** Combined results from Figures 5 and 6 of the project presentation. The mean-model panels show training and validation trajectories and distributions of per-unit predictive performance. The covariance panels compare latent loadings, temporal length scales, and independent noise variances before and after fitting. Together, they show what was optimized and how the covariance parameters changed; parameter changes alone do not establish held-out predictive benefit or physiological meaning.

---

## Repository Structure

```text
mean-cov-model/
├── mean-cov-model.ipynb     # Modeling, evaluation, and analysis notebook
├── filter_data.m            # MATLAB data selection and filtering
├── data.mat                 # Repository data file
├── src/                     # Supporting Python modules
├── assets/
│   └── 1.png               # Combined model figure
└── LICENSE
```

The main workflow is in [`mean-cov-model.ipynb`](mean-cov-model.ipynb). Supporting code is in [`src/`](src/); see [`filter_data.m`](filter_data.m) for the MATLAB preprocessing step. The paths above refer to the `real-data` branch.

---

## References

- **Data:** Shahidi N, Franch M, Parajuli A, Schrater P, Wright A, Pitkow X, Dragoi V. (2024). *Population coding of strategic variables during foraging in freely moving macaques.* **Nature Neuroscience**, 27, 772–781. [https://doi.org/10.1038/s41593-024-01575-w](https://doi.org/10.1038/s41593-024-01575-w)
- **Methodological inspiration:** Burghardt R. *Investigating Inter-Area Covariance in the Primate Frontoparietal Reach Network via Latent Space Modelling.* MSc thesis, University of Göttingen. **Unpublished**; cite as a thesis rather than a peer-reviewed paper.

This repository is an independent analysis of the foraging data. Its learned latent factors and conditional-prediction results should be validated on held-out data before drawing biological conclusions.
