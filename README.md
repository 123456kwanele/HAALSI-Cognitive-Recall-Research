# Serial-Position Recall and Subsequent Memory Performance in a South African Cohort

**BSc Honours in Big Data Analytics — Research Project**  
**University of the Witwatersrand, Johannesburg**  
**Researcher:** Kwanele Mazibuko  
**Year:** 2026

## 1. Project Overview

This repository contains the Python implementation, processed datasets, statistical analyses, predictive modelling experiments, and visualisations developed for an Honours research project investigating serial-position recall patterns in the Health and Aging in Africa: A Longitudinal Study of an INDEPTH Community in South Africa (HAALSI).

The research examines whether the distribution of recalled words across a memory list provides additional predictive information beyond conventional total recall scores.

Particular attention is given to educational attainment, longitudinal recall patterns, and differences between primacy, middle, and recency recall.

The primary predictive outcome is **Wave 2 delayed word-recall performance**, measured as a continuous score. The project does not predict dementia diagnosis.

## 2. Research Objectives

The study investigates:

1. Associations between educational attainment and serial-position recall.
2. Changes in recall patterns across HAALSI dementia assessment waves.
3. Whether decomposed serial-position features improve prediction relative to total immediate recall.
4. The contribution of education to predictive accuracy.
5. The importance of individual predictors across machine learning models.
6. Whether recall patterns provide useful information when participants have equivalent total recall scores.

## 3. Data Sources

The project uses three original HAALSI data releases:

| Original dataset | Local filename |
|---|---|
| HAALSI Dementia Wave 1 respondent data | `HAALSI_DEMENTIA_Respondent_W1.dta` |
| HAALSI Dementia Wave 2 respondent data | `HAALSI_DEMENTIA_Respondent_W2.dta` |
| HAALSI longitudinal parent dataset, Version 5 | `Longitudinal Public Release Dataset V5.dta` |

These datasets are linked using participant identifiers and processed to construct the analytical samples.

### Processed analytical datasets

The `data/` folder also contains:

| Filename | Purpose |
|---|---|
| `HAALSI_Longitudinal_SPC_Dataset.csv` | Longitudinal serial-position recall observations |
| `HAALSI_Narrowed_SPC_Dataset.csv` | Selected participant characteristics and recall features |
| `HAALSI_Paired_Delayed_Recall_Analysis.csv` | Paired analytical data for follow-up delayed-recall analyses |

The primary complete-case prediction sample contains **504 participants**.

Different analyses may use different sample sizes because of variable availability and eligibility criteria.

### Data access and confidentiality

The original datasets are associated with the HAALSI study and its published data releases.

Before sharing this repository publicly, the applicable dataset access, redistribution, and confidentiality conditions must be checked. The presence of data files in this local project does not automatically authorise their public redistribution.

## 4. Feature Construction

The serial-position curve is represented using three regions:

| Region | Positions | Calculation |
|---|---|---|
| Primacy | 1–3 | Number recalled / 3 |
| Middle | 4–7 | Number recalled / 4 |
| Recency | 8–10 | Number recalled / 3 |

Total immediate recall is the sum of recalled words across all ten positions.

Additional derived features include regional recall differences and serial-position shape measures.

These measures describe recall performance and are not clinical diagnoses.

## 5. Repository Structure

```text
HAALSI_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── HAALSI_DEMENTIA_Respondent_W1.dta
│   ├── HAALSI_DEMENTIA_Respondent_W2.dta
│   ├── Longitudinal Public Release Dataset V5.dta
│   ├── HAALSI_Longitudinal_SPC_Dataset.csv
│   ├── HAALSI_Narrowed_SPC_Dataset.csv
│   └── HAALSI_Paired_Delayed_Recall_Analysis.csv
│
├── scripts/
│   ├── 01_load_data.py
│   ├── 02_understand_dataset.py
│   ├── phase1_comprehensive_eda.py
│   ├── phase1_5_advanced_eda.py
│   ├── phase1_5_tsne_corrected.py
│   ├── section3_1.py
│   ├── section3_2.py
│   ├── section3_3.py
│   ├── section3_4.py
│   ├── section3_5.py
│   ├── section3_6.py
│   ├── section3_7.py
│   ├── section3_8_controlled_spc_comparison.py
│   └── section3_9_same_total_spc.py
│
├── notebooks/
│   ├── 01_dataset_overview.ipynb
│   ├── 02_serial_position_and_longitudinal.ipynb
│   └── 03_prediction_and_interpretability.ipynb
│
├── outputs/
│   └── Saved analysis and modelling results
│
├── figures/
│   └── Generated research visualisations
│
└── env/
    └── Local Python virtual environment
```

The `env/` directory is a local environment and is  not  included in the submitted repository or ZIP archive.

## 6. Analysis Pipeline

### Data loading and exploratory analysis

The following scripts support initial data inspection and exploratory analysis:

- `01_load_data.py`
- `02_understand_dataset.py`
- `phase1_comprehensive_eda.py`
- `phase1_5_advanced_eda.py`
- `phase1_5_tsne_corrected.py`

The exploratory work includes investigation of participant characteristics, recall variables, distributions, missingness, and lower-dimensional representations.

### Section 3.1 — Serial-position and education analysis

Implemented in `section3_1.py`.

Investigates serial-position recall patterns and their associations with educational attainment, including longitudinal and regression-based comparisons.

### Section 3.2 — Predictive modelling

Implemented in `section3_2.py`.

Evaluates conventional machine learning models for predicting follow-up delayed recall and compares performance across feature representations.

### Section 3.3 — Extended modelling

Implemented in `section3_3.py`.

Extends the predictive modelling experiments to additional model configurations, including Kolmogorov–Arnold Networks.

### Section 3.4 — Education ablation

Implemented in `section3_4.py`.

Examines changes in predictive performance when education-related features are included or excluded, alongside education-subgroup comparisons.

### Section 3.5 — Longitudinal and recall-pattern analyses

Implemented in `section3_5.py`.

Examines recall-pattern changes and associated predictive comparisons.

### Section 3.6 — Model interpretability

Implemented in `section3_6.py`.

Uses SHAP-based analysis to explore the relative importance of model features.

SHAP attributions describe model predictions and should not be interpreted as causal effects.

### Section 3.7 — Primary predictive comparison

Implemented in `section3_7.py`.

Compares total-recall and decomposed-recall representations using predictive performance metrics and paired uncertainty analyses.

### Section 3.8 — Controlled serial-position comparison

Implemented in `section3_8_controlled_spc_comparison.py`.

Conducts a controlled comparison of feature representations using repeated cross-validation and multiple model configurations.

### Section 3.9 — Same-total serial-position analysis

Implemented in `section3_9_same_total_spc.py`.

Examines whether differences in serial-position recall patterns are informative when total immediate recall is held constant.

The precise inclusion criteria, comparisons, and statistical procedures are defined in the corresponding implementation.

## 7. Predictive Models and Evaluation

The modelling experiments include:

- Ridge regression.
- Random Forest regression.
- Multilayer Perceptron (MLP).
- Kolmogorov–Arnold Network (KAN).

Performance is evaluated using metrics including:

- Mean absolute error (MAE).
- Root mean squared error (RMSE).
- Coefficient of determination (R²).

The project also includes bootstrap comparisons, education-subgroup analyses, and feature-importance investigations.

Different experimental sections may use different evaluation protocols. Results should therefore be interpreted within their respective experimental designs.

## 8. Environment Setup

### Requirements

The project was developed using Python on Windows with VS Code.

The exact package dependencies are recorded in `requirements.txt`.

### Create the virtual environment

From the project root:

```powershell
python -m venv env
```

Activate it in PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

For an existing working environment, recreation is unnecessary.

## 9. Running the Analysis

All commands should be executed from the `HAALSI_Project` root directory.

An individual script can be launched using:

```powershell
python scripts/section3_7.py
```

For the controlled comparison:

```powershell
python scripts/section3_8_controlled_spc_comparison.py
```

For the same-total recall analysis:

```powershell
python scripts/section3_9_same_total_spc.py
```

These commands assume that each script supports direct execution and that its required input files are present.

**Important:** The complete execution order and any dependencies between scripts must be verified against the implementation before performing a full reproduction.

The `outputs/` folder contains previously generated results that can be inspected without retraining the models.

## 10. Visualisation Notebooks

Three Jupyter notebooks provide supervisor-facing visualisations.

### `01_dataset_overview.ipynb`

Explores analytical sample characteristics, education and age distributions, and missingness.

### `02_serial_position_and_longitudinal.ipynb`

Visualises serial-position curves, education-related patterns, and longitudinal recall changes.

### `03_prediction_and_interpretability.ipynb`

Presents model comparisons, education ablation, bootstrap confidence intervals, and model interpretability results.

These notebooks read existing files from `data/` and `outputs/` rather than retraining the predictive models.

## 11. Research Findings

The analysis investigates the relationship between education and the shape of the serial-position recall curve.

The predictive experiments generally indicate that decomposed recall features do not consistently outperform total immediate recall once other relevant covariates are included.

Education contributes predictive information, although its value varies across models and feature representations.

Longitudinal recall changes must be interpreted cautiously because repeated assessment performance can be influenced by practice effects and other factors.

The results concern subsequent delayed-recall performance and should not be interpreted as evidence of clinical dementia prediction.

## 12. Limitations

The study is subject to several limitations:

- A relatively small complete-case predictive sample.
- Unequal sample sizes across education categories.
- Potential practice effects between assessment waves.
- Possible selection effects from missing follow-up observations.
- Limited generalisability beyond the study population.
- Non-causal interpretation of feature importance.
- Variation in evaluation procedures across experiments.

## 13. Reproducibility and Submission

The repository includes Python scripts, analytical datasets, saved model results, and visualisation notebooks.

For reproducibility, users should retain the documented software environment, use the specified input datasets, and preserve the experimental configurations and random seeds defined in the scripts.

The local virtual environment, Python cache files, and temporary notebook checkpoints are not required for submission.

The research manuscript provides the full methodological discussion and interpretation of the results.

## 14. Author

**Kwanele Mazibuko**  
BSc Honours in Big Data Analytics  
University of the Witwatersrand  
Johannesburg, South Africa

## 15. Acknowledgements

The researcher acknowledges the HAALSI investigators and study participants, the institutions responsible for the data releases, and the academic supervisors who supported the research.

## 16. Data Citation and Use

Any reuse of the datasets should acknowledge the appropriate HAALSI releases and follow the applicable data-use conditions.

This repository is intended to support academic research, methodological transparency, and reproducibility.