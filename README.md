# Machine Learning-Based Prediction of Degradation in PEM Fuel Cells Using Time-Series Operational Data

MSc Data Science and Artificial Intelligence dissertation project at Sheffield Hallam University.

## Project overview

This project develops and evaluates a machine-learning analytical framework for predicting Proton Exchange Membrane Fuel Cell (PEMFC) stack voltage from contemporaneous operational time-series measurements, assessing generalisation to unseen later durability stages, and interpreting the resulting evidence alongside comparable-current polarization measurements to evaluate long-term performance deterioration.

A central methodological principle is that **instantaneous voltage prediction is kept analytically distinct from degradation assessment**. Machine-learning residuals and feature attributions are not treated as direct physical degradation measurements.

## Research questions

1. **RQ1:** How effectively can machine-learning models trained on earlier PEMFC operational data predict stack voltage at unseen later durability stages?
2. **RQ2:** How can model predictions and comparable-current polarization measurements be interpreted together to assess long-term PEMFC degradation behaviour?

## Analytical workflow

1. Data preparation and quality validation
2. Exploratory and subsystem behaviour analysis
3. Target-oriented association analysis
4. PEMFC-informed feature engineering
5. Redundancy and multicollinearity assessment
6. Fold-wise XGBoost-Boruta feature selection
7. Ridge, Random Forest and XGBoost regression
8. Expanding chronological validation
9. Untouched later-stage evaluation at 900, 950 and 1000 h
10. SHAP explainability of the development-selected XGBoost model
11. Comparable-current polarization assessment of deterioration and rest-associated recovery
12. Integrated interpretation with explicit causal and degradation claim boundaries

## Dataset

The study uses the long-term dynamic durability dataset published by Zuo et al. (2021). It contains approximately 1000 h of single-PEMFC dynamic operation, periodic polarization measurements, and measurements following a 12 h rest.

The raw dataset is **not redistributed in this repository** because of its size and source provenance. Obtain it from the original publication/data repository and place the CSV files under `data/raw/` as described in `data/README.md`.

**Dataset citation:**

Zuo, J., Lv, H., Zhou, D., Xue, Q., Jin, L., Zhou, W., Yang, D., & Zhang, C. (2021). Long-term dynamic durability test datasets for single proton exchange membrane fuel cell. *Data in Brief, 35*, 106775. https://doi.org/10.1016/j.dib.2021.106775

## Dataset and feature summary

- 20 dynamic durability stages: 50–1000 h in 50 h increments
- 3,629,680 cleaned operational observations
- 18 cleaned operational variables before feature engineering
- Six PEMFC-informed engineered descriptors
- 20 final predictors
- All 20 predictors confirmed by XGBoost-Boruta across all four chronological folds

## Chronological evaluation design

| Fold | Training stages | Validation stages |
|---|---|---|
| 1 | 50–450 h | 500–550 h |
| 2 | 50–550 h | 600–650 h |
| 3 | 50–650 h | 700–750 h |
| 4 | 50–750 h | 800–850 h |

The final later-stage holdout consists of **900, 950 and 1000 h** and was excluded from model development and tuning.

## Principal machine-learning results

| Model | Development RMSE | Holdout RMSE | Holdout MAE | Holdout R² |
|---|---:|---:|---:|---:|
| Ridge | 17.102 mV | 10.694 mV | 7.834 mV | 0.987909 |
| Random Forest | 9.737 mV | **8.403 mV** | **6.195 mV** | **0.992535** |
| XGBoost | **9.638 mV** | 8.788 mV | 6.357 mV | 0.991834 |

XGBoost was selected as the primary model using development evidence before final holdout comparison. Random Forest subsequently achieved a slightly lower aggregate holdout RMSE; the two nonlinear ensembles are therefore interpreted as closely competitive for this dataset rather than as evidence of universal superiority of either algorithm.

## Explainability

SHAP analysis of the frozen XGBoost model identified **current** as the dominant predictive contributor, accounting for approximately **71.26%** of summed mean absolute attribution. SHAP is interpreted as an explanation of model predictive behaviour and **not as evidence of electrochemical causality**.

## Long-term performance deterioration

Comparable-current polarization measurements provide the primary evidence of long-term performance deterioration. At 1000 h:

| Current | Apparent loss | Recovery after 12 h rest | Persistent post-rest loss |
|---:|---:|---:|---:|
| 5 A | 60 mV | 24 mV | 36 mV |
| 25 A | 117 mV | 50 mV | 67 mV |
| 45 A | 126 mV | 33 mV | 93 mV |

The evidence indicates long-term net performance deterioration with load dependence across the investigated currents, non-monotonic stage-to-stage behaviour, and partial rest-associated recovery. Persistent post-rest loss is not automatically interpreted as proven irreversible physical degradation.

## Repository structure

```text
PEMFC_Dissertation/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── figures/
├── results/
├── models/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducibility

The analysis was implemented in Python/JupyterLab using pandas, NumPy, SciPy/statsmodels, scikit-learn, XGBoost, SHAP and Matplotlib. See `requirements.txt` for the core environment and `notebooks/README.md` for the analytical sequence.

## Important interpretation boundaries

- The models predict **instantaneous stack voltage from contemporaneous operational measurements**; they do not forecast future voltage, degradation trajectories or Remaining Useful Life (RUL).
- Prediction residuals are not direct degradation measurements.
- Feature importance and SHAP attribution do not establish physical causality.
- Comparable-current polarization quantifies performance deterioration but does not independently identify a specific catalyst, membrane, ohmic or mass-transport degradation mechanism.
- Results derive from one controlled durability experiment; external-cell generalisability remains future work.

## Future research

Future research should validate the framework across independent PEMFC cells, datasets and operating protocols. Repeated post-rest characterization could support development of a persistent-loss-based longitudinal health indicator, enabling LSTM, GRU and Transformer models to forecast degradation trajectories, End of Life (EOL) and Remaining Useful Life (RUL). Incorporating electrochemical diagnostics could further distinguish recoverable performance losses from persistent physical degradation mechanisms.

## Author

Hafiz Muhammad Usman Aurangzaib  
MSc Data Science and Artificial Intelligence  
Sheffield Hallam University

## Academic context

This repository accompanies an MSc dissertation. Please cite the original dataset and relevant publications when reusing the analytical approach or data source.