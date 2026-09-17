# Notebook workflow

This directory contains the reproducible analytical workflow for the dissertation.

The final workflow progresses through:

1. Environment and dataset setup
2. Data loading and initial understanding
3. Data-quality assessment and cleaning
4. Time-series integrity and exploratory data analysis
5. Subsystem and comparative behaviour analysis
6. Target-oriented association analysis
7. PEMFC-informed feature engineering
8. Feature relevance, redundancy and multicollinearity assessment
9. Fold-wise XGBoost-Boruta feature selection
10. Ridge, Random Forest and XGBoost model development
11. Expanding chronological validation and later-stage evaluation
12. SHAP explainability and robustness analysis
13. Comparable-current polarization, recovery and integrated degradation assessment

## Methodological rule

The machine-learning task is contemporaneous regression: operational measurements at time `t` are used to predict stack voltage at time `t`. The notebooks must therefore not describe this task as future-voltage forecasting or RUL prediction.

The 900, 950 and 1000 h stages constitute the later-stage holdout and must remain excluded from development/tuning decisions.
