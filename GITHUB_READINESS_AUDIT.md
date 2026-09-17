# GitHub Readiness Audit

Scope: Notebooks 01–13, Notebook 12A Random Forest, and current src modules.

Publication preparation completed before repository upload:

- Notebook 14 intentionally excluded pending terminology and interpretation revision.
- Embedded binary/rich image payloads removed from notebook outputs to reduce repository size.
- Code cells, Markdown cells, and textual/tabular outputs preserved.
- Identified machine-specific Windows path to `src/eda.py` replaced with repository-relative `src/eda.py`.
- Duplicate/version suffixes removed from public notebook filenames.
- Every prepared notebook validated as readable JSON.
- Raw PEMFC dataset files excluded from the repository.
- Source modules retained without analytical changes.

## Intended public notebook set

1. `01_Project_Setup_and_Data_Loading.ipynb`
2. `02_Data_Merging.ipynb`
3. `03_Initial_Data_Understanding.ipynb`
4. `04_Data_Cleaning.ipynb`
5. `05_Data_Quality_Assessment.ipynb`
6. `06_Missing_Values_Duplicates_Outliers_and_Time_Series_Integrity.ipynb`
7. `07_Statistical_Exploration.ipynb`
8. `08_Behaviour_Analysis.ipynb`
9. `09_Target-Oriented_Association_Analysis.ipynb`
10. `10_Feature_Engineering.ipynb`
11. `11_Feature_Selection.ipynb`
12. `12_Model_Development_and_Comparative_Evaluation.ipynb`
13. `12A_Random_Forest_Regression.ipynb`
14. `13_SHAP_Explainability.ipynb`

Notebook 14 will be added separately after its model-residual/degradation terminology is aligned with the final dissertation interpretation boundaries.

These publication copies do not replace the original local development notebooks.
