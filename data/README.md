# Data

The raw PEMFC durability dataset is not redistributed in this repository.

## Source

Zuo, J., Lv, H., Zhou, D., Xue, Q., Jin, L., Zhou, W., Yang, D., & Zhang, C. (2021). Long-term dynamic durability test datasets for single proton exchange membrane fuel cell. *Data in Brief, 35*, 106775. https://doi.org/10.1016/j.dib.2021.106775

## Expected raw files

Place the published CSV files in `data/raw/` before running the notebooks. The operational files are durability stages from `50_h.csv` through `1000_h.csv` in 50 h increments. The project also uses the two polarization datasets supplied with the published data.

```text
data/
├── raw/          # original published CSV files (not tracked by Git)
└── processed/    # generated cleaned/engineered datasets (not tracked by Git)
```

Do not modify the original raw files. Reproducible preprocessing should write derived data to `data/processed/`.
