# MP2 – Determinants of Employee Compensation at HCL Technologies

Course DOM207 / DOM3007 / DOM6401. Python re-implementation of the full analysis behind `report/MP2_Report.docx`.

## Layout
```
data/              ITEmp.xlsx (raw, untouched) -> ITEmp_Cleaned.csv, ITEmp_CompleteCases.csv
code/              common.py, modelspec.py (shared) + 01..06 analysis scripts + run_all.py
output_tables/     every table in the report as CSV (table1 ... table12, descriptive_stats, group_means)
figures/           Figures 1-9 (PNG)
report/            MP2_Report.docx (+ PDF copy)
```

## Run
```
pip install -r requirements.txt
cd code && python run_all.py
```

| Script | Produces |
|---|---|
| 01_clean.py | Cleaning audit (Table 1), cleaned + complete-case data |
| 02_models.py | M1-M11, Tables 3, 4, 5, 10, 11, 12 |
| 03_diagnostics.py | Tables 6 (JB, BP, White/Wooldridge, RESET, Cook) and 7 (VIF) |
| 04_robustness.py | Tables 8, 9, influence, turning points, education-quadratic robustness |
| 05_figures.py | Figures 1-9 |
| 06_descriptives.py | descriptive_stats.csv, group_means.csv |

## Key modelling choices
* Dependent variable: log average hourly salary; OLS with mean-centred regressors (`c_` prefix) before squares/interactions.
* Invalid cells (text, out-of-range, off-dictionary) -> missing; median/mode imputation; no rows dropped. Ranges are in `code/common.py`.
* M1-M7 use classical SEs (Table 3); M8 / Tables 11-12 use HC3.
* Table 9 (cross-validation) depends on the random fold split (seed 42 in `04_robustness.py`).
