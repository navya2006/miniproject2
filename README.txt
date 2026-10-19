MP2 - Determinants of Employee Salary at HCL Technologies
Authors: Navya Arora (2410110449), Amitoj (2410110046)

WHAT YOU GET (everything is written to outputs/)
  figures/            18 PNG charts; start with 00_summary_dashboard.png (the one-page overview)
  table_images/       14 PNG tables (regression results, hypotheses, diagnostics, robustness, ...)
  MP2_results.xlsx    formatted Excel workbook: every table, numeric coefficients, cleaning log, cleaned data
  tables/             the same coefficient and search tables as CSV
  results.json        every number produced by the analysis
  cleaning_log.csv, ITEmp_cleaned.csv, analysis_sample.csv

HOW TO RUN
  pip install -r requirements.txt
  python main.py data/ITEmp.xlsx -o outputs            # full run: analysis + figures + table images + Excel (about 40 s)
  python main.py data/ITEmp.xlsx -o outputs --no-visuals   # numbers only
  python tests/test_smoke.py outputs                    # sanity checks on counts, fit and outputs
  Random seed 20261019 is fixed; ITEmp.xlsx is never modified.

CODE LAYOUT (one module per job, all in mp2/)
  main.py                  runs the stages in order
  config.py                variable groups, cleaning ranges, model formulas, labels
  context.py               Context object that carries data and results between stages
  data_cleaning.py         load ITEmp.xlsx, four logged cleaning rules, build the samples
  descriptives.py          summary statistics and group means
  models.py                nested OLS models M1-M6, nested tests, mediation, family background
  battery.py               quadratic / interaction search and alternative functional forms
  effects.py               percent effects, marginal effects, turning points, hypothesis verdicts
  diagnostics.py           normality, heteroskedasticity, RESET, multicollinearity, influence
  robustness.py            alternative estimators and samples, bootstrap, R-squared decomposition
  cleaning_sensitivity.py  M6 refitted under alternative cleaning thresholds
  storage.py               results.json and analysis_sample.csv
  style.py                 plotting theme and validated colour palette
  plots_data.py            data and cleaning figures
  plots_models.py          model-result figures
  plots_diagnostics.py     diagnostic figures
  plots_robustness.py      robustness, sensitivity and search figures
  dashboard.py             one-page summary dashboard
  table_builders.py        DataFrames for every results table
  table_render.py          tables as PNG images
  export_excel.py          formatted Excel workbook

Reading the charts: blue = significant / primary series, grey = not significant, orange = contrast.
In heat maps blue means significantly positive, red significantly negative, grey no evidence;
the numbers in the cells always carry the coefficient and stars, so colour is never the only signal.
