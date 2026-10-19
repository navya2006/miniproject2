"""MP2 - salary determinants at HCL Technologies (DOM207 / DOM3007 / DOM6401).

One module per job:

  config               constants: variable groups, model formulas, labels, seed
  context              the Context object that carries data/results between stages
  data_cleaning        load ITEmp.xlsx, apply the logged cleaning rules, build the samples
  descriptives         summary statistics and group means
  models               nested OLS models M1-M6, nested tests, mediation, family background
  battery              quadratic / interaction search and alternative functional forms
  effects              marginal effects, turning points, hypothesis summary
  diagnostics          normality, heteroskedasticity, RESET, multicollinearity, influence
  robustness           alternative estimators / samples, bootstrap, R2 decomposition
  cleaning_sensitivity re-fit M6 under alternative cleaning thresholds
  storage              results.json, analysis_sample.csv

  style                plotting theme and palette
  plots_data           figures about the data and the cleaning
  plots_models         figures about the model results
  plots_diagnostics    diagnostic figures
  plots_robustness     robustness / sensitivity / battery figures
  dashboard            one-page summary dashboard
  table_render         generic DataFrame -> PNG table renderer
  table_builders       DataFrames for every results table
  export_excel         formatted Excel workbook with all tables
"""
