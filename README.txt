MP2 Salary Analysis project
============================

Files
-----
- MP2_Salary_Analysis_Report.docx: formal report draft. Replace bracketed title-page fields.
- analysis.py: reproducible cleaning, regression, visualization and diagnostic pipeline.
- ITEmp.xlsx: original input workbook supplied for this project.
- cleaned_data.xlsx: separate cleaned data copy; original workbook is preserved.
- model_comparison.csv: model fit comparison.
- regression_coefficients.csv: coefficients, HC3 robust standard errors, p-values and confidence intervals.
- vif_model3.csv: variance inflation factors.
- data_cleaning_log.csv: missingness and range-check log.
- diagnostics.txt: heteroskedasticity, RESET and residual-normality diagnostics.
- PNG files: report figures.

Run
---
Install dependencies:
    pip install pandas numpy matplotlib statsmodels scipy openpyxl python-docx

Keep ITEmp.xlsx beside analysis.py, then run:
    python analysis.py

Important
---------
The report is a complete draft but the title page contains placeholders for student/collaborator names, roll numbers, course/instructor details. Review the cleaning rule that salary values above 100 are flagged as likely entry anomalies; confirm against the original source if possible. The analysis is associational, not causal. The data codebook has some inconsistent names and incomplete category definitions.
