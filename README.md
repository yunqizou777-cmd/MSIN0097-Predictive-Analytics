# Telco Churn Coursework Starter

This project gives you a reproducible starter workflow for the MSIN0097 individual coursework using:

- dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- pipeline script: `telco_churn_pipeline.py`
- primary presentation notebook: `telco_churn_workflow.ipynb`

## Quick Start

1. Create/activate the virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python telco_churn_pipeline.py \
  --data-path WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --output-dir outputs \
  --random-state 42 \
  --top-models-to-tune 3
```

4. Run the notebook:

```bash
jupyter lab telco_churn_workflow.ipynb
```

## Data Access Instructions

- Expected data file in repo root: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- If missing, download IBM Telco Customer Churn CSV and place it in the project root with the same filename.
- You can also pass a custom location:

```bash
python telco_churn_pipeline.py --data-path /absolute/path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Presentation Format (Notebook Style with Six Required Headings)

Use `telco_churn_workflow.ipynb` if you want a markdown-rich, report-style notebook format similar to your preferred class notebook style.

The notebook includes these exact six headings:

1. Obtain dataset and frame problem
2. Explore data
3. Prepare data
4. Explore models and shortlist
5. Fine-tune and evaluate
6. Present final solution

It writes outputs into `outputs_notebook/`.

## Alternative Presentation Format (R Markdown)

Use `telco_churn_workflow.Rmd` when you want code shown in clearly separated sections that mirror the coursework structure:

1. Obtain dataset and frame problem
2. Explore data
3. Prepare data
4. Explore models and shortlist
5. Fine-tune and evaluate
6. Present final solution

Render command (if R + rmarkdown are installed):

```bash
Rscript -e "rmarkdown::render('telco_churn_workflow.Rmd')"
```

This R Markdown workflow writes outputs into `outputs_rmd/` so it does not overwrite your main `outputs/` run.

## What The Script Produces

Inside `outputs/`:

- `tables/`
  - data quality checks
  - business KPI summary
  - feature engineering summary
  - schema + split validation checks
  - class-balance preservation check
  - transformed feature-space validation check
  - split summary
  - baseline validation/test model comparison metrics
  - test metrics at validation-optimal threshold
  - ablation study output
  - shortlisted-model CV search results
  - best tuned hyperparameters table
  - Step 5 workflow evidence JSON
  - Step 6 workflow evidence JSON
  - shortlist evidence table (selection + threshold context)
  - tuned-model validation/test metrics
  - tuned threshold tables
  - threshold policy comparison table
  - decile/lift analysis table
  - best-model predictions
  - error analysis by contract
  - subgroup metrics (fairness/error slices)
  - permutation importance
  - EDA leakage-risk correlation checks
  - EDA outlier summary (IQR method)
- `figures/`
  - EDA figures
  - segment-grid EDA figure
  - missingness bar chart
  - numeric correlation heatmap
  - ROC/PR curves
  - best-model ROC/PR curve alias
  - confusion matrix
  - best-model confusion matrix alias
  - calibration curve
  - best-model calibration curve alias
  - decile lift chart
  - feature importance plot
- `models/`
  - saved best model (`best_model.joblib`)
- `reports/`
  - `pipeline_summary.md`
  - `model_card.md`
  - `final_solution_brief.md`
  - `report_skeleton.md`
  - `step1_problem_framing.md`
  - `agent_usage_log_template.md`
  - `workflow_evidence_tracker.md`
  - `metric_guidance.md`

## Coursework Alignment (6 Steps)

The script is intentionally organized around:

1. Frame predictive problem.
2. Explore data.
3. Prepare data with leakage-safe split and preprocessing pipeline.
4. Compare multiple models.
5. Fine-tune and evaluate robustly (CV tuning, thresholds, subgroup/error checks).
6. Save final artefacts + reporting outputs.

## Suggested Next Refinements

1. Expand CV grids and compare stability across different random seeds.
2. Add subgroup fairness checks (`gender`, `SeniorCitizen`, `Contract`).
3. Add a short notebook that imports outputs and turns them into report-ready charts/tables.
4. Fill the generated appendix templates with your real interaction evidence + decisions.

## Notes

- Target variable is `Churn` (`Yes`/`No`), converted to binary.
- `TotalCharges` is converted to numeric with coercion; missing values are imputed in the pipeline.
- `customerID` is excluded from modeling features.
- Engineered features are added to improve signal (`TenureBand`, service count, charge-derived features).
- Current model set includes baseline + advanced models (`LogisticRegression`, `MLPClassifier`, `GradientBoostingClassifier`, `RandomForest`).
- Metric note: this is classification, so use ROC-AUC/PR-AUC/F1 (not R-squared).

## Final Submission Package (What To Upload)

Upload these files/folders to GitHub (or OneDrive equivalent) so your repository meets the brief:

### Required

- Project code:
  - `telco_churn_pipeline.py`
  - `telco_churn_workflow.ipynb`
  - `telco_churn_workflow.Rmd`
- Environment specification:
  - `requirements.txt`
- Run and reproduction instructions:
  - `README.md`
- Data access instructions (already covered here):
  - this README section: **Data Access Instructions**

### Strongly Recommended (for evidence and auditability)

- Workflow and agentic evidence:
  - `AGENTIC_WORKFLOW_EVIDENCE.md`
  - `outputs/reports/appendix_agent_usage_log_complete.md`
  - `outputs/reports/agent_decision_register_submission.md`
  - `outputs/tables/agent_decision_register_submission.csv`
  - `outputs/reports/appendix_screenshot_plan.md`
- Final report drafting support:
  - `outputs/reports/report_2000word_near_final.md`
  - `outputs/reports/final_compliance_check.md`

### Data file policy

- If redistribution is allowed, include:
  - `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- If redistribution is not allowed, do **not** upload the CSV. Keep only source link + placement instructions in README.

### Do not upload

- local environment/cache files: `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`
- unnecessary large binaries not needed for marking

## GitHub Access Note (Important)

- If the repository is **private**, markers cannot open it unless you explicitly grant access.
- Safe options:
  1. make repo **public** before submission, or
  2. keep it private and invite module staff GitHub accounts as collaborators with at least read access.
- A private link alone in your PDF is usually **not sufficient**.

## Troubleshooting: ModuleNotFoundError

If you see errors such as `No module named 'matplotlib'`, your active Python environment is not the one with project dependencies.

Use the same interpreter for install and run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip show matplotlib pandas scikit-learn
python telco_churn_pipeline.py --data-path WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If you use Jupyter, register and select the same kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name msin0097-pa --display-name "MSIN0097-PA"
```
