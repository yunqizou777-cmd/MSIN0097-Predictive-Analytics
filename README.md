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

## Presentation Format

Use `telco_churn_workflow.ipynb`

The notebook includes these exact six headings:

1. Obtain dataset and frame problem
2. Explore data
3. Prepare data
4. Explore models and shortlist
5. Fine-tune and evaluate
6. Present final solution

It writes outputs into `outputs_notebook/`.

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

## Notes

- Target variable is `Churn` (`Yes`/`No`), converted to binary.
- `TotalCharges` is converted to numeric with coercion; missing values are imputed in the pipeline.
- `customerID` is excluded from modeling features.
- Engineered features are added to improve signal (`TenureBand`, service count, charge-derived features).
- Current model set includes baseline + advanced models (`LogisticRegression`, `MLPClassifier`, `GradientBoostingClassifier`, `RandomForest`).
- Metric note: this is classification, so use ROC-AUC/PR-AUC/F1 (not R-squared).

### Required

- Project code:
  - `telco_churn_pipeline.py`
  - `telco_churn_workflow.ipynb`
- Environment specification:
  - `requirements.txt`
- Run and reproduction instructions:
  - `README.md`
- Data access instructions:
  - this README section: **Data Access Instructions**

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
