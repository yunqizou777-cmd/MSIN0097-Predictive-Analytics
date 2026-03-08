# Agentic Workflow Evidence (Plan -> Delegate -> Verify -> Revise)

This file provides explicit evidence for requirement (3): showing how workflow was organized and progressed using an agentic methodology.

## Method Used
- Planning: define six-step ML workflow and measurable success criteria.
- Delegation: use agent support for code scaffolding, evaluation harness, and documentation templates.
- Verification: run full pipeline, check leakage/split integrity, validate metrics and artefacts.
- Revision: fix detected issues, add ablations, refine feature engineering, and retune top models with CV.

## Stage-by-Stage Evidence
1. Plan
- `outputs/tables/problem_framing.json`
- `outputs/reports/step1_problem_framing.md`
- `outputs/reports/report_skeleton.md`

2. Delegate (implementation support)
- `telco_churn_pipeline.py`
- `telco_churn_workflow.ipynb`
- `telco_churn_workflow.Rmd`

3. Verify
- Data quality and schema checks:
  - `outputs/tables/data_quality_summary.json`
  - `outputs/tables/schema_validation.json`
  - `outputs/tables/missing_summary.csv`
- Leakage and split discipline:
  - `outputs/tables/split_summary.json`
  - `outputs/tables/split_validation.json`
  - `outputs/tables/class_balance_split.json`
  - `outputs/tables/feature_space_validation.json`
- Reproducible metrics:
  - `outputs/tables/model_metrics_validation.csv`
  - `outputs/tables/model_metrics_validation_tuned_cv.csv`
  - `outputs/tables/best_model_test_metrics_tuned_threshold.csv`

4. Revise
- Ablation and tuning:
  - `outputs/tables/ablation_class_weight_logreg_validation.csv`
  - `outputs/tables/model_shortlist_evidence.csv`
  - `outputs/tables/model_metrics_test_optimal_threshold_from_val.csv`
  - `outputs/tables/cv_search_results_shortlisted_models.csv`
  - `outputs/tables/best_hyperparameters.csv`
  - `outputs/tables/workflow_step5.json`
- Failure mode and subgroup review:
  - `outputs/tables/error_analysis_by_contract.csv`
  - `outputs/tables/subgroup_metrics.csv`
- Final communication artefacts:
  - `outputs/reports/model_card.md`
  - `outputs/reports/final_solution_brief.md`
  - `outputs/reports/pipeline_summary.md`
  - `outputs/reports/final_report_draft.md`
  - `outputs/tables/workflow_step6.json`

## Explicit Agent-Mistake Case
- `outputs/reports/agent_mistake_example.md`
- Appendix template to complete with your own logs:
  - `outputs/reports/agent_usage_log_template.md`
  - `outputs/reports/workflow_evidence_tracker.md`
  - `outputs/tables/agent_decision_register_template.csv`
