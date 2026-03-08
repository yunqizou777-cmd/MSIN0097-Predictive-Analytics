# Appendix: Agent Usage Log (Screenshot-Ready)

Use this file directly for screenshot/export evidence. Each log below is short, meaningful, and tied to saved artefacts.

## Log L01 (Step 1: Problem Framing)
**User ask to agent:** Define target, prediction type, metrics, constraints, assumptions, limitations, and delegation plan.

**Agent output (implemented):**
```python
problem_framing = write_problem_framing_artifacts(df, paths)
```

**Human verification:** Confirmed classification metrics (ROC-AUC/PR-AUC/F1) are appropriate; rejected regression-only metrics.

**Evidence:** `outputs_notebook/tables/problem_framing.json`, `outputs_notebook/reports/step1_problem_framing.md`

---
## Log L02 (Step 2: EDA Scope Expansion)
**User ask to agent:** Add concise visual EDA for missingness, leakage risk, class imbalance, and outliers.

**Agent output (implemented):**
```python
plot_missingness(df, paths["figures"] / "eda_missingness.png")
plot_numeric_correlation(df, paths["figures"] / "eda_numeric_correlation.png")
save_eda_risk_tables(df, paths["tables"])
```

**Human verification:** Checked missingness table consistency and reviewed high-correlation pairs as proxy-risk (not auto-leakage).

**Evidence:** `outputs_notebook/figures/eda_missingness.png`, `outputs_notebook/tables/eda_leakage_risk_checks.csv`, `outputs_notebook/tables/eda_outlier_summary_iqr.csv`

---
## Log L03 (Step 3: Validation Checks)
**User ask to agent:** Add explicit schema and split-validation checks.

**Agent output (implemented):**
```python
assert list(x_val.columns) == expected_features
assert list(x_test.columns) == expected_features
(paths["tables"] / "class_balance_split.json").write_text(...)
```

**Human verification:** Confirmed target/ID are excluded and preprocessing fit is train-only.

**Evidence:** `outputs_notebook/tables/split_validation.json`, `outputs_notebook/tables/class_balance_split.json`, `outputs_notebook/tables/feature_space_validation.json`

---
## Log L04 (Step 4: Baseline + Modern Model Comparison)
**User ask to agent:** Compare baseline and modern models, then shortlist using evidence.

**Agent output (implemented):**
```python
for model_name, model in model_specs(RANDOM_STATE).items():
    ...
val_df.to_csv(paths["tables"] / "model_metrics_validation.csv", index=False)
```

**Human verification:** Confirmed identical preprocessing and identical split for fair comparison.

**Evidence:** `outputs_notebook/tables/model_metrics_validation.csv`, `outputs_notebook/tables/model_shortlist_evidence.csv`

---
## Log L05 (Step 4: Threshold-Aware Comparison)
**User ask to agent:** Go beyond fixed threshold 0.5 in model comparison.

**Agent output (implemented):**
```python
test_m_opt = evaluate(y_test, y_test_prob, threshold=best_thr)
```

**Human verification:** Checked threshold source is validation only.

**Evidence:** `outputs_notebook/tables/model_metrics_test_optimal_threshold_from_val.csv`, `outputs_notebook/tables/thresholds_from_validation.csv`

---
## Log L06 (Step 4: Ablation)
**User ask to agent:** Justify class-weight setting experimentally.

**Agent output (implemented):**
```python
run_ablation_study(...)
```

**Human verification:** Compared balanced vs unbalanced logistic regression and kept setting with better trade-off.

**Evidence:** `outputs_notebook/tables/ablation_class_weight_logreg_validation.csv`

---
## Log L07 (Step 5: CV Tuning)
**User ask to agent:** Tune shortlisted models with robust strategy.

**Agent output (implemented):**
```python
tuned_pipelines, cv_results_df = tune_top_models_with_cv(...)
```

**Human verification:** Confirmed test set not used during CV search.

**Evidence:** `outputs_notebook/tables/cv_search_results_shortlisted_models.csv`, `outputs_notebook/tables/best_hyperparameters.csv`

---
## Log L08 (Step 5: Robust Evaluation)
**User ask to agent:** Add confusion matrix, ROC/PR, calibration, and failure-mode analysis.

**Agent output (implemented):**
```python
plot_conf_matrix(...)
plot_roc_pr_curves(...)
plot_calibration(...)
run_error_analysis(...)
run_subgroup_analysis(...)
```

**Human verification:** Cross-checked figures against test metrics and threshold policy table.

**Evidence:** `outputs_notebook/figures/best_model_confusion_matrix.png`, `outputs_notebook/figures/test_roc_pr_curves.png`, `outputs_notebook/tables/error_analysis_by_contract.csv`, `outputs_notebook/tables/subgroup_metrics.csv`

---
## Log L09 (Explicit Agent Mistake and Correction)
**Agent mistake:** Suggested default threshold `0.50` for final decisioning.

**Correction by human:** Replaced with validation-derived threshold (max F1), then re-evaluated test metrics.

**Evidence:** `outputs_notebook/tables/model_metrics_test_threshold_0_5.csv`, `outputs_notebook/tables/model_metrics_test_optimal_threshold_from_val.csv`, `outputs_notebook/reports/agent_mistake_example.md`

---
## Log L10 (Step 6: Final Communication Artifacts)
**User ask to agent:** Prepare final report-ready outputs and model card.

**Agent output (implemented):**
```python
write_model_card(best_model, best_thr, best_test_metrics, paths)
write_final_solution_brief(best_model, best_thr, best_test_metrics, paths)
```

**Human verification:** Ensured intended use, caveats, limitations, and next steps are explicit and evidence-linked.

**Evidence:** `outputs_notebook/reports/model_card.md`, `outputs_notebook/reports/final_solution_brief.md`

---
## Log L11 (Notebook Error Handling)
**User issue:** `NameError` / `AttributeError` for helper functions in notebook state.

**Agent output (implemented):** Added robust fallback calls and safe checks in Step 6 cell.

**Human verification:** Re-ran cell after kernel refresh; confirmed artefacts are written.

**Evidence:** `telco_churn_workflow.ipynb` Step 6 cell, `outputs_notebook/tables/workflow_step6.json`

---
## Log L12 (Workflow and Decision Evidence)
**User ask to agent:** Ensure appendix evidence is complete for submission.

**Agent output (implemented):** `workflow_step1..6.json` and unified decision register.

**Human verification:** Confirmed each stage has actions, checks, and output files.

**Evidence:** `outputs_notebook/tables/workflow_step1.json` ... `workflow_step6.json`, `outputs_notebook/tables/agent_decision_register.csv`
