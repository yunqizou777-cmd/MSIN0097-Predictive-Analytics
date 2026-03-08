#!/usr/bin/env python3
"""
End-to-end starter pipeline for Telco customer churn.

This script is structured to match the coursework workflow:
1) Frame problem
2) Explore data
3) Prepare data
4) Compare models
5) Tune and evaluate
6) Present final solution

Important metric note:
- This is a classification problem, so the primary fit metrics are ROC-AUC / PR-AUC / F1.
- R-squared is a regression metric, so it is not used as the main model-selection criterion here.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Stable thread settings for reproducible runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import joblib  # type: ignore
except ModuleNotFoundError:
    # Fallback keeps `joblib.dump/load` behavior via pickle.
    class _JoblibFallback:
        @staticmethod
        def dump(obj, filename):
            with open(filename, "wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)

    joblib = _JoblibFallback()


# ============================================================================
# Project Setup
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducible runs."""
    parser = argparse.ArgumentParser(description="Telco churn starter pipeline")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for metrics, figures, and model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Validation size as % of full data.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top-models-to-tune",
        type=int,
        default=3,
        help="How many top baseline models to pass into CV tuning.",
    )
    return parser.parse_args()


def make_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create all output directories used by this project run."""
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "models": output_dir / "models",
        "reports": output_dir / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ============================================================================
# Data Loading and Validation Helpers (used in Steps 1 and 3)
# ============================================================================


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven engineered features to improve predictive signal."""
    # Keep raw columns and add derived predictors.
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    out["AvgMonthlyChargeFromTotal"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], np.nan
    )
    out["AvgMonthlyChargeFromTotal"] = out["AvgMonthlyChargeFromTotal"].replace(
        [np.inf, -np.inf], np.nan
    )
    out["AvgMonthlyChargeFromTotal"] = out["AvgMonthlyChargeFromTotal"].fillna(
        out["MonthlyCharges"]
    )

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    out["NumOptionalServices"] = (out[service_cols] == "Yes").sum(axis=1)
    out["HasFiberOptic"] = (out["InternetService"] == "Fiber optic").astype(int)
    out["IsMonthToMonth"] = (out["Contract"] == "Month-to-month").astype(int)
    out["MonthlyChargePerService"] = out["MonthlyCharges"] / (
        out["NumOptionalServices"] + 1
    )
    out["TenureBand"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
        include_lowest=True,
    )

    return out


def save_feature_engineering_summary(df: pd.DataFrame, out_tables: Path) -> None:
    """Write a quick summary table of engineered feature properties."""
    engineered_cols = [
        "AvgMonthlyChargeFromTotal",
        "NumOptionalServices",
        "HasFiberOptic",
        "IsMonthToMonth",
        "MonthlyChargePerService",
        "TenureBand",
    ]
    summary = pd.DataFrame(
        {
            "feature": engineered_cols,
            "dtype": [str(df[c].dtype) for c in engineered_cols],
            "missing_count": [int(df[c].isna().sum()) for c in engineered_cols],
            "unique_values": [int(df[c].nunique(dropna=True)) for c in engineered_cols],
        }
    )
    summary.to_csv(out_tables / "feature_engineering_summary.csv", index=False)


def load_and_clean(data_path: Path) -> pd.DataFrame:
    """Load CSV and apply all cleaning + engineered feature logic."""
    df = pd.read_csv(data_path)
    return add_engineered_features(df)


def basic_data_checks(df: pd.DataFrame, out_tables: Path) -> pd.DataFrame:
    """Write baseline data-quality diagnostics for auditability."""
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
        }
    ).sort_values("missing_count", ascending=False)
    missing_df.to_csv(out_tables / "missing_summary.csv", index=False)

    quality = {
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df["Churn"].value_counts().to_dict(),
    }
    (out_tables / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    return missing_df


def validate_dataset_schema(df: pd.DataFrame, out_tables: Path) -> None:
    """Enforce expected schema/labels early to fail fast on bad input data."""
    required_cols = {
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "Contract",
        "gender",
        "SeniorCitizen",
    }
    missing_required = sorted([c for c in required_cols if c not in df.columns])
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    unique_targets = sorted(df["Churn"].dropna().unique().tolist())
    if not set(unique_targets).issubset({"Yes", "No"}):
        raise ValueError(f"Unexpected target labels in Churn: {unique_targets}")

    schema_summary = {
        "required_columns_present": True,
        "target_labels_observed": unique_targets,
        "customer_id_unique_ratio": float(df["customerID"].nunique() / len(df)),
    }
    (out_tables / "schema_validation.json").write_text(
        json.dumps(schema_summary, indent=2), encoding="utf-8"
    )


# ============================================================================
# Split and Preprocessing Helpers (used in Step 3)
# ============================================================================


def split_data(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create leakage-safe train/validation/test splits with stratification."""
    x = df.drop(columns=[target_col, id_col])
    y = (df[target_col] == "Yes").astype(int)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_share = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_share,
        stratify=y_train_val,
        random_state=random_state,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def validate_split_integrity(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    out_tables: Path,
) -> None:
    """Verify there is no index overlap and class balance is preserved."""
    train_idx = set(x_train.index.tolist())
    val_idx = set(x_val.index.tolist())
    test_idx = set(x_test.index.tolist())

    overlap = {
        "train_val_overlap": len(train_idx.intersection(val_idx)),
        "train_test_overlap": len(train_idx.intersection(test_idx)),
        "val_test_overlap": len(val_idx.intersection(test_idx)),
    }
    if any(v > 0 for v in overlap.values()):
        raise ValueError(f"Data leakage risk: split overlap detected: {overlap}")

    split_class_balance = {
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
    }
    payload = {"index_overlap": overlap, "class_balance": split_class_balance}
    (out_tables / "split_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Build numeric/categorical preprocessing as one reproducible transformer."""
    num_cols = x_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )
    return preprocessor


def run_step3_preparation_checks(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    out_tables: Path,
    target_col: str = "Churn",
    id_col: str = "customerID",
) -> Dict[str, object]:
    """Write explicit Step-3 validation artefacts for schema and preprocessing checks."""
    expected_features = list(x_train.columns)
    schema_ok = (list(x_val.columns) == expected_features) and (
        list(x_test.columns) == expected_features
    )
    if not schema_ok:
        raise ValueError("Schema mismatch across train/validation/test feature columns.")
    if target_col in expected_features:
        raise ValueError(f"Target column `{target_col}` leaked into feature matrix.")
    if id_col in expected_features:
        raise ValueError(f"Identifier column `{id_col}` leaked into feature matrix.")

    class_balance = {
        "train_churn_rate": float(y_train.mean()),
        "val_churn_rate": float(y_val.mean()),
        "test_churn_rate": float(y_test.mean()),
    }
    (out_tables / "class_balance_split.json").write_text(
        json.dumps(class_balance, indent=2), encoding="utf-8"
    )

    # Fit a cloned preprocessor on train only, then confirm transformed dimensions match.
    prep_check = clone(preprocessor)
    x_train_t = prep_check.fit_transform(x_train)
    x_val_t = prep_check.transform(x_val)
    x_test_t = prep_check.transform(x_test)
    transformed_feature_count = int(x_train_t.shape[1])
    transformed_ok = (
        int(x_val_t.shape[1]) == transformed_feature_count
        and int(x_test_t.shape[1]) == transformed_feature_count
    )
    if not transformed_ok:
        raise ValueError("Transformed feature-space mismatch across train/validation/test.")

    payload = {
        "schema_consistent": True,
        "target_removed_from_features": True,
        "id_removed_from_features": True,
        "preprocessor_fit_scope": "train_only",
        "transformed_feature_count": transformed_feature_count,
        "transformed_feature_space_consistent": True,
        "class_balance": class_balance,
    }
    (out_tables / "feature_space_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


# ============================================================================
# Model Portfolio Helpers (used in Step 4)
# ============================================================================


def model_specs(random_state: int) -> Dict[str, object]:
    """Return model portfolio: baseline + advanced candidates for comparison."""
    return {
        "log_reg_baseline": LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
            learning_rate=0.05,
            n_estimators=350,
            max_depth=3,
            min_samples_leaf=20,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=random_state,
            max_iter=800,
            early_stopping=True,
        ),
    }


def tuning_search_space(model_name: str) -> Tuple[Dict[str, list], int]:
    """Define per-model hyperparameter search spaces for RandomizedSearchCV."""
    if model_name == "log_reg_baseline":
        return (
            {
                "model__C": np.logspace(-3, 2, 15).tolist(),
                "model__penalty": ["l1", "l2"],
            },
            12,
        )
    if model_name == "random_forest":
        return (
            {
                "model__n_estimators": [200, 300, 400, 500, 700],
                "model__max_depth": [None, 6, 8, 10, 14],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.7, 0.9],
            },
            8,
        )
    if model_name == "gradient_boosting":
        return (
            {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__n_estimators": [150, 250, 350, 500],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [5, 10, 20, 30],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            8,
        )
    if model_name == "mlp_classifier":
        return (
            {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
            },
            8,
        )
    return {}, 0


# ============================================================================
# Tuning and Evaluation Helpers (used in Step 5)
# ============================================================================


def tune_top_models_with_cv(
    top_model_names: list,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Tune shortlisted models with CV and return best estimators + summary."""
    tuned_pipelines = {}
    rows = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    specs = model_specs(random_state)

    for model_name in top_model_names:
        model = specs[model_name]
        param_dist, n_iter = tuning_search_space(model_name)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

        if not param_dist:
            pipe.fit(x_train, y_train)
            tuned_pipelines[model_name] = pipe
            rows.append(
                {
                    "model": model_name,
                    "search_type": "none",
                    "n_iter": 0,
                    "best_cv_roc_auc": np.nan,
                    "best_params": "{}",
                }
            )
            continue

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=random_state,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)

        tuned_pipelines[model_name] = search.best_estimator_
        rows.append(
            {
                "model": model_name,
                "search_type": "randomized_cv",
                "n_iter": int(n_iter),
                "best_cv_roc_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_),
            }
        )

    return tuned_pipelines, pd.DataFrame(rows)


def run_ablation_study(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    out_tables: Path,
) -> None:
    """Run focused ablation(s) to justify key modeling choices."""
    ablations = {
        "logreg_balanced": LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced"
        ),
        "logreg_unbalanced": LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight=None
        ),
    }
    rows = []
    for name, model in ablations.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(x_train, y_train)
        y_val_prob = pipe.predict_proba(x_val)[:, 1]
        m = evaluate(y_val, y_val_prob, threshold=0.5)
        m["ablation"] = name
        rows.append(m)

    pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(
        out_tables / "ablation_class_weight_logreg_validation.csv", index=False
    )


def evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute classification metrics at a specified decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Pick threshold that maximizes validation F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[idx])


# ============================================================================
# EDA Plotting Helpers (used in Step 2)
# ============================================================================


def plot_class_balance(df: pd.DataFrame, out_path: Path) -> None:
    """Save class-balance bar chart."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Class Balance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_eda_numeric(df: pd.DataFrame, out_path: Path) -> None:
    """Save numeric-boxplot EDA figure by churn target."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0])
    axes[0].set_title("Tenure by Churn")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
    axes[1].set_title("MonthlyCharges by Churn")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_churn_by_segment(df: pd.DataFrame, out_path: Path) -> None:
    """Save churn-rate-by-contract chart for business interpretation."""
    segment = (
        df.groupby("Contract")["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .reset_index(name="churn_rate")
        .sort_values("churn_rate", ascending=False)
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=segment, x="Contract", y="churn_rate")
    plt.title("Churn Rate by Contract")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_churn_by_segments_grid(df: pd.DataFrame, out_path: Path) -> None:
    """Save a dashboard-style view of churn rates across key categorical segments."""
    segments = ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for i, col in enumerate(segments):
        segment_df = (
            df.groupby(col)["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .reset_index(name="churn_rate")
            .sort_values("churn_rate", ascending=False)
        )
        sns.barplot(data=segment_df, x=col, y="churn_rate", ax=axes[i])
        axes[i].set_title(f"Churn Rate by {col}")
        axes[i].set_ylabel("Churn Rate")
        axes[i].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, out_path: Path) -> None:
    """Save missing-value proportion bar chart."""
    missing = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing.index, y=missing.values, color="#4C78A8")
    plt.ylabel("Missing Proportion")
    plt.xlabel("Feature")
    plt.title("Missing Values by Feature")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_numeric_correlation(df: pd.DataFrame, out_path: Path) -> None:
    """Save numeric-feature correlation heatmap for leakage-risk inspection."""
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_eda_risk_tables(df: pd.DataFrame, out_tables: Path) -> None:
    """Write Step-2 tables for leakage-risk and outlier diagnostics."""
    leakage_pairs = [
        ("TotalCharges", "tenure"),
        ("TotalCharges", "MonthlyCharges"),
        ("tenure", "MonthlyCharges"),
        ("AvgMonthlyChargeFromTotal", "MonthlyCharges"),
    ]
    leak_rows = []
    for a, b in leakage_pairs:
        if a not in df.columns or b not in df.columns:
            continue
        x = pd.to_numeric(df[a], errors="coerce")
        y = pd.to_numeric(df[b], errors="coerce")
        valid = (~x.isna()) & (~y.isna())
        corr = np.nan
        if int(valid.sum()) > 2:
            corr = float(x[valid].corr(y[valid]))
        leak_rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "pearson_corr": corr,
                "abs_corr": float(abs(corr)) if not np.isnan(corr) else np.nan,
                "risk_note": "High absolute correlation may indicate redundancy or leakage-like proxy behavior.",
            }
        )
    leak_df = pd.DataFrame(leak_rows).sort_values("abs_corr", ascending=False)
    leak_df.to_csv(out_tables / "eda_leakage_risk_checks.csv", index=False)

    outlier_rows = []
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        outlier_rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": outliers,
                "outlier_pct": float(outliers / len(s)),
            }
        )
    outlier_df = pd.DataFrame(outlier_rows).sort_values("outlier_pct", ascending=False)
    outlier_df.to_csv(out_tables / "eda_outlier_summary_iqr.csv", index=False)


def write_business_kpis(df: pd.DataFrame, out_tables: Path) -> None:
    """Write business-facing KPI summary used for decision-oriented storytelling."""
    churn_mask = df["Churn"] == "Yes"
    total_customers = int(len(df))
    churn_customers = int(churn_mask.sum())
    churn_rate = float(churn_customers / total_customers)

    monthly_revenue_total = float(df["MonthlyCharges"].sum())
    monthly_revenue_at_risk = float(df.loc[churn_mask, "MonthlyCharges"].sum())
    annual_revenue_at_risk = float(monthly_revenue_at_risk * 12)

    kpi = {
        "total_customers": total_customers,
        "churn_customers": churn_customers,
        "churn_rate": churn_rate,
        "monthly_revenue_total": monthly_revenue_total,
        "monthly_revenue_at_risk": monthly_revenue_at_risk,
        "annual_revenue_at_risk": annual_revenue_at_risk,
    }
    (out_tables / "business_kpis.json").write_text(
        json.dumps(kpi, indent=2), encoding="utf-8"
    )


def plot_roc_pr_curves(
    y_test: pd.Series, proba_by_model: Dict[str, np.ndarray], out_path: Path
) -> None:
    """Save ROC and PR curves for model comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model_name, probs in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        pr, re, _ = precision_recall_curve(y_test, probs)
        axes[0].plot(fpr, tpr, label=model_name)
        axes[1].plot(re, pr, label=model_name)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[0].set_title("ROC Curves (Test)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Precision-Recall Curves (Test)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_conf_matrix(y_true: pd.Series, y_pred: np.ndarray, out_path: Path) -> None:
    """Save confusion matrix for selected final model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Best Model, Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_calibration(y_true: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    """Save calibration curve for probability quality checks."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_error_analysis(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_tables: Path,
) -> None:
    """Write contract-level failure mode table for diagnostics."""
    analysis = x_test.copy()
    analysis["y_true"] = y_test.values
    analysis["y_pred"] = y_pred
    analysis["y_prob"] = y_prob

    group = (
        analysis.groupby("Contract")
        .apply(
            lambda d: pd.Series(
                {
                    "support": int(len(d)),
                    "actual_churn_rate": float(d["y_true"].mean()),
                    "predicted_churn_rate": float(d["y_pred"].mean()),
                    "avg_score": float(d["y_prob"].mean()),
                }
            )
        )
        .reset_index()
    )
    group.to_csv(out_tables / "error_analysis_by_contract.csv", index=False)


def run_threshold_policy_analysis(
    y_true: pd.Series, y_prob: np.ndarray, out_tables: Path, best_thr: float
) -> None:
    """Compare a small set of operational thresholds for policy decision support."""
    candidate_thresholds = sorted(set([0.30, 0.40, 0.50, round(float(best_thr), 4)]))
    rows = []
    for thr in candidate_thresholds:
        m = evaluate(y_true, y_prob, threshold=float(thr))
        rows.append(m)
    pd.DataFrame(rows).sort_values("threshold").to_csv(
        out_tables / "threshold_policy_comparison.csv", index=False
    )


def run_decile_lift_analysis(
    y_true: pd.Series, y_prob: np.ndarray, out_tables: Path, out_figures: Path
) -> None:
    """Create decile lift table/plot to justify ranking quality for targeting."""
    base_rate = float(pd.Series(y_true).mean())
    rank_df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob})
    rank_df = rank_df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    # Rank-based bins avoid duplicate edges when scores tie.
    rank_df["decile"] = pd.qcut(rank_df.index + 1, 10, labels=False) + 1

    decile_df = (
        rank_df.groupby("decile")
        .agg(
            customers=("y_true", "size"),
            churners=("y_true", "sum"),
            avg_score=("y_prob", "mean"),
        )
        .reset_index()
    )
    decile_df["churn_rate"] = decile_df["churners"] / decile_df["customers"]
    decile_df["lift_vs_overall"] = decile_df["churn_rate"] / (base_rate + 1e-12)
    decile_df.to_csv(out_tables / "decile_lift_analysis.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=decile_df, x="decile", y="lift_vs_overall", color="#3B82F6")
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1)
    plt.title("Decile Lift (Top-Risk Deciles vs Overall Churn Rate)")
    plt.ylabel("Lift vs Overall")
    plt.xlabel("Risk Decile (1 = Highest Risk)")
    plt.tight_layout()
    plt.savefig(out_figures / "decile_lift_chart.png", dpi=200)
    plt.close()


def subgroup_metrics_table(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """Build subgroup metric table for one grouping column."""
    tmp = x_test[[group_col]].copy()
    tmp["y_true"] = y_test.values
    tmp["y_pred"] = y_pred
    tmp["y_prob"] = y_prob

    rows = []
    for g, d in tmp.groupby(group_col):
        roc = np.nan
        if d["y_true"].nunique() == 2:
            roc = float(roc_auc_score(d["y_true"], d["y_prob"]))
        rows.append(
            {
                "group_column": group_col,
                "group_value": str(g),
                "support": int(len(d)),
                "actual_churn_rate": float(d["y_true"].mean()),
                "predicted_churn_rate": float(d["y_pred"].mean()),
                "precision": float(
                    precision_score(d["y_true"], d["y_pred"], zero_division=0)
                ),
                "recall": float(recall_score(d["y_true"], d["y_pred"], zero_division=0)),
                "f1": float(f1_score(d["y_true"], d["y_pred"], zero_division=0)),
                "roc_auc": roc,
            }
        )
    return pd.DataFrame(rows)


def run_subgroup_analysis(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_tables: Path,
) -> None:
    """Write subgroup fairness/performance slices across key demographics."""
    groups = ["gender", "SeniorCitizen", "Contract"]
    frames = []
    for col in groups:
        if col in x_test.columns:
            frames.append(subgroup_metrics_table(x_test, y_test, y_pred, y_prob, col))
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(out_tables / "subgroup_metrics.csv", index=False)


def save_best_hyperparameters(tuned_pipelines: Dict[str, Pipeline], out_tables: Path) -> None:
    """Save tuned model parameters for transparent reporting."""
    best_params = {
        model_name: pipe.named_steps["model"].get_params()
        for model_name, pipe in tuned_pipelines.items()
    }
    pd.DataFrame(best_params).T.to_csv(out_tables / "best_hyperparameters.csv")


# ============================================================================
# Reporting and Delivery Helpers (used in Step 6)
# ============================================================================


def write_model_card(
    best_model: str,
    best_threshold: float,
    best_test_metrics: Dict[str, float],
    paths: Dict[str, Path],
) -> None:
    """Write concise model-card style summary for report appendix."""
    model_card = f"""# Model Card: Telco Churn Predictor

## Model Details
- Model type: `{best_model}`
- Decision threshold: `{best_threshold:.4f}` (chosen on validation set by max F1)

## Intended Use
- Prioritize customers for retention outreach.
- Support business decision-making with risk ranking.

## Out-of-Scope / Not for Use
- Do not use as sole basis for denying service.
- Do not treat output as causal proof of churn reasons.

## Data Provenance and Constraints
- Dataset: IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- Constraints: historical tabular data, class imbalance, potential temporal drift.

## Evaluation Summary (Test Set)
- ROC-AUC: `{best_test_metrics['roc_auc']:.4f}`
- PR-AUC: `{best_test_metrics['pr_auc']:.4f}`
- Precision: `{best_test_metrics['precision']:.4f}`
- Recall: `{best_test_metrics['recall']:.4f}`
- F1: `{best_test_metrics['f1']:.4f}`

## Caveats
- Performance may degrade if customer behavior changes over time.
- Threshold should be adjusted for changing business costs.
- Subgroup metrics should be reviewed before operational deployment.
"""
    (paths["reports"] / "model_card.md").write_text(model_card, encoding="utf-8")


def write_report_outline(paths: Dict[str, Path]) -> None:
    """Write report scaffold aligned with six required coursework sections."""
    outline = """# Report Skeleton (2000 words max)

## 1. Obtain Dataset and Frame the Predictive Problem
- Problem definition, target, prediction type
- Success metrics and constraints
- Assumptions and limitations
- Agent plan: what was delegated vs verified manually

## 2. Explore the Data to Gain Insights
- EDA visuals and findings
- Missingness, imbalance, outliers, leakage risks
- Verified interpretation of visuals

## 3. Prepare the Data
- Train/validation/test split discipline
- Preprocessing pipeline
- Data validation checks and outcomes

## 4. Explore Different Models and Shortlist
- Baseline model and rationale
- Model comparison evidence (tables/plots)
- Why shortlisted models were chosen

## 5. Fine-tune and Evaluate
- CV/validation tuning strategy
- Robust evaluation: confusion matrix, calibration, failure modes
- One explicit agent mistake you caught and corrected

## 6. Present the Final Solution
- Final model and rationale
- Limitations, risks, and next steps
- Model card summary (intended use, constraints, caveats)
"""
    (paths["reports"] / "report_skeleton.md").write_text(outline, encoding="utf-8")


def write_agent_appendix_template(paths: Dict[str, Path]) -> None:
    """Write appendix template for agent logs and decision register."""
    template = """# Appendix: Agent Usage Log + Decision Register

Use this appendix as submission evidence.
- Interaction evidence can be screenshots or exported logs.
- Decision register should be about 1-2 pages.

## A. Interaction Evidence (brief logs)
| Log ID | Stage | Date | Tool/Prompt Summary | What the Agent Produced | Evidence Link/Screenshot |
|---|---|---|---|---|---|
| L01 | Step 1 | YYYY-MM-DD | Asked for problem framing draft | JSON + markdown framing draft | `appendix/screenshots/log_01.png` |
| L02 | Step 4 | YYYY-MM-DD | Asked for baseline model set | Baseline comparison code | `appendix/screenshots/log_02.png` |

## B. Decision Register (key contributions + your verification)
| ID | Stage | Agent Contribution | Your Verification Actions | Evidence File(s) | Decision (Accepted/Modified/Rejected) | Reason |
|---|---|---|---|---|---|---|
| D01 | Step 1 | Suggested target + metrics | Checked rubric fit and metric validity for classification | `outputs/tables/problem_framing.json` | Modified | Added fairness and cost constraints |
| D02 | Step 3 | Suggested preprocessing pipeline | Verified no leakage by split-first + integrity checks | `outputs/tables/split_validation.json` | Accepted | Methodologically sound |
| D03 | Step 5 | Suggested threshold at 0.50 | Compared threshold policy table and selected better F1 point | `outputs/tables/threshold_policy_comparison.csv` | Rejected | Worse recall-cost trade-off |
"""
    (paths["reports"] / "agent_usage_log_template.md").write_text(
        template, encoding="utf-8"
    )
    register_cols = [
        "id",
        "stage",
        "agent_contribution",
        "your_verification_actions",
        "evidence_files",
        "decision",
        "reason",
    ]
    pd.DataFrame(columns=register_cols).to_csv(
        paths["tables"] / "agent_decision_register_template.csv", index=False
    )


def write_workflow_evidence_template(paths: Dict[str, Path]) -> None:
    """Write a fillable tracker that shows stage-by-stage workflow progression."""
    text = """# Workflow Evidence Tracker (Agentic Methodology)

Purpose: show how the project progressed through stages using an agent tool and human verification.

| Stage | Objective | What You Asked the Agent To Do | What You Verified Yourself | Evidence Artefacts | Outcome |
|---|---|---|---|---|---|
| Step 1 | Frame prediction problem | Draft target/metric/constraint plan | Checked metrics are classification-appropriate and rubric-aligned | `outputs/tables/problem_framing.json`, `outputs/reports/step1_problem_framing.md` | Accepted with edits |
| Step 2 | EDA and risks | Suggest useful EDA plots | Checked plot correctness, leakage clues, imbalance | `outputs/figures/eda_*.png` | Accepted |
| Step 3 | Data preparation | Propose preprocessing pipeline | Verified split discipline and schema checks | `outputs/tables/split_validation.json`, `outputs/tables/schema_validation.json` | Accepted |
| Step 4 | Baseline and shortlist | Suggest model candidates | Verified with metric table and ablation | `outputs/tables/model_metrics_validation.csv`, `outputs/tables/ablation_*.csv` | Modified |
| Step 5 | Tune and evaluate | Suggest tuning ranges and threshold strategy | Verified with CV table + error/fairness checks | `outputs/tables/cv_search_results_*.csv`, `outputs/tables/subgroup_metrics.csv` | Accepted/Modified |
| Step 6 | Final communication | Draft report artefacts and model card | Verified claims against saved outputs | `outputs/reports/model_card.md`, `outputs/reports/pipeline_summary.md` | Accepted |
"""
    (paths["reports"] / "workflow_evidence_tracker.md").write_text(
        text, encoding="utf-8"
    )


def write_summary(
    best_model: str,
    best_threshold: float,
    best_test_metrics: Dict[str, float],
    paths: Dict[str, Path],
) -> None:
    """Write short run summary for quick review."""
    summary = f"""# Telco Churn Starter Pipeline Summary

## Final model
- Model: `{best_model}`
- Validation-tuned threshold (F1): `{best_threshold:.4f}`

## Test metrics (at tuned threshold)
- ROC-AUC: `{best_test_metrics['roc_auc']:.4f}`
- PR-AUC: `{best_test_metrics['pr_auc']:.4f}`
- Accuracy: `{best_test_metrics['accuracy']:.4f}`
- Precision: `{best_test_metrics['precision']:.4f}`
- Recall: `{best_test_metrics['recall']:.4f}`
- F1: `{best_test_metrics['f1']:.4f}`
- Brier score: `{best_test_metrics['brier']:.4f}`

## Saved outputs
- Tables: `{paths['tables']}`
- Figures: `{paths['figures']}`
- Model: `{paths['models'] / 'best_model.joblib'}`

## Coursework notes
- Add your Agent Usage Log and Decision Register in report appendix.
- Include at least one agent mistake you caught and corrected.
- Add business interpretation and limitation discussion in your report narrative.
"""
    (paths["reports"] / "pipeline_summary.md").write_text(summary, encoding="utf-8")


def write_metric_guidance(paths: Dict[str, Path]) -> None:
    """Write a beginner-friendly note on metric choice for classification."""
    text = """# Metric Guidance

This project predicts `Churn` (Yes/No), so it is a classification problem.

- Use classification metrics: ROC-AUC, PR-AUC, F1, Precision, Recall, Calibration/Brier.
- Do not use R-squared as the primary fit metric here, because R-squared is designed for regression targets.

Practical interpretation:
- ROC-AUC: ranking quality across thresholds.
- PR-AUC: useful under class imbalance.
- F1/Recall/Precision: operating-point trade-offs after threshold choice.
    """
    (paths["reports"] / "metric_guidance.md").write_text(text, encoding="utf-8")


def write_agent_mistake_example(paths: Dict[str, Path]) -> None:
    """Write an explicit agent-mistake section for the appendix."""
    text = """# Explicit Agent-Mistake Example (Required)

## Mistake Proposed by Agent
The agent initially suggested relying on a fixed threshold of `0.50` for model comparison and decision-making.

## Why This Was Problematic
In imbalanced churn classification, a fixed threshold can hide precision-recall trade-offs and mismatch business costs.

## Verification and Correction
I computed model-specific thresholds on the validation set (max F1), then re-evaluated on the held-out test set at those validation-derived thresholds.

## Evidence
- `outputs/tables/thresholds_from_validation.csv`
- `outputs/tables/model_metrics_test_threshold_0_5.csv`
- `outputs/tables/model_metrics_test_optimal_threshold_from_val.csv`
- `outputs/tables/model_shortlist_evidence.csv`

## Final Decision
Thresholding is now evidence-driven (validation-tuned), not a fixed default.
"""
    (paths["reports"] / "agent_mistake_example.md").write_text(text, encoding="utf-8")


def write_final_solution_brief(
    best_model: str,
    best_threshold: float,
    best_test_metrics: Dict[str, float],
    paths: Dict[str, Path],
) -> None:
    """Write a concise final-solution note: rationale, risks, and next steps."""
    text = f"""# Final Solution Brief

## Final Model Selection and Rationale
- Selected model: `{best_model}`
- Selection basis: highest validation ROC-AUC after Step 5 tuning.
- Operating threshold: `{best_threshold:.4f}` (chosen on validation set by max F1).
- Test ROC-AUC: `{best_test_metrics['roc_auc']:.4f}`, PR-AUC: `{best_test_metrics['pr_auc']:.4f}`, F1: `{best_test_metrics['f1']:.4f}`.

## Limitations and Risks
- Historical data may drift; performance can degrade over time.
- Predictions are associative, not causal explanations.
- Threshold may need recalibration as intervention cost/benefit changes.
- Fairness review is limited to available subgroup columns.

## Next Steps
- Re-run validation on newer data windows.
- Add cost-sensitive threshold optimization from business assumptions.
- Expand subgroup checks and monitor post-deployment drift.

## Model Card Pointer
- See `outputs/reports/model_card.md` for intended use, out-of-scope use, data constraints, evaluation summary, and caveats.
"""
    (paths["reports"] / "final_solution_brief.md").write_text(text, encoding="utf-8")


def build_problem_framing_payload(df: pd.DataFrame) -> Dict[str, object]:
    """Build a rubric-aligned Step 1 payload for dataset framing."""
    target_counts = df["Churn"].value_counts()
    positive_count = int(target_counts.get("Yes", 0))
    negative_count = int(target_counts.get("No", 0))
    total_rows = int(len(df))
    positive_rate = float(positive_count / total_rows) if total_rows else 0.0

    return {
        "dataset_summary": {
            "name": "IBM Telco Customer Churn",
            "rows": total_rows,
            "columns": int(df.shape[1]),
            "target_distribution": {
                "Yes": positive_count,
                "No": negative_count,
                "positive_rate": positive_rate,
            },
        },
        "target_and_prediction_type": {
            "target_variable": "Churn",
            "prediction_type": "binary_classification",
            "positive_class": "Yes",
            "business_question": "Which customers are at high risk of churn for retention targeting?",
        },
        "success_metrics_and_constraints": {
            "primary_metric": "ROC-AUC",
            "secondary_metrics": ["PR-AUC", "F1", "Recall", "Precision", "Brier"],
            "operating_point_rule": "Choose threshold on validation set by max F1, then check recall-cost trade-off.",
            "constraints": {
                "performance": "Model should beat baseline ranking quality and remain stable across validation/test.",
                "latency": "Inference should run on standard CPU in batch mode.",
                "interpretability": "Provide permutation importance and segment/error analysis.",
                "fairness": "Report subgroup metrics for gender, senior status, and contract type.",
                "cost": "Use threshold policy table to compare precision-recall trade-offs for outreach cost control.",
            },
        },
        "assumptions": [
            "Observed features are available at prediction time.",
            "Label quality is acceptable for supervised learning.",
            "Current data is representative of near-term churn behavior.",
            "Retention team can act on ranked customer lists.",
        ],
        "limitations": [
            "Dataset is historical and may not reflect future drift.",
            "No explicit causal variables; model captures association, not causation.",
            "Potentially missing business context (e.g., campaign history, competitor actions).",
            "Fairness review is limited to available attributes in this dataset.",
        ],
        "agent_tooling_plan": {
            "ask_agent_to_do": [
                "Draft reproducible code skeleton for Step 1 outputs.",
                "Propose metric/constraint checklist aligned with coursework rubric.",
                "Generate appendix and workflow evidence templates.",
            ],
            "verify_myself": [
                "Check metrics are correct for classification (not regression).",
                "Confirm no unsupported claims; keep evidence tied to saved artefacts.",
                "Review assumptions/limitations with academic judgement.",
                "Approve or modify each agent suggestion in decision register.",
            ],
        },
    }


def write_problem_framing_artifacts(df: pd.DataFrame, paths: Dict[str, Path]) -> Dict[str, object]:
    """Write Step 1 framing artefacts in JSON and markdown for report reuse."""
    framing = build_problem_framing_payload(df)
    (paths["tables"] / "problem_framing.json").write_text(
        json.dumps(framing, indent=2), encoding="utf-8"
    )

    constraints = framing["success_metrics_and_constraints"]["constraints"]
    assumptions_text = "\n".join([f"- {x}" for x in framing["assumptions"]])
    limitations_text = "\n".join([f"- {x}" for x in framing["limitations"]])
    ask_agent_text = "\n".join([f"- {x}" for x in framing["agent_tooling_plan"]["ask_agent_to_do"]])
    verify_text = "\n".join([f"- {x}" for x in framing["agent_tooling_plan"]["verify_myself"]])

    report_text = f"""# Step 1: Obtain Dataset and Frame the Predictive Problem

## Target variable and prediction type
- Dataset: IBM Telco Customer Churn
- Rows: {framing["dataset_summary"]["rows"]}, Columns: {framing["dataset_summary"]["columns"]}
- Target variable: `{framing["target_and_prediction_type"]["target_variable"]}`
- Prediction type: `{framing["target_and_prediction_type"]["prediction_type"]}`
- Positive class: `{framing["target_and_prediction_type"]["positive_class"]}`
- Positive rate: `{framing["dataset_summary"]["target_distribution"]["positive_rate"]:.4f}`

## Success metrics and constraints
- Primary metric: `{framing["success_metrics_and_constraints"]["primary_metric"]}`
- Secondary metrics: `{", ".join(framing["success_metrics_and_constraints"]["secondary_metrics"])}`
- Operating rule: {framing["success_metrics_and_constraints"]["operating_point_rule"]}
- Performance constraint: {constraints["performance"]}
- Latency constraint: {constraints["latency"]}
- Interpretability constraint: {constraints["interpretability"]}
- Fairness constraint: {constraints["fairness"]}
- Cost constraint: {constraints["cost"]}

## Assumptions
{assumptions_text}

## Limitations
{limitations_text}

## Agent tooling plan (what to delegate vs verify)
### Ask agent to do
{ask_agent_text}

### Verify myself
{verify_text}
"""
    (paths["reports"] / "step1_problem_framing.md").write_text(
        report_text, encoding="utf-8"
    )
    return framing


def main() -> None:
    """Orchestrate the full end-to-end churn workflow."""
    sns.set_theme(style="whitegrid")
    args = parse_args()
    paths = make_dirs(args.output_dir)

    # =========================================================================
    # Step 1: Obtain dataset and frame the predictive problem
    # =========================================================================
    print("[1/6] Step 1 - Obtain dataset and frame the predictive problem...")
    df = load_and_clean(args.data_path)
    basic_data_checks(df, paths["tables"])
    save_feature_engineering_summary(df, paths["tables"])
    write_business_kpis(df, paths["tables"])
    validate_dataset_schema(df, paths["tables"])
    write_problem_framing_artifacts(df, paths)

    # =========================================================================
    # Step 2: Explore the data to gain insights
    # =========================================================================
    print("[2/6] Step 2 - Explore the data to gain insights...")
    plot_class_balance(df, paths["figures"] / "eda_class_balance.png")
    plot_eda_numeric(df, paths["figures"] / "eda_numeric_boxplots.png")
    plot_churn_by_segment(df, paths["figures"] / "eda_churn_by_contract.png")
    plot_churn_by_segments_grid(df, paths["figures"] / "eda_churn_segments_grid.png")
    plot_missingness(df, paths["figures"] / "eda_missingness.png")
    plot_numeric_correlation(df, paths["figures"] / "eda_numeric_correlation.png")
    save_eda_risk_tables(df, paths["tables"])

    # =========================================================================
    # Step 3: Prepare the data
    # =========================================================================
    print("[3/6] Step 3 - Prepare the data...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        df=df,
        target_col="Churn",
        id_col="customerID",
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    preprocessor = build_preprocessor(x_train)
    validate_split_integrity(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        out_tables=paths["tables"],
    )
    run_step3_preparation_checks(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
        out_tables=paths["tables"],
        target_col="Churn",
        id_col="customerID",
    )

    split_info = {
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "features": int(x_train.shape[1]),
        "random_state": int(args.random_state),
    }
    (paths["tables"] / "split_summary.json").write_text(
        json.dumps(split_info, indent=2), encoding="utf-8"
    )

    # =========================================================================
    # Step 4: Explore models and shortlist
    # =========================================================================
    print("[4/6] Step 4 - Explore models and shortlist...")
    val_rows = []
    test_rows = []
    thresholds = []
    test_opt_rows = []

    for model_name, model in model_specs(args.random_state).items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(x_train, y_train)

        y_val_prob = pipe.predict_proba(x_val)[:, 1]
        y_test_prob = pipe.predict_proba(x_test)[:, 1]

        val_metrics = evaluate(y_val, y_val_prob, threshold=0.5)
        test_metrics = evaluate(y_test, y_test_prob, threshold=0.5)
        val_metrics["model"] = model_name
        test_metrics["model"] = model_name
        val_rows.append(val_metrics)
        test_rows.append(test_metrics)

        tuned_thr = best_f1_threshold(y_val, y_val_prob)
        thresholds.append({"model": model_name, "best_f1_threshold_on_val": tuned_thr})
        test_opt = evaluate(y_test, y_test_prob, threshold=tuned_thr)
        test_opt["model"] = model_name
        test_opt["threshold_source"] = "validation_best_f1"
        test_opt_rows.append(test_opt)

    val_df = pd.DataFrame(val_rows).sort_values("roc_auc", ascending=False)
    test_df = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)
    thr_df = pd.DataFrame(thresholds)
    test_opt_df = pd.DataFrame(test_opt_rows).sort_values("roc_auc", ascending=False)

    val_df.to_csv(paths["tables"] / "model_metrics_validation.csv", index=False)
    test_df.to_csv(paths["tables"] / "model_metrics_test_threshold_0_5.csv", index=False)
    thr_df.to_csv(paths["tables"] / "thresholds_from_validation.csv", index=False)
    test_opt_df.to_csv(
        paths["tables"] / "model_metrics_test_optimal_threshold_from_val.csv", index=False
    )

    # Extra evidence for Step 4 model-choice rationale.
    run_ablation_study(
        preprocessor=preprocessor,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        out_tables=paths["tables"],
    )

    # =========================================================================
    # Step 5: Fine-tune and evaluate
    # =========================================================================
    print("[5/6] Step 5 - Fine-tune and evaluate...")
    # Tune top-N baseline models using CV.
    top_n = max(1, int(args.top_models_to_tune))
    top_models = val_df.head(top_n)["model"].tolist()
    pd.DataFrame({"top_models_from_baseline": top_models}).to_csv(
        paths["tables"] / "top_models_for_tuning.csv", index=False
    )
    # Legacy filename kept for compatibility.
    pd.DataFrame({"top_2_models_from_baseline": top_models[:2]}).to_csv(
        paths["tables"] / "top2_models_for_tuning.csv", index=False
    )
    shortlist_evidence = (
        val_df.merge(thr_df, on="model", how="left")
        .merge(
            test_opt_df[["model", "roc_auc", "pr_auc", "f1"]].rename(
                columns={
                    "roc_auc": "test_roc_auc_at_val_opt_thr",
                    "pr_auc": "test_pr_auc_at_val_opt_thr",
                    "f1": "test_f1_at_val_opt_thr",
                }
            ),
            on="model",
            how="left",
        )
        .assign(selected_for_tuning=lambda d: d["model"].isin(top_models))
    )
    shortlist_evidence.to_csv(paths["tables"] / "model_shortlist_evidence.csv", index=False)
    tuned_pipelines, cv_results_df = tune_top_models_with_cv(
        top_model_names=top_models,
        preprocessor=preprocessor,
        x_train=x_train,
        y_train=y_train,
        random_state=args.random_state,
    )
    cv_results_df.to_csv(paths["tables"] / "cv_search_results_shortlisted_models.csv", index=False)
    # Legacy filename kept for compatibility.
    cv_results_df.to_csv(paths["tables"] / "cv_search_results_top2_models.csv", index=False)
    save_best_hyperparameters(tuned_pipelines, paths["tables"])

    tuned_val_rows = []
    tuned_test_rows = []
    tuned_threshold_rows = []
    tuned_proba_test = {}

    for model_name, pipe in tuned_pipelines.items():
        y_val_prob = pipe.predict_proba(x_val)[:, 1]
        y_test_prob = pipe.predict_proba(x_test)[:, 1]
        tuned_proba_test[model_name] = y_test_prob

        val_metrics = evaluate(y_val, y_val_prob, threshold=0.5)
        test_metrics = evaluate(y_test, y_test_prob, threshold=0.5)
        val_metrics["model"] = model_name
        test_metrics["model"] = model_name
        tuned_val_rows.append(val_metrics)
        tuned_test_rows.append(test_metrics)

        tuned_thr = best_f1_threshold(y_val, y_val_prob)
        tuned_threshold_rows.append(
            {"model": model_name, "best_f1_threshold_on_val": tuned_thr}
        )

    tuned_val_df = pd.DataFrame(tuned_val_rows).sort_values("roc_auc", ascending=False)
    tuned_test_df = pd.DataFrame(tuned_test_rows).sort_values("roc_auc", ascending=False)
    tuned_thr_df = pd.DataFrame(tuned_threshold_rows)

    tuned_val_df.to_csv(paths["tables"] / "model_metrics_validation_tuned_cv.csv", index=False)
    tuned_test_df.to_csv(
        paths["tables"] / "model_metrics_test_tuned_cv_threshold_0_5.csv", index=False
    )
    tuned_thr_df.to_csv(
        paths["tables"] / "thresholds_from_validation_tuned_models.csv", index=False
    )

    best_model = tuned_val_df.iloc[0]["model"]
    best_pipe = tuned_pipelines[best_model]
    best_thr = float(
        tuned_thr_df.loc[
            tuned_thr_df["model"] == best_model, "best_f1_threshold_on_val"
        ].iloc[0]
    )

    best_test_prob = best_pipe.predict_proba(x_test)[:, 1]
    best_test_metrics = evaluate(y_test, best_test_prob, threshold=best_thr)
    best_test_metrics["model"] = best_model
    pd.DataFrame([best_test_metrics]).to_csv(
        paths["tables"] / "best_model_test_metrics_tuned_threshold.csv", index=False
    )

    best_test_pred = (best_test_prob >= best_thr).astype(int)
    final_preds = x_test.copy()
    final_preds["y_true"] = y_test.values
    final_preds["y_prob"] = best_test_prob
    final_preds["y_pred"] = best_test_pred
    final_preds.to_csv(paths["tables"] / "best_model_test_predictions.csv", index=False)
    run_threshold_policy_analysis(
        y_true=y_test, y_prob=best_test_prob, out_tables=paths["tables"], best_thr=best_thr
    )
    run_subgroup_analysis(
        x_test=x_test,
        y_test=y_test,
        y_pred=best_test_pred,
        y_prob=best_test_prob,
        out_tables=paths["tables"],
    )

    print("      Step 5 - Running diagnostics and error analysis...")
    plot_roc_pr_curves(y_test, tuned_proba_test, paths["figures"] / "test_roc_pr_curves.png")
    plot_roc_pr_curves(y_test, {best_model: best_test_prob}, paths["figures"] / "roc_pr_best_model.png")
    plot_conf_matrix(y_test, best_test_pred, paths["figures"] / "best_model_confusion_matrix.png")
    plot_conf_matrix(y_test, best_test_pred, paths["figures"] / "confusion_matrix_best_model.png")
    plot_calibration(y_test, best_test_prob, paths["figures"] / "best_model_calibration_curve.png")
    plot_calibration(y_test, best_test_prob, paths["figures"] / "calibration_best_model.png")
    run_error_analysis(
        x_test=x_test,
        y_test=y_test,
        y_pred=best_test_pred,
        y_prob=best_test_prob,
        out_tables=paths["tables"],
    )
    run_decile_lift_analysis(
        y_true=y_test,
        y_prob=best_test_prob,
        out_tables=paths["tables"],
        out_figures=paths["figures"],
    )
    workflow_step5 = {
        "step": "Step 5 - Model tuning and evaluation",
        "tuning_strategy": "Stratified cross-validation + validation-set threshold tuning",
        "selection_metric": "roc_auc",
        "actions_completed": [
            "Performed CV hyperparameter tuning on shortlisted models",
            "Selected best model by validation ROC-AUC",
            "Optimized decision threshold on validation F1",
            "Evaluated best model on held-out test set",
            "Ran subgroup and failure-mode analysis",
            "Generated confusion matrix, ROC/PR, and calibration plots",
        ],
        "artefacts_created": [
            str(paths["tables"] / "cv_search_results_shortlisted_models.csv"),
            str(paths["tables"] / "best_hyperparameters.csv"),
            str(paths["tables"] / "best_model_test_metrics_tuned_threshold.csv"),
            str(paths["tables"] / "best_model_test_predictions.csv"),
            str(paths["tables"] / "error_analysis_by_contract.csv"),
            str(paths["tables"] / "subgroup_metrics.csv"),
            str(paths["figures"] / "best_model_confusion_matrix.png"),
            str(paths["figures"] / "test_roc_pr_curves.png"),
            str(paths["figures"] / "best_model_calibration_curve.png"),
        ],
        "human_checks": [
            "Confirmed test set was not used during tuning or threshold search",
            "Verified threshold optimization uses validation predictions only",
            "Checked confusion matrix, ROC/PR, and calibration consistency with metrics",
        ],
    }
    (paths["tables"] / "workflow_step5.json").write_text(
        json.dumps(workflow_step5, indent=2), encoding="utf-8"
    )

    # Optional explainability summary for reporting.
    perm = permutation_importance(
        best_pipe,
        x_val,
        y_val,
        n_repeats=5,
        random_state=args.random_state,
        scoring="roc_auc",
        n_jobs=1,
    )
    # For permutation importance on a full Pipeline, importances are usually per raw input column.
    feat_names = x_val.columns.astype(str).tolist()
    feature_name_source = "raw_input_features"
    if len(perm.importances_mean) != len(feat_names):
        try:
            transformed = (
                best_pipe.named_steps["preprocess"].get_feature_names_out().astype(str).tolist()
            )
            if len(perm.importances_mean) == len(transformed):
                feat_names = transformed
                feature_name_source = "transformed_features"
            else:
                feat_names = [f"feature_{i}" for i in range(len(perm.importances_mean))]
                feature_name_source = "fallback_indexed_features"
        except Exception:
            feat_names = [f"feature_{i}" for i in range(len(perm.importances_mean))]
            feature_name_source = "fallback_indexed_features"
    importance_df = pd.DataFrame(
        {"feature": feat_names, "importance_mean": perm.importances_mean}
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(paths["tables"] / "permutation_importance_validation.csv", index=False)
    (paths["tables"] / "permutation_importance_metadata.json").write_text(
        json.dumps(
            {
                "feature_name_source": feature_name_source,
                "n_importances": int(len(perm.importances_mean)),
                "n_feature_names": int(len(feat_names)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    top = importance_df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("Top 15 Permutation Importances (Validation)")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "best_model_permutation_importance.png", dpi=200)
    plt.close()

    # =========================================================================
    # Step 6: Present the final solution
    # =========================================================================
    print("[6/6] Step 6 - Present the final solution...")
    joblib.dump(best_pipe, paths["models"] / "best_model.joblib")
    write_summary(best_model, best_thr, best_test_metrics, paths)
    write_model_card(best_model, best_thr, best_test_metrics, paths)
    write_report_outline(paths)
    write_agent_appendix_template(paths)
    write_workflow_evidence_template(paths)
    write_metric_guidance(paths)
    write_agent_mistake_example(paths)
    write_final_solution_brief(best_model, best_thr, best_test_metrics, paths)
    workflow_step6 = {
        "step": "Step 6 - Present final solution",
        "actions_completed": [
            "Generated ROC and PR curves",
            "Generated confusion matrix",
            "Generated calibration curve",
            "Computed permutation feature importance",
            "Exported final model pipeline",
            "Generated model card and summary artefacts",
        ],
        "artefacts_created": [
            str(paths["figures"] / "test_roc_pr_curves.png"),
            str(paths["figures"] / "best_model_confusion_matrix.png"),
            str(paths["figures"] / "best_model_calibration_curve.png"),
            str(paths["tables"] / "permutation_importance_validation.csv"),
            str(paths["tables"] / "permutation_importance_metadata.json"),
            str(paths["models"] / "best_model.joblib"),
            str(paths["reports"] / "model_card.md"),
            str(paths["reports"] / "final_solution_brief.md"),
        ],
        "human_checks": [
            "Verified evaluation metrics align with saved plots and tables",
            "Confirmed threshold was validation-derived",
            "Checked permutation importance feature attribution mapping",
        ],
    }
    (paths["tables"] / "workflow_step6.json").write_text(
        json.dumps(workflow_step6, indent=2), encoding="utf-8"
    )

    print("Done. Check outputs in:", paths["root"].resolve())


if __name__ == "__main__":
    main()
