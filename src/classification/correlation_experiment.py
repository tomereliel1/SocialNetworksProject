"""Correlation pruning and sparse-logistic follow-up experiments.

This file is responsible for testing whether highly correlated structural
features can be removed, or sparse Logistic Regression can be used, without
hurting train-only development performance. It reuses the frozen `train.csv`
workflow, keeps the held-out test set untouched, evaluates candidate pipelines
with leakage-safe cross-validation, and writes comparison tables, figures,
out-of-fold diagnostics, and a Markdown report.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-social-networks")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classification.constants import POSITIVE_LABEL, RANDOM_STATE
from src.classification.features import make_numeric_frame, select_model_features
from src.classification.modeling import make_cv_splits


REFERENCE_THRESHOLD = 0.485
DEFAULT_THRESHOLD = 0.500
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
REAL_COLOR = "#3B6EA8"
FAKE_COLOR = "#C44E52"
ACCENT_COLOR = "#55A868"
NEUTRAL_COLOR = "#5E6472"


@dataclass(frozen=True)
class CorrelationExperimentArtifacts:
    """Artifacts produced by the correlation-pruning follow-up experiment.

    Purpose:
        Bundle the main in-memory result tables and selected configuration that
        `run_correlation_experiment` also writes to disk.

    Attributes:
        model_comparison: Mean and standard-deviation cross-validation metrics
            for each evaluated configuration and decision threshold.
        cv_results: Per-fold metrics, feature counts, removed features, and
            sparse-model best-parameter annotations.
        feature_counts: Summary of original, removed, and retained feature
            counts by configuration.
        removed_feature_frequency: Features most often removed across
            cross-validation folds by the correlation pruner.
        coefficient_comparison: Logistic coefficient diagnostics for baseline,
            pruned, and sparse configurations.
        sparse_model_features: Sparse-model coefficient table marking zero and
            nonzero selected coefficients.
        best_configuration: JSON-safe description of the selected development
            configuration and experiment policy.
    """

    model_comparison: pd.DataFrame
    cv_results: pd.DataFrame
    feature_counts: pd.DataFrame
    removed_feature_frequency: pd.DataFrame
    coefficient_comparison: pd.DataFrame
    sparse_model_features: pd.DataFrame
    best_configuration: dict[str, object]


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop redundant columns using correlations learned on the training fold only.

    Purpose:
        Provide an sklearn-compatible transformer that can live inside a
        `Pipeline`. In cross-validation, `fit` receives only the training
        partition, computes the correlation matrix there, and stores the
        columns to retain.

    Attributes:
        threshold: Minimum absolute Spearman correlation used to consider a
            feature pair redundant.
    """

    def __init__(self, threshold: float = 0.98):
        """Initialize the deterministic correlation-pruning transformer.

        Purpose:
            Store the absolute Spearman correlation threshold used to decide
            whether one feature from a pair should be removed.

        Args:
            threshold: Minimum absolute pairwise correlation required for a
                pair to be considered redundant.
        """

        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn which columns to retain from the fitting partition.

        Purpose:
            Compute a train-fold-only Spearman correlation matrix and build the
            deterministic pruning table used later by `transform`.

        Args:
            X: Feature matrix as a pandas dataframe or array-like object.
            y: Optional target values accepted for sklearn compatibility and
                ignored by this transformer.

        Returns:
            The fitted `CorrelationPruner` instance.
        """

        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        corr = frame.apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
        self.pruning_table_ = _pruning_table(corr, self.threshold, self.feature_names_in_)
        removed = set(self.pruning_table_["removed_feature"]) if not self.pruning_table_.empty else set()
        self.removed_features_ = sorted(removed)
        self.retained_features_ = [feature for feature in self.feature_names_in_ if feature not in removed]
        return self

    def transform(self, X):
        """Return the fitted retained-feature subset.

        Purpose:
            Apply the fold-specific pruning decision learned by `fit` to any
            compatible feature matrix.

        Args:
            X: Feature matrix containing the columns observed during fitting.

        Returns:
            A dataframe containing only retained feature columns.
        """

        frame = _as_frame(X, columns=getattr(self, "feature_names_in_", None))
        return frame[self.retained_features_]

    def get_feature_names_out(self, input_features=None):
        """Return output feature names after pruning.

        Purpose:
            Provide the sklearn feature-name API so downstream code can inspect
            the columns that survived pruning.

        Args:
            input_features: Optional sklearn-compatible input feature names.
                The fitted retained names are returned regardless.

        Returns:
            Numpy object array of retained feature names.
        """

        return np.asarray(self.retained_features_, dtype=object)


def run_correlation_experiment(train_path: Path, results_dir: Path) -> CorrelationExperimentArtifacts:
    """Run the train-only correlated-feature and sparse-logistic follow-up.

    Purpose:
        Own the complete follow-up experiment: load `train.csv`, select
        classifier-safe structural features, audit high correlations, evaluate
        baseline/pruned/sparse Logistic Regression variants, select the best
        development configuration, and write all reproducibility artifacts.

    Args:
        train_path: Path to the frozen training CSV.
        results_dir: Directory where CSV, JSON, figure, and report artifacts
            should be written.

    Returns:
        A `CorrelationExperimentArtifacts` bundle containing the main generated
        result tables and selected configuration.

    Raises:
        ValueError: If the training data cannot satisfy the shared
            classification feature or label contract.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path, low_memory=False)
    selection = select_model_features(df)
    X = make_numeric_frame(df, selection.features)
    y = pd.to_numeric(df["labels"], errors="raise").astype(int)
    cv = make_cv_splits(df)

    full_corr = X.corr(method="spearman")
    correlation_pairs = _correlation_pairs(full_corr)
    correlation_pairs.to_csv(results_dir / "correlation_pairs.csv", index=False)

    pruning_reports = {}
    for threshold in [0.98, 0.95]:
        pruner = CorrelationPruner(threshold=threshold).fit(X)
        report = pruner.pruning_table_.copy()
        report.insert(0, "threshold", threshold)
        report.to_csv(results_dir / f"pruned_features_{int(threshold * 100):03d}.csv", index=False)
        pruning_reports[threshold] = report

    baseline = fixed_l2_pipeline()
    runs: list[dict[str, object]] = []
    per_fold_frames: list[pd.DataFrame] = []
    oof_frames: dict[str, pd.DataFrame] = {}
    coefficient_frames: dict[str, pd.DataFrame] = {}

    fixed_configs = [
        ("Baseline Logistic", baseline),
        ("Corr prune 0.98", fixed_l2_pipeline(correlation_threshold=0.98)),
        ("Corr prune 0.95", fixed_l2_pipeline(correlation_threshold=0.95)),
    ]
    for name, pipeline in fixed_configs:
        print(f"Evaluating {name}", flush=True)
        result = evaluate_pipeline(name, pipeline, X, y, cv)
        runs.extend(result["summary_rows"])
        per_fold_frames.append(result["fold_metrics"])
        oof_frames[name] = result["oof_predictions"]
        coefficient_frames[name] = fit_coefficients(pipeline, X, y, name)

    l1_search = sparse_search(
        "L1 Logistic",
        X,
        y,
        cv,
        parameter_grid=[{"classifier__C": c} for c in [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]],
        pipeline_factory=lambda params: sparse_pipeline(penalty="l1", solver="liblinear", **params),
    )
    runs.extend(l1_search["summary_rows"])
    per_fold_frames.append(l1_search["fold_metrics"])
    oof_frames["L1 Logistic"] = l1_search["oof_predictions"]
    coefficient_frames["L1 Logistic"] = fit_coefficients(l1_search["best_pipeline"], X, y, "L1 Logistic")

    elastic_search = sparse_search(
        "Elastic Net Logistic",
        X,
        y,
        cv,
        parameter_grid=[
            {"classifier__C": c, "classifier__l1_ratio": ratio}
            for c, ratio in itertools.product([0.01, 0.03, 0.1, 0.3, 1.0], [0.1, 0.5, 0.9])
        ],
        pipeline_factory=lambda params: sparse_pipeline(penalty="elasticnet", solver="saga", **params),
    )
    runs.extend(elastic_search["summary_rows"])
    per_fold_frames.append(elastic_search["fold_metrics"])
    oof_frames["Elastic Net Logistic"] = elastic_search["oof_predictions"]
    coefficient_frames["Elastic Net Logistic"] = fit_coefficients(
        elastic_search["best_pipeline"], X, y, "Elastic Net Logistic"
    )
    sparse_search_results = pd.concat(
        [l1_search["search_results"], elastic_search["search_results"]],
        ignore_index=True,
    )
    sparse_search_results.to_csv(results_dir / "sparse_hyperparameter_search.csv", index=False)

    model_comparison = pd.DataFrame(runs)
    model_comparison = add_baseline_deltas(model_comparison)
    model_comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    cv_results = pd.concat(per_fold_frames, ignore_index=True)
    cv_results.to_csv(results_dir / "cv_results.csv", index=False)

    feature_counts = feature_count_table(cv_results, len(selection.features))
    feature_counts.to_csv(results_dir / "feature_counts.csv", index=False)
    removed_frequency = removed_feature_frequency(cv_results)
    removed_frequency.to_csv(results_dir / "removed_feature_frequency.csv", index=False)

    coefficient_comparison = coefficient_comparison_table(coefficient_frames)
    coefficient_comparison.to_csv(results_dir / "coefficient_comparison.csv", index=False)

    sparse_features = sparse_feature_table(
        {
            "L1 Logistic": coefficient_frames["L1 Logistic"],
            "Elastic Net Logistic": coefficient_frames["Elastic Net Logistic"],
        }
    )
    sparse_features.to_csv(results_dir / "sparse_model_features.csv", index=False)

    primary = model_comparison[model_comparison["threshold"] == REFERENCE_THRESHOLD].copy()
    best = primary.sort_values(["f1_mean", "roc_auc_mean", "features_remaining_mean"], ascending=[False, False, True]).iloc[0]
    best_config = {
        "selected_configuration": best["configuration"],
        "threshold": float(best["threshold"]),
        "primary_metric": "f1",
        "data_used": str(train_path),
        "test_set_policy": "test.csv was not loaded or evaluated by this follow-up experiment",
        "baseline_reference": "Baseline Logistic with C=0.1, class_weight=balanced, threshold=0.485",
        "l1_best_params": l1_search["best_params"],
        "elastic_net_best_params": elastic_search["best_params"],
        "correlation_pruning_rule": "For pairs above threshold, retain the deterministic higher-priority structural feature; otherwise retain alphabetical first.",
    }
    (results_dir / "best_correlation_experiment_config.json").write_text(
        json.dumps(best_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    best_name = str(best["configuration"])
    write_oof_artifacts(results_dir, oof_frames[best_name], best_name)
    write_figures(
        figures_dir,
        model_comparison,
        feature_counts,
        coefficient_comparison,
        oof_frames[best_name],
        best_name,
        correlation_pairs,
    )
    write_report(
        results_dir,
        train_path,
        model_comparison,
        feature_counts,
        removed_frequency,
        coefficient_comparison,
        sparse_features,
        best_config,
        pruning_reports,
    )

    return CorrelationExperimentArtifacts(
        model_comparison=model_comparison,
        cv_results=cv_results,
        feature_counts=feature_counts,
        removed_feature_frequency=removed_frequency,
        coefficient_comparison=coefficient_comparison,
        sparse_model_features=sparse_features,
        best_configuration=best_config,
    )


def fixed_l2_pipeline(correlation_threshold: float | None = None) -> Pipeline:
    """Build the fixed L2 Logistic Regression comparison pipeline.

    Purpose:
        Reproduce the baseline Logistic Regression setup and optionally insert
        the train-fold-only correlation pruner before imputation and scaling.

    Args:
        correlation_threshold: Absolute Spearman threshold for correlation
            pruning, or `None` for the unpruned baseline.

    Returns:
        A sklearn pipeline ready for cross-validation.
    """

    steps = []
    if correlation_threshold is not None:
        steps.append(("correlation_pruner", CorrelationPruner(threshold=correlation_threshold)))
    steps.extend(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def sparse_pipeline(penalty: str, solver: str, **params) -> Pipeline:
    """Build a sparse Logistic Regression pipeline.

    Purpose:
        Construct the L1 or Elastic Net candidate pipelines used to test
        coefficient sparsity as an alternative to manual correlation pruning.

    Args:
        penalty: Logistic Regression penalty name, such as `l1` or
            `elasticnet`.
        solver: Solver compatible with the requested penalty.
        **params: Search parameters using sklearn pipeline names, such as
            `classifier__C` or `classifier__l1_ratio`.

    Returns:
        A median-imputation, standard-scaling, Logistic Regression pipeline.
    """

    classifier = LogisticRegression(
        penalty=penalty,
        solver=solver,
        class_weight="balanced",
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_STATE,
        **_classifier_params(params),
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def evaluate_pipeline(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict[str, object]:
    """Evaluate one pipeline with explicit cross-validation splits.

    Purpose:
        Fit a fresh clone on each training fold, collect validation
        probabilities, compute metrics at both the default and project reference
        thresholds, and record fold-specific feature pruning decisions.

    Args:
        name: Display name to store in output tables.
        pipeline: Sklearn pipeline with `predict_proba` support.
        X: Numeric model feature matrix.
        y: Binary label series where fake news is the positive class.
        cv: Iterable of `(train_index, validation_index)` cross-validation
            splits.

    Returns:
        Dictionary containing summary rows, per-fold metrics, and out-of-fold
        probabilities for the evaluated pipeline.
    """

    fold_rows = []
    oof = pd.DataFrame(index=X.index, data={"labels": y})
    for fold, (train_idx, valid_idx) in enumerate(cv, start=1):
        estimator = clone(pipeline)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_valid)[:, 1]
        retained, removed = pipeline_features(estimator, list(X.columns))
        for threshold in [DEFAULT_THRESHOLD, REFERENCE_THRESHOLD]:
            preds = (probabilities >= threshold).astype(int)
            row = {
                "configuration": name,
                "fold": fold,
                "threshold": threshold,
                "features_original": len(X.columns),
                "features_removed": len(removed),
                "features_remaining": len(retained),
                "removed_features": ";".join(removed),
                **metric_dict(y_valid, preds, probabilities),
            }
            fold_rows.append(row)
        oof.loc[X_valid.index, f"{name}__probability_fake"] = probabilities
    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = summarize_fold_metrics(name, fold_metrics)
    oof_predictions = pd.DataFrame({"labels": y, "probability_fake": oof[f"{name}__probability_fake"]})
    return {"summary_rows": summary_rows, "fold_metrics": fold_metrics, "oof_predictions": oof_predictions}


def sparse_search(name: str, X: pd.DataFrame, y: pd.Series, cv, parameter_grid, pipeline_factory) -> dict[str, object]:
    """Run a compact manual hyperparameter search for a sparse pipeline.

    Purpose:
        Evaluate each sparse Logistic Regression parameter setting, choose the
        one with the best mean F1 at the default threshold, and then re-evaluate
        that best pipeline under the standard reporting thresholds.

    Args:
        name: Display name for the sparse model family.
        X: Numeric model feature matrix.
        y: Binary label series where fake news is the positive class.
        cv: Iterable of cross-validation splits.
        parameter_grid: Iterable of sklearn-style parameter dictionaries.
        pipeline_factory: Callable that accepts one parameter dictionary and
            returns a configured sklearn pipeline.

    Returns:
        Dictionary with summary rows, per-fold metrics, out-of-fold
        predictions, best parameters, best pipeline, and the full search table.
    """

    search_rows = []
    evaluated: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}
    for params in parameter_grid:
        print(f"Evaluating {name} params={params}", flush=True)
        pipeline = pipeline_factory(params)
        label = f"{name} search"
        result = evaluate_pipeline(label, pipeline, X, y, cv)
        fold_metrics = result["fold_metrics"]
        mean_f1 = fold_metrics[fold_metrics["threshold"] == DEFAULT_THRESHOLD]["f1"].mean()
        key = tuple(sorted(params.items()))
        evaluated[key] = {"params": params, "pipeline": pipeline, "mean_f1": mean_f1}
        primary = fold_metrics[fold_metrics["threshold"] == DEFAULT_THRESHOLD]
        row = {"configuration": name, "params": json.dumps(params, sort_keys=True)}
        for metric in METRICS:
            row[f"mean_{metric}_default_threshold"] = primary[metric].mean()
            row[f"std_{metric}_default_threshold"] = primary[metric].std(ddof=1)
        search_rows.append(row)

    best = max(evaluated.values(), key=lambda item: item["mean_f1"])
    result = evaluate_pipeline(name, best["pipeline"], X, y, cv)
    result["fold_metrics"]["best_params"] = json.dumps(best["params"], sort_keys=True)
    search_frame = pd.DataFrame(search_rows)
    return {
        "summary_rows": result["summary_rows"],
        "fold_metrics": result["fold_metrics"],
        "oof_predictions": result["oof_predictions"],
        "best_params": best["params"],
        "best_pipeline": best["pipeline"],
        "search_results": search_frame,
    }


def summarize_fold_metrics(name: str, fold_metrics: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize per-fold metrics for each decision threshold.

    Purpose:
        Convert fold-level metric rows into the model-comparison schema by
        aggregating feature counts and metric means/standard deviations.

    Args:
        name: Configuration name to store in each summary row.
        fold_metrics: Per-fold rows produced by `evaluate_pipeline`.

    Returns:
        Summary dictionaries, one per evaluated decision threshold.
    """

    rows = []
    for threshold, group in fold_metrics.groupby("threshold", sort=True):
        row: dict[str, object] = {
            "configuration": name,
            "threshold": threshold,
            "folds": int(group["fold"].nunique()),
            "features_original": int(group["features_original"].iloc[0]),
            "features_removed_mean": group["features_removed"].mean(),
            "features_removed_min": int(group["features_removed"].min()),
            "features_removed_max": int(group["features_removed"].max()),
            "features_remaining_mean": group["features_remaining"].mean(),
            "features_remaining_min": int(group["features_remaining"].min()),
            "features_remaining_max": int(group["features_remaining"].max()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return rows


def add_baseline_deltas(model_comparison: pd.DataFrame) -> pd.DataFrame:
    """Add metric deltas relative to the reference baseline row.

    Purpose:
        Make the model-comparison table easier to interpret by expressing each
        candidate's precision, recall, F1, ROC-AUC, and PR-AUC against the
        unpruned baseline at threshold 0.485.

    Args:
        model_comparison: Summary table containing the baseline row and all
            candidate rows.

    Returns:
        Sorted copy of `model_comparison` with delta columns added.
    """

    out = model_comparison.copy()
    reference = out[
        (out["configuration"] == "Baseline Logistic") & (out["threshold"] == REFERENCE_THRESHOLD)
    ].iloc[0]
    for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        out[f"delta_{metric}"] = out[f"{metric}_mean"] - reference[f"{metric}_mean"]
    return out.sort_values(["threshold", "f1_mean", "roc_auc_mean"], ascending=[False, False, False])


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute the shared binary-classification metrics.

    Purpose:
        Keep metric calculation consistent across baseline, pruned, and sparse
        candidate evaluations.

    Args:
        y_true: True binary labels.
        y_pred: Thresholded binary predictions.
        y_score: Positive-class probability scores.

    Returns:
        Mapping of metric name to computed score.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def pipeline_features(estimator: Pipeline, original_features: list[str]) -> tuple[list[str], list[str]]:
    """Return retained and removed features for a fitted pipeline.

    Purpose:
        Hide the difference between pruned and unpruned pipelines so reporting
        code can always ask for feature-count diagnostics.

    Args:
        estimator: Fitted sklearn pipeline.
        original_features: Original feature order before any pruning step.

    Returns:
        Tuple of retained feature names and removed feature names.
    """

    if "correlation_pruner" not in estimator.named_steps:
        return original_features, []
    pruner = estimator.named_steps["correlation_pruner"]
    return list(pruner.retained_features_), list(pruner.removed_features_)


def fit_coefficients(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, configuration: str) -> pd.DataFrame:
    """Fit a pipeline on all training data and tabulate coefficients.

    Purpose:
        Produce coefficient diagnostics for interpretability after each
        configuration has already been evaluated by cross-validation.

    Args:
        pipeline: Logistic Regression pipeline to clone and fit.
        X: Numeric model feature matrix.
        y: Binary label series where fake news is the positive class.
        configuration: Display name to attach to each coefficient row.

    Returns:
        Dataframe with signed, absolute, and nonzero coefficient diagnostics.
    """

    estimator = clone(pipeline).fit(X, y)
    features, _ = pipeline_features(estimator, list(X.columns))
    coefficients = estimator.named_steps["classifier"].coef_[0]
    return pd.DataFrame(
        {
            "configuration": configuration,
            "feature": features,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "sign": np.where(coefficients >= 0, "positive_fake", "negative_fake"),
            "nonzero": np.abs(coefficients) > 1e-8,
        }
    )


def coefficient_comparison_table(coefficient_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine coefficient diagnostics across configurations.

    Purpose:
        Create one sortable table for comparing which features receive the
        largest Logistic Regression weights under baseline, pruned, and sparse
        settings.

    Args:
        coefficient_frames: Mapping from configuration name to coefficient
            dataframe.

    Returns:
        Concatenated coefficient table sorted by configuration and coefficient
        magnitude.
    """

    return pd.concat(coefficient_frames.values(), ignore_index=True).sort_values(
        ["configuration", "abs_coefficient"], ascending=[True, False]
    )


def sparse_feature_table(coefficient_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a zero/nonzero feature table for sparse models.

    Purpose:
        Show which L1 and Elastic Net coefficients survived sparse
        regularization and which were driven to zero.

    Args:
        coefficient_frames: Sparse-model coefficient tables keyed by
            configuration name.

    Returns:
        Dataframe marking each sparse coefficient as `nonzero` or `zero`.
    """

    rows = []
    for configuration, frame in coefficient_frames.items():
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "configuration": configuration,
                    "feature": row.feature,
                    "coefficient": row.coefficient,
                    "selected_nonzero": bool(row.nonzero),
                    "status": "nonzero" if row.nonzero else "zero",
                }
            )
    return pd.DataFrame(rows).sort_values(["configuration", "status", "feature"])


def feature_count_table(cv_results: pd.DataFrame, original_feature_count: int) -> pd.DataFrame:
    """Summarize feature reduction by configuration.

    Purpose:
        Report how many features each pruning or sparse configuration starts
        with, removes, and retains across folds.

    Args:
        cv_results: Per-fold evaluation results from all configurations.
        original_feature_count: Number of classifier-safe features before any
            pruning.

    Returns:
        Dataframe with original, removed, and remaining feature-count summaries.
    """

    rows = [
        {
            "configuration": "Baseline Logistic",
            "original_features": original_feature_count,
            "removed_mean": 0,
            "removed_min": 0,
            "removed_max": 0,
            "remaining_mean": original_feature_count,
            "remaining_min": original_feature_count,
            "remaining_max": original_feature_count,
        }
    ]
    for configuration, group in cv_results.groupby("configuration"):
        if configuration == "Baseline Logistic" or configuration.endswith("search"):
            continue
        base = group[group["threshold"] == DEFAULT_THRESHOLD]
        rows.append(
            {
                "configuration": configuration,
                "original_features": original_feature_count,
                "removed_mean": base["features_removed"].mean(),
                "removed_min": int(base["features_removed"].min()),
                "removed_max": int(base["features_removed"].max()),
                "remaining_mean": base["features_remaining"].mean(),
                "remaining_min": int(base["features_remaining"].min()),
                "remaining_max": int(base["features_remaining"].max()),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("configuration")


def removed_feature_frequency(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Count how often each feature is pruned across folds.

    Purpose:
        Identify redundant features that are consistently removed by the
        correlation pruner rather than only appearing in one fold's decision.

    Args:
        cv_results: Per-fold evaluation results containing semicolon-separated
            removed feature names.

    Returns:
        Dataframe with configuration, feature, and number of folds where that
        feature was removed.
    """

    rows = []
    for row in cv_results[cv_results["threshold"] == DEFAULT_THRESHOLD].itertuples(index=False):
        for feature in str(row.removed_features).split(";"):
            if feature and feature != "nan":
                rows.append({"configuration": row.configuration, "feature": feature, "fold": row.fold})
    if not rows:
        return pd.DataFrame(columns=["configuration", "feature", "fold_count"])
    return (
        pd.DataFrame(rows)
        .groupby(["configuration", "feature"], as_index=False)
        .agg(fold_count=("fold", "nunique"))
        .sort_values(["configuration", "fold_count", "feature"], ascending=[True, False, True])
    )


def write_oof_artifacts(results_dir: Path, oof: pd.DataFrame, configuration: str) -> None:
    """Write out-of-fold diagnostics for the selected configuration.

    Purpose:
        Save train-only prediction, ROC, precision-recall, and confusion-matrix
        artifacts for the best development configuration.

    Args:
        results_dir: Directory where diagnostics should be written.
        oof: Out-of-fold label and probability table.
        configuration: Selected configuration name. Accepted for report
            symmetry even though filenames encode the selected model generically.
    """

    out = oof.copy()
    out["prediction_0485"] = (out["probability_fake"] >= REFERENCE_THRESHOLD).astype(int)
    out.to_csv(results_dir / "best_oof_predictions_train_only.csv", index=False)
    fpr, tpr, roc_thresholds = roc_curve(out["labels"], out["probability_fake"], pos_label=POSITIVE_LABEL)
    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_thresholds}).to_csv(
        results_dir / "best_roc_curve_oof.csv", index=False
    )
    precision, recall, pr_thresholds = precision_recall_curve(
        out["labels"], out["probability_fake"], pos_label=POSITIVE_LABEL
    )
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(pr_thresholds, np.nan)}).to_csv(
        results_dir / "best_precision_recall_curve_oof.csv", index=False
    )
    conf = confusion_matrix(out["labels"], out["prediction_0485"], labels=[0, 1])
    pd.DataFrame(conf, index=["actual_real", "actual_fake"], columns=["pred_real", "pred_fake"]).to_csv(
        results_dir / "best_confusion_matrix_oof.csv"
    )


def write_figures(
    figures_dir: Path,
    model_comparison: pd.DataFrame,
    feature_counts: pd.DataFrame,
    coefficient_comparison: pd.DataFrame,
    best_oof: pd.DataFrame,
    best_name: str,
    correlation_pairs: pd.DataFrame,
) -> None:
    """Write all experiment figures and figure captions.

    Purpose:
        Produce publication/report-ready visual summaries for cross-validation
        metrics, retained feature counts, coefficient changes, selected-model
        out-of-fold curves, and the strongest feature correlations.

    Args:
        figures_dir: Directory where PNG files and captions should be written.
        model_comparison: Summary metrics by configuration and threshold.
        feature_counts: Feature-retention summary table.
        coefficient_comparison: Combined coefficient diagnostics.
        best_oof: Out-of-fold predictions for the selected configuration.
        best_name: Selected configuration name for curve labels.
        correlation_pairs: Pairwise correlation table sorted by magnitude.
    """

    _configure_style()
    primary = model_comparison[model_comparison["threshold"] == REFERENCE_THRESHOLD].copy()
    _bar_metric(primary, "f1", figures_dir / "cv_f1_comparison.png", "5-Fold CV F1 Comparison")
    _paired_auc(primary, figures_dir / "roc_pr_auc_comparison.png")
    _feature_count_plot(feature_counts, figures_dir / "retained_features_by_method.png")
    _coefficient_plot(coefficient_comparison, figures_dir / "baseline_vs_pruned_coefficients.png")
    _oof_curves(best_oof, best_name, figures_dir)
    _correlation_heatmap(correlation_pairs, figures_dir / "top_correlation_heatmap.png")
    pd.DataFrame(
        [
            {"figure": "cv_f1_comparison.png", "caption": "Mean 5-fold CV F1 for Fake at threshold 0.485."},
            {"figure": "roc_pr_auc_comparison.png", "caption": "Mean train-only CV ROC-AUC and PR-AUC by method."},
            {"figure": "retained_features_by_method.png", "caption": "Average number of features retained by method."},
            {"figure": "baseline_vs_pruned_coefficients.png", "caption": "Largest coefficients for baseline and pruned Logistic Regression."},
            {"figure": "best_oof_curves.png", "caption": "Out-of-fold ROC and precision-recall curves for the selected development configuration."},
            {"figure": "top_correlation_heatmap.png", "caption": "Readable heatmap for features appearing in the strongest correlated pairs."},
        ]
    ).to_csv(figures_dir / "captions.csv", index=False)


def write_report(
    results_dir: Path,
    train_path: Path,
    model_comparison: pd.DataFrame,
    feature_counts: pd.DataFrame,
    removed_frequency: pd.DataFrame,
    coefficient_comparison: pd.DataFrame,
    sparse_features: pd.DataFrame,
    best_config: dict[str, object],
    pruning_reports: dict[float, pd.DataFrame],
) -> Path:
    """Write the Markdown report for the follow-up experiment.

    Purpose:
        Convert generated result tables into a human-readable narrative that
        documents motivation, train-only method, results, feature reduction,
        sparse-model behavior, interpretability, and direct answers to the
        research questions.

    Args:
        results_dir: Directory where the report should be written.
        train_path: Path to the training data used by the experiment.
        model_comparison: Summary metrics by configuration and threshold.
        feature_counts: Feature-retention summary table.
        removed_frequency: Frequency table for pruned features.
        coefficient_comparison: Combined coefficient diagnostics.
        sparse_features: Sparse coefficient zero/nonzero table.
        best_config: Selected configuration metadata.
        pruning_reports: Full-training pruning tables keyed by threshold.

    Returns:
        Path to the generated Markdown report.
    """

    primary = model_comparison[model_comparison["threshold"] == REFERENCE_THRESHOLD].copy()
    baseline = primary[primary["configuration"] == "Baseline Logistic"].iloc[0]
    best = primary.sort_values(["f1_mean", "roc_auc_mean", "features_remaining_mean"], ascending=[False, False, True]).iloc[0]
    lines = [
        "# Correlation Pruning and Sparse Logistic Regression",
        "",
        "## Motivation",
        "",
        "Several structural features in the previous classification stage had correlations close to or equal to |r| = 1. This follow-up tests whether removing redundant features improves development performance, stability, simplicity, or coefficient interpretability.",
        "",
        "## Method",
        "",
        f"- Reused existing training data only: `{train_path}`.",
        "- The frozen `test.csv` was not loaded, predicted, inspected, or evaluated.",
        "- Used the same deterministic 5-fold topic x label stratification strategy and `RANDOM_STATE = 42`.",
        "- Fake (`labels = 1`) remains the positive class.",
        "- Correlation pruning is implemented as an sklearn transformer inside the pipeline, so each fold computes redundant features from its training partition only.",
        "- The controlled pruning experiments keep Logistic Regression fixed at `C=0.1`, `class_weight=balanced`.",
        "- L1 and Elastic Net Logistic Regression were evaluated separately without manual correlation pruning.",
        "",
        "## Results",
        "",
        _comparison_table(primary),
        "",
        "Baseline reproduction at threshold 0.485:",
        "",
        f"- F1 = {baseline.f1_mean:.3f} +/- {baseline.f1_std:.3f}",
        f"- ROC-AUC = {baseline.roc_auc_mean:.3f} +/- {baseline.roc_auc_std:.3f}",
        f"- PR-AUC = {baseline.pr_auc_mean:.3f} +/- {baseline.pr_auc_std:.3f}",
        f"- Precision = {baseline.precision_mean:.3f} +/- {baseline.precision_std:.3f}",
        f"- Recall = {baseline.recall_mean:.3f} +/- {baseline.recall_std:.3f}",
        "",
        "## Feature Reduction",
        "",
        _markdown_table(feature_counts),
        "",
        "Most consistently removed features:",
        "",
        _markdown_table(removed_frequency.head(25)),
        "",
        "Full-train pruning summaries for inspection:",
        "",
        f"- |corr| >= 0.98 removed {len(pruning_reports[0.98])} feature-pair decisions.",
        f"- |corr| >= 0.95 removed {len(pruning_reports[0.95])} feature-pair decisions.",
        "",
        "## Sparse Models",
        "",
        f"L1 best params: `{best_config['l1_best_params']}`",
        "",
        f"Elastic Net best params: `{best_config['elastic_net_best_params']}`",
        "",
        "Sparse selected/non-selected coefficients:",
        "",
        _markdown_table(sparse_features.groupby(["configuration", "status"], as_index=False).size()),
        "",
        "## Interpretability",
        "",
        "Coefficient files compare the baseline Logistic model and the best pruned Logistic variants. Positive coefficients indicate higher probability assigned to Fake. Correlation pruning primarily improves interpretability if it removes redundant near-duplicate columns while preserving similar CV metrics.",
        "",
        "Largest coefficient rows:",
        "",
        _markdown_table(coefficient_comparison.head(25)),
        "",
        "## Conclusions",
        "",
        conclusion_text(primary, best),
        "",
        "Questions:",
        "",
        f"1. Very-high correlation pruning improved predictive performance: {answer_improvement(primary, 'Corr prune 0.98')}",
        f"2. Correlation pruning reduced features without meaningful performance loss: {answer_simplicity(primary)}",
        f"3. Stability improved between folds: {answer_stability(primary)}",
        f"4. Coefficient interpretability improved: {answer_interpretability(feature_counts)}",
        f"5. L1 or Elastic Net outperformed ordinary L2 Logistic Regression: {answer_sparse(primary)}",
        f"6. Preferred development model: `{best['configuration']}` at threshold 0.485.",
    ]
    path = results_dir / "correlation_experiment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _classifier_params(params: dict[str, object]) -> dict[str, object]:
    """Strip sklearn pipeline prefixes from classifier parameters.

    Purpose:
        Convert search-grid keys such as `classifier__C` into keyword
        arguments accepted directly by `LogisticRegression`.

    Args:
        params: Pipeline-style parameter dictionary.

    Returns:
        Parameter dictionary without the `classifier__` prefix.
    """

    return {key.removeprefix("classifier__"): value for key, value in params.items()}


def _as_frame(X, columns: list[str] | None = None) -> pd.DataFrame:
    """Return feature data as a pandas dataframe.

    Purpose:
        Normalize sklearn inputs so the pruner can work with either dataframe
        or array-like data while preserving column names when available.

    Args:
        X: Input feature matrix.
        columns: Optional column names to use when `X` is not already a
            dataframe.

    Returns:
        Copy of `X` as a pandas dataframe.
    """

    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=columns)


def _pruning_table(corr: pd.DataFrame, threshold: float, features: list[str]) -> pd.DataFrame:
    """Build deterministic pairwise pruning decisions.

    Purpose:
        Find highly correlated feature pairs, process strongest pairs first,
        and record which feature is retained or removed without consulting
        validation performance.

    Args:
        corr: Absolute pairwise correlation matrix.
        threshold: Minimum absolute correlation for pruning consideration.
        features: Feature names in their original model-input order.

    Returns:
        Dataframe describing retained and removed feature decisions.
    """

    removed: set[str] = set()
    rows = []
    pairs = []
    for i, feature_a in enumerate(features):
        for feature_b in features[i + 1 :]:
            value = corr.loc[feature_a, feature_b]
            if pd.notna(value) and value >= threshold:
                pairs.append((feature_a, feature_b, float(value)))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    for feature_a, feature_b, value in pairs:
        if feature_a in removed or feature_b in removed:
            continue
        retained, dropped = _choose_representative(feature_a, feature_b)
        removed.add(dropped)
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "absolute_correlation": value,
                "retained_feature": retained,
                "removed_feature": dropped,
                "reason": "deterministic structural-priority rule; no validation performance used",
            }
        )
    return pd.DataFrame(rows)


def _choose_representative(feature_a: str, feature_b: str) -> tuple[str, str]:
    """Choose which feature from a correlated pair to keep.

    Purpose:
        Apply the shared deterministic feature-priority rule so pruning is
        reproducible and independent of model performance.

    Args:
        feature_a: First feature name.
        feature_b: Second feature name.

    Returns:
        Tuple of `(retained_feature, removed_feature)`.
    """

    ordered = sorted([feature_a, feature_b], key=_feature_priority)
    return ordered[0], ordered[1]


def _feature_priority(feature: str) -> tuple[int, int, str]:
    """Return the deterministic sort key used for pruning choices.

    Purpose:
        Prefer interpretable structural features, then volume features, then
        other features, and finally diagnostics when two highly correlated
        columns compete.

    Args:
        feature: Feature name to rank.

    Returns:
        Tuple used as a stable sort key for feature priority.
    """

    diagnostic_terms = ("parse", "has_cycle", "is_dag", "is_structurally_clean")
    structural_terms = ("virality", "depth", "breadth", "branching", "leaves_ratio", "root_ratio")
    volume_terms = ("n_nodes", "n_edges", "n_users", "n_tweets", "n_retweets", "n_replies")
    if any(term in feature for term in structural_terms):
        family = 0
    elif any(term in feature for term in volume_terms):
        family = 1
    elif any(term in feature for term in diagnostic_terms):
        family = 3
    else:
        family = 2
    return family, len(feature), feature


def _correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix into a sorted pair table.

    Purpose:
        Create an inspectable artifact of all feature-pair correlations and
        flag which pairs pass the 0.98 and 0.95 pruning thresholds.

    Args:
        corr: Pairwise correlation matrix.

    Returns:
        Long-form correlation-pair dataframe sorted by absolute correlation.
    """

    rows = []
    columns = list(corr.columns)
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            value = corr.loc[feature_a, feature_b]
            if pd.notna(value):
                rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                        "passes_098": abs(value) >= 0.98,
                        "passes_095": abs(value) >= 0.95,
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False)


def _configure_style() -> None:
    """Configure matplotlib defaults for experiment figures.

    Purpose:
        Keep all generated plots visually consistent across metrics,
        coefficient, curve, and heatmap figures.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _bar_metric(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    """Write a horizontal bar chart for one metric.

    Purpose:
        Visualize mean cross-validation performance and fold variability for
        each configuration.

    Args:
        df: Model-comparison rows for a single decision threshold.
        metric: Metric prefix to plot, such as `f1`.
        path: Output PNG path.
        title: Plot title.
    """

    data = df.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(data))
    ax.barh(y, data[f"{metric}_mean"], xerr=data[f"{metric}_std"], color=ACCENT_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(data["configuration"])
    ax.set_xlabel(f"{metric.upper()} mean +/- std")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _paired_auc(df: pd.DataFrame, path: Path) -> None:
    """Write the paired ROC-AUC and PR-AUC comparison chart.

    Purpose:
        Show ranking-quality and precision-recall-quality summaries side by
        side for each configuration.

    Args:
        df: Model-comparison rows for a single decision threshold.
        path: Output PNG path.
    """

    data = df.set_index("configuration")
    x = np.arange(len(data))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, data["roc_auc_mean"], width, label="ROC-AUC", color=REAL_COLOR)
    ax.bar(x + width / 2, data["pr_auc_mean"], width, label="PR-AUC", color=FAKE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=25, ha="right")
    ax.set_ylabel("Mean CV Score")
    ax.set_title("ROC-AUC and PR-AUC Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _feature_count_plot(df: pd.DataFrame, path: Path) -> None:
    """Write the retained-feature-count comparison chart.

    Purpose:
        Visualize how much each pruning or sparse configuration simplifies the
        feature set.

    Args:
        df: Feature-count summary table.
        path: Output PNG path.
    """

    data = df.sort_values("remaining_mean")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(data["configuration"], data["remaining_mean"], color=ACCENT_COLOR)
    ax.set_xlabel("Mean Retained Features")
    ax.set_title("Feature Count by Method")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _coefficient_plot(df: pd.DataFrame, path: Path, top_n: int = 18) -> None:
    """Write the baseline-versus-pruned coefficient comparison chart.

    Purpose:
        Compare the largest baseline coefficients with corresponding pruned
        model coefficients to inspect interpretability changes.

    Args:
        df: Combined coefficient diagnostics.
        path: Output PNG path.
        top_n: Number of largest baseline coefficients to include.
    """

    selected_configs = ["Baseline Logistic", "Corr prune 0.98", "Corr prune 0.95"]
    data = df[df["configuration"].isin(selected_configs)].copy()
    top_features = (
        data[data["configuration"] == "Baseline Logistic"]
        .sort_values("abs_coefficient", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    data = data[data["feature"].isin(top_features)]
    pivot = data.pivot_table(index="feature", columns="configuration", values="coefficient", aggfunc="first").fillna(0)
    pivot = pivot.reindex(top_features[::-1])
    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.34 * len(pivot))))
    y = np.arange(len(pivot))
    width = 0.25
    for offset, config in zip([-width, 0, width], selected_configs):
        if config in pivot.columns:
            ax.barh(y + offset, pivot[config], width, label=config)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Coefficient (positive = Fake)")
    ax.set_title("Baseline vs Pruned Logistic Coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _oof_curves(oof: pd.DataFrame, best_name: str, figures_dir: Path) -> None:
    """Write ROC and precision-recall curves for the selected model.

    Purpose:
        Visualize train-only out-of-fold ranking behavior for the chosen
        development configuration.

    Args:
        oof: Out-of-fold labels and positive-class probabilities.
        best_name: Selected configuration name for plot labels.
        figures_dir: Directory where the combined curve PNG should be written.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    fpr, tpr, _ = roc_curve(oof["labels"], oof["probability_fake"], pos_label=POSITIVE_LABEL)
    precision, recall, _ = precision_recall_curve(oof["labels"], oof["probability_fake"], pos_label=POSITIVE_LABEL)
    axes[0].plot(fpr, tpr, color=FAKE_COLOR, label=best_name)
    axes[0].plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle="--", label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Out-of-Fold ROC Curve")
    axes[0].legend()
    axes[1].plot(recall, precision, color=FAKE_COLOR, label=best_name)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Out-of-Fold Precision-Recall Curve")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "best_oof_curves.png")
    plt.close(fig)


def _correlation_heatmap(correlation_pairs: pd.DataFrame, path: Path, top_n: int = 14) -> None:
    """Write a readable heatmap for top correlated features.

    Purpose:
        Summarize the strongest pairwise correlations without rendering the
        full feature matrix, which would be too large for quick inspection.

    Args:
        correlation_pairs: Long-form pair table sorted by absolute correlation.
        path: Output PNG path.
        top_n: Number of top correlation pairs used to choose heatmap features.
    """

    features = sorted(
        set(correlation_pairs.head(top_n)["feature_a"]).union(set(correlation_pairs.head(top_n)["feature_b"]))
    )
    if not features:
        return
    matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    for row in correlation_pairs.itertuples(index=False):
        if row.feature_a in matrix.index and row.feature_b in matrix.columns:
            matrix.loc[row.feature_a, row.feature_b] = row.correlation
            matrix.loc[row.feature_b, row.feature_a] = row.correlation
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(features)), max(6.0, 0.45 * len(features))))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    ax.set_title("Top Correlated Feature Subset")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _comparison_table(df: pd.DataFrame) -> str:
    """Format model-comparison rows as a Markdown table.

    Purpose:
        Produce the compact results table embedded in the generated report.

    Args:
        df: Model-comparison rows for the reference threshold.

    Returns:
        Markdown table string.
    """

    rows = []
    for row in df.sort_values("f1_mean", ascending=False).itertuples(index=False):
        rows.append(
            {
                "Configuration": row.configuration,
                "Features": f"{row.features_remaining_mean:.1f}",
                "Precision": _pm(row.precision_mean, row.precision_std),
                "Recall": _pm(row.recall_mean, row.recall_std),
                "F1": _pm(row.f1_mean, row.f1_std),
                "ROC-AUC": _pm(row.roc_auc_mean, row.roc_auc_std),
                "PR-AUC": _pm(row.pr_auc_mean, row.pr_auc_std),
                "Delta F1": f"{row.delta_f1:+.4f}",
            }
        )
    return _markdown_table(pd.DataFrame(rows))


def _pm(mean: float, std: float) -> str:
    """Format a mean plus standard deviation string.

    Purpose:
        Keep report metric formatting consistent.

    Args:
        mean: Mean metric value.
        std: Standard deviation value.

    Returns:
        Formatted `mean +/- std` string.
    """

    return f"{mean:.3f} +/- {std:.3f}"


def _markdown_table(df: pd.DataFrame) -> str:
    """Convert a dataframe to a simple Markdown table.

    Purpose:
        Render report tables without adding a dependency on a tabulation
        library.

    Args:
        df: Dataframe to render.

    Returns:
        Markdown table string, or `_None._` for empty input.
    """

    if df.empty:
        return "_None._"
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    out = out.astype(str)
    columns = list(out.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |" for _, row in out.iterrows()]
    return "\n".join([header, separator, *rows])


def conclusion_text(primary: pd.DataFrame, best: pd.Series) -> str:
    """Write the report conclusion for the best configuration.

    Purpose:
        Interpret whether the selected model's F1 gain is meaningful relative
        to baseline fold-to-fold variability.

    Args:
        primary: Reference-threshold model-comparison rows.
        best: Best row selected from `primary`.

    Returns:
        Human-readable conclusion sentence.
    """

    baseline = primary[primary["configuration"] == "Baseline Logistic"].iloc[0]
    delta = best["f1_mean"] - baseline["f1_mean"]
    if abs(delta) < baseline["f1_std"]:
        return (
            f"The best configuration by mean F1 was `{best['configuration']}`, but its F1 difference "
            "is small relative to the baseline fold-to-fold variability. This should be interpreted conservatively."
        )
    if delta > 0:
        return f"`{best['configuration']}` improved mean F1 beyond the baseline variability and is the preferred development model."
    return "The baseline Logistic Regression remains preferable because pruning/sparse variants did not improve F1."


def answer_improvement(primary: pd.DataFrame, configuration: str) -> str:
    """Answer whether a named configuration clearly improved F1.

    Purpose:
        Support the report's direct research-question answers using the
        baseline F1 standard deviation as a conservative comparison scale.

    Args:
        primary: Reference-threshold model-comparison rows.
        configuration: Candidate configuration to compare with baseline.

    Returns:
        Short textual answer for the generated report.
    """

    baseline = primary[primary["configuration"] == "Baseline Logistic"].iloc[0]
    candidate = primary[primary["configuration"] == configuration].iloc[0]
    delta = candidate["f1_mean"] - baseline["f1_mean"]
    return "yes" if delta > baseline["f1_std"] else f"not clearly; delta F1 = {delta:+.4f}"


def answer_simplicity(primary: pd.DataFrame) -> str:
    """Answer whether pruning produced a simpler comparable model.

    Purpose:
        Determine whether any pruned configuration retained fewer features
        while staying within baseline F1 variability.

    Args:
        primary: Reference-threshold model-comparison rows.

    Returns:
        Short textual answer for the generated report.
    """

    baseline = primary[primary["configuration"] == "Baseline Logistic"].iloc[0]
    pruned = primary[primary["configuration"].str.startswith("Corr prune")]
    simpler = pruned[pruned["features_remaining_mean"] < baseline["features_remaining_mean"]]
    close = simpler[(simpler["f1_mean"] - baseline["f1_mean"]).abs() <= baseline["f1_std"]]
    return "yes" if not close.empty else "no clear simpler-equivalent model found"


def answer_stability(primary: pd.DataFrame) -> str:
    """Answer whether pruning improved F1 stability across folds.

    Purpose:
        Compare the best pruned F1 standard deviation against the baseline
        standard deviation.

    Args:
        primary: Reference-threshold model-comparison rows.

    Returns:
        Short textual answer for the generated report.
    """

    baseline_std = primary[primary["configuration"] == "Baseline Logistic"].iloc[0]["f1_std"]
    best_pruned_std = primary[primary["configuration"].str.startswith("Corr prune")]["f1_std"].min()
    return "yes" if best_pruned_std < baseline_std else "not for F1 standard deviation"


def answer_interpretability(feature_counts: pd.DataFrame) -> str:
    """Answer whether pruning improved coefficient interpretability.

    Purpose:
        Treat feature-count reduction as the experiment's operational signal
        that fewer redundant coefficients need to be interpreted.

    Args:
        feature_counts: Feature-count summary table.

    Returns:
        Short textual answer for the generated report.
    """

    pruned = feature_counts[feature_counts["configuration"].str.startswith("Corr prune")]
    if pruned.empty:
        return "not assessed"
    return "yes, fewer redundant coefficients need interpretation" if pruned["remaining_mean"].min() < pruned["original_features"].iloc[0] else "no feature reduction occurred"


def answer_sparse(primary: pd.DataFrame) -> str:
    """Answer whether sparse models clearly beat the baseline.

    Purpose:
        Compare the best sparse Logistic Regression F1 against the baseline
        using baseline fold variability as the conservative margin.

    Args:
        primary: Reference-threshold model-comparison rows.

    Returns:
        Short textual answer for the generated report.
    """

    baseline = primary[primary["configuration"] == "Baseline Logistic"].iloc[0]
    sparse = primary[primary["configuration"].isin(["L1 Logistic", "Elastic Net Logistic"])]
    if sparse.empty:
        return "not evaluated"
    best_sparse = sparse.sort_values("f1_mean", ascending=False).iloc[0]
    return "yes" if best_sparse["f1_mean"] > baseline["f1_mean"] + baseline["f1_std"] else "not clearly"
