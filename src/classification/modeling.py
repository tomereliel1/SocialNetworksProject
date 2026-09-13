"""Train-only model selection and diagnostics for MC-Fake classification.

This file is responsible for the complete leakage-safe training experiment over
`train.csv`: selecting model-safe structural features, running cross-validated
hyperparameter searches, comparing candidate classifiers, tuning the decision
threshold from out-of-fold predictions, fitting the final model, and writing the
CSV/JSON/joblib artifacts that document the experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classification.constants import N_SPLITS, POSITIVE_LABEL, RANDOM_STATE
from src.classification.features import feature_sets, make_numeric_frame, select_model_features, stratum_labels


SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Artifacts produced by the train-only classification experiment.

    Purpose:
        Bundle every in-memory table, configuration dictionary, fitted-model
        diagnostic, and curve output that `run_train_only_experiment` also
        writes to disk.

    Attributes:
        model_comparison: Cross-validated summary metrics for each candidate
            model's best hyperparameter configuration.
        cv_results: Candidate-level cross-validation results from the sklearn
            searches.
        ablation_results: Cross-validated metrics for interpretable feature
            subsets.
        selected_features: Columns selected for classifier input.
        excluded_features: Columns excluded from classifier input and their
            reasons.
        feature_correlations: Spearman correlation matrix for selected
            features.
        high_correlations: Feature pairs with large absolute Spearman
            correlations.
        individual_feature_scores: Univariate feature ranking diagnostics.
        best_model_config: JSON-safe configuration for the selected model.
        oof_predictions: Out-of-fold probabilities and thresholded labels.
        train_diagnostics: Training-set resubstitution metrics for diagnostic
            inspection only.
        confusion: Out-of-fold confusion matrix ordered as real/fake by
            predicted real/fake.
        roc_curve: Out-of-fold ROC curve points.
        pr_curve: Out-of-fold precision-recall curve points.
        feature_importance: Model-native and permutation feature importance
            diagnostics.
    """

    model_comparison: pd.DataFrame
    cv_results: pd.DataFrame
    ablation_results: pd.DataFrame
    selected_features: pd.DataFrame
    excluded_features: pd.DataFrame
    feature_correlations: pd.DataFrame
    high_correlations: pd.DataFrame
    individual_feature_scores: pd.DataFrame
    best_model_config: dict[str, object]
    oof_predictions: pd.DataFrame
    train_diagnostics: pd.DataFrame
    confusion: np.ndarray
    roc_curve: pd.DataFrame
    pr_curve: pd.DataFrame
    feature_importance: pd.DataFrame


def run_train_only_experiment(train_path: Path, results_dir: Path) -> ExperimentArtifacts:
    """Run leakage-safe classification experiments using train.csv only.

    Purpose:
        Own the main modeling workflow: load the frozen training split, select
        safe structural features, evaluate model families by cross-validation,
        run feature ablations, tune a probability threshold from out-of-fold
        predictions, fit the final model, and write all reproducibility
        artifacts.

    Args:
        train_path: Path to the frozen training CSV.
        results_dir: Directory where result tables, configuration files, and the
            fitted model should be written.

    Returns:
        An `ExperimentArtifacts` instance containing the generated tables,
        configuration, curves, confusion matrix, and feature importance data.

    Raises:
        ValueError: If the input data cannot be converted to the expected
            numeric label or feature schema.
    """

    df = pd.read_csv(train_path, low_memory=False)
    selection = select_model_features(df)
    X = make_numeric_frame(df, selection.features)
    y = pd.to_numeric(df["labels"], errors="raise").astype(int)
    cv = make_cv_splits(df)

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected_features = pd.DataFrame({"feature": selection.features, "selected_for": "all_structural"})
    excluded_features = pd.DataFrame(
        [{"column": column, "reason": reason} for column, reason in selection.excluded_columns.items()]
    )
    selected_features.to_csv(results_dir / "selected_features.csv", index=False)
    excluded_features.to_csv(results_dir / "excluded_features.csv", index=False)

    feature_correlations = X.corr(method="spearman")
    feature_correlations.to_csv(results_dir / "feature_correlations_spearman.csv")
    high_correlations = _high_correlations(feature_correlations)
    high_correlations.to_csv(results_dir / "highly_correlated_features.csv", index=False)

    individual_scores = individual_feature_scores(X, y)
    individual_scores.to_csv(results_dir / "individual_feature_scores.csv", index=False)

    estimators = model_searches()
    comparison_rows: list[dict[str, object]] = []
    cv_result_rows: list[pd.DataFrame] = []
    best_estimators: dict[str, BaseEstimator] = {}

    for name, search in estimators.items():
        fit_search_with_cv(search, X, y, cv)
        best_estimators[name] = search.best_estimator_
        comparison_rows.append(_comparison_row(name, search))
        cv_result_rows.append(_search_results_frame(name, search))

    model_comparison = pd.DataFrame(comparison_rows).sort_values(["f1_mean", "balanced_accuracy_mean"], ascending=False)
    model_comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    cv_results = pd.concat(cv_result_rows, ignore_index=True)
    cv_results.to_csv(results_dir / "cv_results.csv", index=False)

    best_name = str(model_comparison.iloc[0]["model"])
    selected_best_params = model_comparison.iloc[0]["best_params"]
    best_estimator = best_estimators[best_name]
    ablation_results = run_ablation_experiments(X, y, cv, feature_sets(selection.features))
    ablation_results.to_csv(results_dir / "feature_ablation_results.csv", index=False)

    best_features = selection.features
    oof = cross_val_predict(best_estimator, X[best_features], y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold, threshold_f1 = tune_threshold(y, oof)
    oof_labels = (oof >= threshold).astype(int)
    oof_predictions = pd.DataFrame(
        {
            "news_id": df["news_id"],
            "labels": y,
            "oof_probability_fake": oof,
            "oof_prediction": oof_labels,
        }
    )
    oof_predictions.to_csv(results_dir / "oof_predictions_train_only.csv", index=False)

    train_diagnostics = training_diagnostics(best_estimator, X[best_features], y, threshold)
    train_diagnostics.to_csv(results_dir / "training_diagnostics.csv", index=False)

    best_estimator.fit(X[best_features], y)
    joblib.dump(best_estimator, results_dir / "best_model.joblib")

    importance = feature_importance_table(best_estimator, X[best_features], y, best_features)
    if best_name != "Logistic Regression" and "Logistic Regression" in best_estimators:
        importance = pd.concat(
            [
                importance,
                logistic_coefficient_table(best_estimators["Logistic Regression"], best_features),
            ],
            ignore_index=True,
        )
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    config = {
        "selected_model": best_name,
        "positive_class": "Fake (labels=1)",
        "primary_metric": "f1",
        "selected_features": best_features,
        "excluded_columns": selection.excluded_columns,
        "best_params": selected_best_params,
        "default_threshold": 0.5,
        "selected_threshold": threshold,
        "selected_threshold_cv_f1": threshold_f1,
        "data_used": str(train_path),
        "test_set_policy": "test.csv was not loaded or evaluated by this experiment command",
    }
    serializable_config = _json_safe(config)
    (results_dir / "best_model_config.json").write_text(
        json.dumps(serializable_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    roc = _roc_curve_frame(y, oof)
    pr = _pr_curve_frame(y, oof)
    roc.to_csv(results_dir / "roc_curve_oof.csv", index=False)
    pr.to_csv(results_dir / "precision_recall_curve_oof.csv", index=False)
    conf = confusion_matrix(y, oof_labels, labels=[0, 1])
    pd.DataFrame(conf, index=["actual_real", "actual_fake"], columns=["pred_real", "pred_fake"]).to_csv(
        results_dir / "confusion_matrix_oof.csv"
    )

    return ExperimentArtifacts(
        model_comparison=model_comparison,
        cv_results=cv_results,
        ablation_results=ablation_results,
        selected_features=selected_features,
        excluded_features=excluded_features,
        feature_correlations=feature_correlations,
        high_correlations=high_correlations,
        individual_feature_scores=individual_scores,
        best_model_config=serializable_config,
        oof_predictions=oof_predictions,
        train_diagnostics=train_diagnostics,
        confusion=conf,
        roc_curve=roc,
        pr_curve=pr,
        feature_importance=importance,
    )


def make_cv_splits(df: pd.DataFrame, n_splits: int = N_SPLITS):
    """Create leakage-aware stratified cross-validation splits.

    Purpose:
        Preserve topic-label balance across folds and keep repeated `news_id`
        groups in a single fold when duplicate IDs are present.

    Args:
        df: Training dataframe containing `labels`, `news_id`, and either
            `topic` or `data_name`.
        n_splits: Number of cross-validation folds to create.

    Returns:
        A list of `(train_index, validation_index)` tuples suitable for sklearn
        `cv` arguments.
    """

    strata = stratum_labels(df)
    if df["news_id"].duplicated().any():
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(df, strata, df["news_id"].astype(str)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(df, strata))


def model_searches() -> dict[str, GridSearchCV | RandomizedSearchCV]:
    """Build the candidate model search objects.

    Purpose:
        Define the classifier families, preprocessing pipelines, hyperparameter
        grids, scoring metrics, and F1 refit policy used in the main experiment.
        The real cross-validation splitter is attached later by
        `fit_search_with_cv`.

    Returns:
        A mapping from display model name to configured sklearn search object.
    """

    cv_placeholder = None
    return {
        "Dummy most_frequent": GridSearchCV(
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", DummyClassifier())]),
            {"classifier__strategy": ["most_frequent"]},
            scoring=SCORING,
            refit="f1",
            cv=cv_placeholder,
            n_jobs=-1,
        ),
        "Dummy stratified": GridSearchCV(
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", DummyClassifier(random_state=RANDOM_STATE))]),
            {"classifier__strategy": ["stratified"]},
            scoring=SCORING,
            refit="f1",
            cv=cv_placeholder,
            n_jobs=-1,
        ),
        "Logistic Regression": GridSearchCV(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
            scoring=SCORING,
            refit="f1",
            cv=cv_placeholder,
            n_jobs=-1,
        ),
        "Random Forest": RandomizedSearchCV(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=180,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [80, 120],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 8],
                "classifier__min_samples_leaf": [1, 4],
                "classifier__max_features": ["sqrt", 0.5],
                "classifier__class_weight": [None, "balanced"],
            },
            n_iter=5,
            scoring=SCORING,
            refit="f1",
            cv=cv_placeholder,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": RandomizedSearchCV(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_iter": [100, 160],
                "classifier__max_leaf_nodes": [15, 31],
                "classifier__l2_regularization": [0.0, 0.01, 0.1],
            },
            n_iter=5,
            scoring=SCORING,
            refit="f1",
            cv=cv_placeholder,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }


def fit_search_with_cv(search: GridSearchCV | RandomizedSearchCV, X: pd.DataFrame, y: pd.Series, cv):
    """Fit one sklearn search object with externally prepared CV splits.

    Purpose:
        Attach the leakage-aware split list produced by `make_cv_splits` to a
        search object and run the hyperparameter search.

    Args:
        search: Grid or randomized search object from `model_searches`.
        X: Numeric model feature matrix.
        y: Binary label series where the positive class is fake news.
        cv: Cross-validation splits accepted by sklearn search estimators.

    Returns:
        The fitted search object.
    """

    search.cv = cv
    return search.fit(X, y)


def individual_feature_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank individual feature usefulness on train.csv without test-set access.

    Purpose:
        Produce lightweight univariate diagnostics that show how informative
        each selected structural feature is by itself, without influencing model
        selection with held-out test data.

    Args:
        X: Numeric model feature matrix.
        y: Binary label series where the positive class is fake news.

    Returns:
        A dataframe sorted by best-direction ROC AUC and absolute Spearman
        correlation with the label.
    """

    rows = []
    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce")
        fill_value = values.median()
        values = values.fillna(0 if pd.isna(fill_value) else fill_value)
        if values.nunique(dropna=True) <= 1:
            auc = 0.5
            direction = "constant"
        else:
            auc_raw = roc_auc_score(y, values)
            auc = max(auc_raw, 1 - auc_raw)
            direction = "higher_fake" if auc_raw >= 0.5 else "lower_fake"
        rows.append(
            {
                "feature": feature,
                "univariate_roc_auc_best_direction": auc,
                "direction": direction,
                "absolute_spearman_with_label": abs(values.corr(y, method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["univariate_roc_auc_best_direction", "absolute_spearman_with_label"],
        ascending=False,
    )


def run_ablation_experiments(X: pd.DataFrame, y: pd.Series, cv, sets: dict[str, list[str]]) -> pd.DataFrame:
    """Evaluate interpretable feature subsets with a fixed logistic model.

    Purpose:
        Measure how much predictive signal comes from each structural feature
        group, such as retweet-only, reply-only, size-free, or shape/virality
        features.

    Args:
        X: Numeric model feature matrix.
        y: Binary label series where the positive class is fake news.
        cv: Cross-validation splits accepted by `cross_val_predict`.
        sets: Mapping from feature-set name to columns in `X`.

    Returns:
        A dataframe of cross-validated metrics sorted by F1 and balanced
        accuracy.
    """

    rows = []
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
        ]
    )
    for name, features in sets.items():
        probabilities = cross_val_predict(estimator, X[features], y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        preds = (probabilities >= 0.5).astype(int)
        rows.append({"feature_set": name, "n_features": len(features), **metric_dict(y, preds, probabilities)})
    return pd.DataFrame(rows).sort_values(["f1", "balanced_accuracy"], ascending=False)


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """Select the probability threshold with the best out-of-fold F1 score.

    Purpose:
        Convert calibrated or uncalibrated fake-news probabilities into binary
        labels using a train-only threshold sweep, preferring thresholds closest
        to 0.5 when F1 scores tie.

    Args:
        y_true: True binary labels.
        probabilities: Positive-class out-of-fold probabilities.

    Returns:
        The selected threshold and its F1 score.
    """

    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [(threshold, f1_score(y_true, probabilities >= threshold, zero_division=0)) for threshold in thresholds]
    best_threshold, best_score = max(scores, key=lambda item: (item[1], -abs(item[0] - 0.5)))
    return float(best_threshold), float(best_score)


def training_diagnostics(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, threshold: float) -> pd.DataFrame:
    """Compute training resubstitution diagnostics for the selected estimator.

    Purpose:
        Show how the fitted model behaves on the same data used for fitting at
        both the default 0.5 threshold and the selected threshold. These rows are
        diagnostics only and are not used as validation performance.

    Args:
        estimator: Selected sklearn estimator or pipeline.
        X: Numeric model feature matrix.
        y: Binary label series.
        threshold: Train-only threshold selected from out-of-fold predictions.

    Returns:
        A two-row metric table for default-threshold and tuned-threshold
        training diagnostics.
    """

    fitted = clone(estimator).fit(X, y)
    probabilities = fitted.predict_proba(X)[:, 1]
    default_preds = (probabilities >= 0.5).astype(int)
    tuned_preds = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        [
            {"evaluation": "training_resubstitution_default_threshold", **metric_dict(y, default_preds, probabilities)},
            {"evaluation": "training_resubstitution_selected_threshold", **metric_dict(y, tuned_preds, probabilities)},
        ]
    )


def feature_importance_table(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.DataFrame:
    """Build feature-importance diagnostics for a fitted estimator.

    Purpose:
        Collect model-native importances when available and add training
        permutation importance so different model families can be inspected in a
        shared output table.

    Args:
        estimator: Fitted sklearn estimator or pipeline.
        X: Numeric model feature matrix.
        y: Binary label series.
        features: Feature names in the same order as columns in `X`.

    Returns:
        A dataframe with feature names, importance type, importance value, and
        optional standard deviation for permutation importance.
    """

    classifier = estimator.named_steps["classifier"] if isinstance(estimator, Pipeline) else estimator
    rows = []
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        for feature, coef in zip(features, coefficients):
            rows.append({"feature": feature, "importance_type": "logistic_coefficient", "importance": float(coef)})
    elif hasattr(classifier, "feature_importances_"):
        for feature, value in zip(features, classifier.feature_importances_):
            rows.append({"feature": feature, "importance_type": "tree_importance", "importance": float(value)})

    perm = permutation_importance(
        estimator,
        X,
        y,
        scoring="f1",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    for feature, mean, std in zip(features, perm.importances_mean, perm.importances_std):
        rows.append(
            {
                "feature": feature,
                "importance_type": "training_permutation_f1",
                "importance": float(mean),
                "importance_std": float(std),
            }
        )
    return pd.DataFrame(rows).sort_values(["importance_type", "importance"], ascending=[True, False])


def logistic_coefficient_table(estimator: BaseEstimator, features: list[str]) -> pd.DataFrame:
    """Return logistic-regression coefficients in the shared importance schema.

    Purpose:
        Preserve an interpretable coefficient table for logistic regression even
        when the selected best model is a non-linear estimator.

    Args:
        estimator: Fitted sklearn estimator or pipeline.
        features: Feature names in the same order used to fit the estimator.

    Returns:
        A dataframe of coefficients, or an empty dataframe when the estimator
        does not expose `coef_`.
    """

    classifier = estimator.named_steps["classifier"] if isinstance(estimator, Pipeline) else estimator
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"feature": feature, "importance_type": "logistic_coefficient", "importance": float(coef)}
            for feature, coef in zip(features, classifier.coef_[0])
        ]
    )


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute the shared binary-classification metric set.

    Purpose:
        Keep model comparison, ablation, and diagnostic outputs on the same
        metric schema.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_score: Positive-class probabilities or decision scores.

    Returns:
        A dictionary containing accuracy, balanced accuracy, precision, recall,
        F1, ROC AUC, and PR AUC.
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


def _comparison_row(name: str, search: GridSearchCV | RandomizedSearchCV) -> dict[str, object]:
    """Summarize the best candidate from one fitted sklearn search.

    Purpose:
        Convert sklearn's verbose `cv_results_` arrays into one model-level row
        for the experiment comparison table.

    Args:
        name: Display name of the searched model family.
        search: Fitted grid or randomized search object.

    Returns:
        A dictionary containing the best parameters and mean/std metrics for the
        search's selected candidate.
    """

    row: dict[str, object] = {"model": name, "best_params": search.best_params_}
    idx = search.best_index_
    for metric in SCORING:
        row[f"{metric}_mean"] = search.cv_results_[f"mean_test_{metric}"][idx]
        row[f"{metric}_std"] = search.cv_results_[f"std_test_{metric}"][idx]
    return row


def _search_results_frame(name: str, search: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """Format candidate-level sklearn search results for CSV output.

    Purpose:
        Retain the metrics and parameters needed to audit every evaluated
        hyperparameter candidate without exposing the full sklearn result
        object.

    Args:
        name: Display name of the searched model family.
        search: Fitted grid or randomized search object.

    Returns:
        A sorted dataframe of candidate ranks, metrics, and parameters.
    """

    frame = pd.DataFrame(search.cv_results_)
    keep = ["rank_test_f1", "mean_test_f1", "std_test_f1", "mean_test_balanced_accuracy", "std_test_balanced_accuracy", "params"]
    for metric in ["accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        keep.extend([f"mean_test_{metric}", f"std_test_{metric}"])
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    frame.insert(0, "model", name)
    return frame.sort_values(["rank_test_f1", "model"])


def _high_correlations(correlation: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Return highly correlated non-duplicate feature pairs.

    Purpose:
        Identify selected structural features that may be redundant or
        scientifically similar before model interpretation.

    Args:
        correlation: Square feature correlation matrix.
        threshold: Minimum absolute correlation to include.

    Returns:
        A dataframe sorted by absolute correlation, or an empty dataframe when
        no pair meets the threshold.
    """

    rows = []
    columns = list(correlation.columns)
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            value = correlation.loc[feature_a, feature_b]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False) if rows else pd.DataFrame()


def _roc_curve_frame(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Convert out-of-fold probabilities into ROC curve points.

    Purpose:
        Store the ROC curve in a reproducible CSV-friendly format for plotting
        and reporting.

    Args:
        y_true: True binary labels.
        probabilities: Positive-class out-of-fold probabilities.

    Returns:
        A dataframe containing false-positive rate, true-positive rate, and
        threshold columns.
    """

    fpr, tpr, thresholds = roc_curve(y_true, probabilities, pos_label=POSITIVE_LABEL)
    return pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds})


def _pr_curve_frame(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Convert out-of-fold probabilities into precision-recall curve points.

    Purpose:
        Store the precision-recall curve in a reproducible CSV-friendly format
        for plotting and reporting.

    Args:
        y_true: True binary labels.
        probabilities: Positive-class out-of-fold probabilities.

    Returns:
        A dataframe containing precision, recall, and threshold columns.
    """

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities, pos_label=POSITIVE_LABEL)
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": threshold_values})


def _json_safe(value):
    """Convert nested experiment configuration values to JSON-safe objects.

    Purpose:
        Normalize numpy scalars, arrays, tuples, and other non-JSON-native
        values before writing experiment configuration files.

    Args:
        value: Arbitrary nested value from an experiment configuration.

    Returns:
        A JSON-serializable equivalent of `value`.
    """

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
