"""Feature preparation helpers for leakage-safe classification experiments.

This module owns the classifier-facing feature contract: it derives non-model
topic and stratum labels, chooses numeric structural columns that may be passed
to models, defines reusable ablation feature groups, and converts selected
columns into numeric matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.analysis.feature_selection import METADATA_COLUMNS
from src.classification.constants import TOPIC_GROUPS


EXTRA_LEAKAGE_COLUMNS = {
    "label",
    "topic",
    "category",
    "stratum",
    "split",
}

ID_MARKERS = ("_id", "_ids")


@dataclass(frozen=True)
class ModelFeatureSelection:
    """Classifier input features and excluded columns.

    Purpose:
        Store the final list of columns that are safe for model training and a
        diagnostic explanation for each column that was excluded.

    Attributes:
        features: Sorted model input column names.
        excluded_columns: Mapping from excluded column name to the exclusion
            reason.
    """

    features: list[str]
    excluded_columns: dict[str, str]


def add_topic_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the feature table with a derived non-model topic column.

    Purpose:
        Map each dataset name to its broader topic group so train/test splitting
        and cross-validation can preserve topic balance without using topic as a
        model feature.

    Args:
        df: Feature table containing a `data_name` column.

    Returns:
        A copy of `df` with a derived `topic` column.

    Raises:
        ValueError: If `data_name` is missing or contains values that are not
            listed in `TOPIC_GROUPS`.
    """

    if "data_name" not in df.columns:
        raise ValueError("Expected a 'data_name' column to derive topic groups")

    mapping = {
        data_name: topic
        for topic, data_names in TOPIC_GROUPS.items()
        for data_name in data_names
    }
    out = df.copy()
    out["topic"] = out["data_name"].map(mapping)
    missing = sorted(out.loc[out["topic"].isna(), "data_name"].dropna().unique())
    if missing:
        raise ValueError(f"Missing topic mapping for data_name values: {missing}")
    return out


def stratum_labels(df: pd.DataFrame) -> pd.Series:
    """Return joint topic-by-label labels used for stratified split and CV.

    Purpose:
        Build stable strata that preserve both topical groups and real/fake
        label balance when partitioning classification data.

    Args:
        df: Feature table containing `labels` and either `topic` or
            `data_name`.

    Returns:
        A string series in the form `<topic>__<label>`.

    Raises:
        ValueError: If `topic` must be derived and `data_name` is missing or
            unmapped.
    """

    if "topic" not in df.columns:
        df = add_topic_column(df)
    labels = pd.to_numeric(df["labels"], errors="raise").astype(int).astype(str)
    return df["topic"].astype(str) + "__" + labels


def select_model_features(df: pd.DataFrame) -> ModelFeatureSelection:
    """Select numeric and boolean structural columns for classifier input.

    Purpose:
        Keep only model-safe structural measurements while recording why labels,
        metadata, topic fields, IDs, graph diagnostics, and non-numeric columns
        were excluded.

    Args:
        df: Full feature table before model feature filtering.

    Returns:
        A `ModelFeatureSelection` with sorted feature names and sorted exclusion
        diagnostics.

    Raises:
        ValueError: If a forbidden leakage column survives feature selection.
    """

    excluded: dict[str, str] = {}
    features: list[str] = []
    blocked = set(METADATA_COLUMNS) | EXTRA_LEAKAGE_COLUMNS

    for column in df.columns:
        if column in blocked:
            excluded[column] = "label, topic, data_name, metadata, raw text, raw IDs, or traceability field"
            continue
        if column.endswith("_graph_status"):
            excluded[column] = "categorical graph diagnostic, excluded from numeric classifier input"
            continue
        lowered = column.lower()
        if lowered == "id" or lowered.endswith(ID_MARKERS):
            excluded[column] = "identifier column"
            continue

        series = df[column]
        if is_bool_dtype(series):
            features.append(column)
            continue
        if is_numeric_dtype(series):
            features.append(column)
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            features.append(column)
        else:
            excluded[column] = "non-numeric column"

    features = sorted(features)
    validate_no_leakage_features(features)
    return ModelFeatureSelection(features=features, excluded_columns=dict(sorted(excluded.items())))


def validate_no_leakage_features(features: list[str]) -> None:
    """Raise if a forbidden metadata, label, topic, or ID column is selected.

    Purpose:
        Act as a final guardrail against target leakage before a feature list is
        used by model training or evaluation code.

    Args:
        features: Candidate model input column names.

    Raises:
        ValueError: If `features` includes a known metadata, label, topic, or ID
            field.
    """

    forbidden_exact = set(METADATA_COLUMNS) | EXTRA_LEAKAGE_COLUMNS | {"labels", "data_name", "news_id"}
    leaks = [feature for feature in features if feature in forbidden_exact]
    leaks.extend(feature for feature in features if feature.lower() == "id" or feature.lower().endswith(ID_MARKERS))
    if leaks:
        raise ValueError(f"Leakage features selected for model input: {sorted(set(leaks))}")


def feature_sets(all_features: list[str]) -> dict[str, list[str]]:
    """Build scientifically useful structural ablation feature sets.

    Purpose:
        Group the available classifier features into interpretable subsets so
        experiments can compare all structural features against retweet-only,
        reply-only, basic size/depth/breadth, size-free, and shape/virality
        views.

    Args:
        all_features: Complete list of selected model feature names.

    Returns:
        A mapping from feature-set name to sorted feature names. Empty sets and
        sets containing unavailable features are omitted.
    """

    all_set = set(all_features)
    size_terms = (
        "n_tweets",
        "n_retweets",
        "n_replies",
        "n_users",
        "_n_nodes",
        "_n_edges",
        "_n_users",
        "largest_component_nodes",
        "largest_component_edges",
        "mean_component_nodes",
        "median_component_nodes",
        "std_component_nodes",
    )
    shape_terms = (
        "depth",
        "breadth",
        "branching",
        "root",
        "leaves",
        "virality",
        "component_node_fraction",
        "repeat_user_ratio",
        "max_out_degree",
    )
    basic_terms = ("n_tweets", "n_retweets", "n_replies", "n_users", "n_nodes", "n_edges", "max_depth", "max_breadth")

    sets = {
        "all_structural": sorted(all_features),
        "retweet_only": sorted(feature for feature in all_features if feature.startswith("rt_")),
        "reply_only": sorted(feature for feature in all_features if feature.startswith("reply_")),
        "retweet_reply": sorted(feature for feature in all_features if feature.startswith(("rt_", "reply_"))),
        "basic_size_depth_breadth": sorted(
            feature
            for feature in all_features
            if feature in {"n_tweets", "n_retweets", "n_replies", "n_users"}
            or any(term in feature for term in basic_terms)
        ),
        "without_size_volume": sorted(
            feature for feature in all_features if not any(term in feature for term in size_terms)
        ),
        "shape_virality_only": sorted(feature for feature in all_features if any(term in feature for term in shape_terms)),
    }
    return {name: features for name, features in sets.items() if features and set(features) <= all_set}


def make_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a numeric matrix for model input.

    Purpose:
        Convert the selected classifier columns into numeric values expected by
        scikit-learn pipelines while preserving missing values for downstream
        imputation.

    Args:
        df: Feature table containing every column listed in `features`.
        features: Ordered model input columns to extract.

    Returns:
        A copy of the selected columns with booleans encoded as integers and
        other values coerced with `pandas.to_numeric`.
    """

    out = df[features].copy()
    for column in features:
        if is_bool_dtype(out[column]):
            out[column] = out[column].astype(int)
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
