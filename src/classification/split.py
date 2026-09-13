"""Create, validate, and audit the frozen classification train/test split.

This file is responsible for turning the full engineered feature table into the
project's stable train and held-out test CSVs. It preserves label/topic balance,
keeps repeated `news_id` groups out of both splits at the same time, validates
that the split is leakage-safe, and writes audit artifacts describing the final
partition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.classification.constants import RANDOM_STATE, TEST_SIZE
from src.classification.features import add_topic_column, stratum_labels


def create_frozen_split(
    input_path: Path,
    train_path: Path,
    test_path: Path,
    audit_dir: Path,
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict[str, Path | str]:
    """Create and audit the frozen train/test split without evaluating test data.

    Purpose:
        Own the complete split workflow: load the feature table, derive topics,
        choose a row-level or group-aware split strategy, validate the partition,
        write train/test CSVs, and save audit files that document the split.

    Args:
        input_path: CSV path for the complete feature table.
        train_path: Output CSV path for training rows.
        test_path: Output CSV path for held-out test rows.
        audit_dir: Directory where split audit files should be written.
        random_state: Seed used for reproducible split shuffling.
        test_size: Fraction of rows to reserve for test when all `news_id`
            values are unique.

    Returns:
        Paths to the generated train, test, and audit artifacts plus a text
        description of the grouping strategy that was used.

    Raises:
        ValueError: If the input schema is invalid, the split leaks `news_id`
            groups across train/test, or the final topic-label proportions drift
            too far from the full dataset.
    """

    df = pd.read_csv(input_path, low_memory=False)
    df = add_topic_column(df)
    validate_minimum_schema(df)
    duplicate_news_ids = int(df["news_id"].duplicated().sum())
    group_column = "news_id"

    if duplicate_news_ids == 0:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratum_labels(df),
        )
        grouping_strategy = "Rows are independent event rows: every news_id is unique. Stratified row split was used."
    else:
        train_df, test_df = _group_aware_split(df, group_column, random_state)
        grouping_strategy = (
            "Repeated news_id values were found. StratifiedGroupKFold was used so each news_id stays in one split."
        )

    train_df = train_df.sort_index()
    test_df = test_df.sort_index()
    verify_split(df, train_df, test_df, group_column)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    split_summary = split_audit_table(df, train_df, test_df)
    split_summary_path = audit_dir / "split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)
    group_report_path = audit_dir / "grouping_and_split_checks.txt"
    group_report_path.write_text(
        "\n".join(
            [
                f"group_column={group_column}",
                f"duplicate_news_id_rows={duplicate_news_ids}",
                f"grouping_strategy={grouping_strategy}",
                f"full_rows={len(df)}",
                f"train_rows={len(train_df)}",
                f"test_rows={len(test_df)}",
                "train_test_news_id_overlap=0",
                "test_set_status=frozen_for_future_final_evaluation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "train_path": train_path,
        "test_path": test_path,
        "split_summary": split_summary_path,
        "group_report": group_report_path,
        "grouping_strategy": grouping_strategy,
    }


def validate_minimum_schema(df: pd.DataFrame) -> None:
    """Validate the minimum columns and label values required for splitting.

    Purpose:
        Fail early when the feature table cannot support a leakage-safe,
        stratified classification split.

    Args:
        df: Feature table after topic derivation.

    Raises:
        ValueError: If required columns are missing or `labels` contains values
            outside the expected binary 0/1 label space.
    """

    required = {"news_id", "labels", "data_name", "topic"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")
    labels = set(pd.to_numeric(df["labels"], errors="raise").astype(int).unique())
    if labels - {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, found: {sorted(labels)}")


def verify_split(full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, group_column: str) -> None:
    """Verify that train and test form a complete, leakage-safe partition.

    Purpose:
        Enforce the core split invariants before artifacts are written: every
        source row appears exactly once, no `news_id` or configured group appears
        in both splits, and topic-label balance stays close to the full table.

    Args:
        full_df: Complete source feature table.
        train_df: Candidate training split.
        test_df: Candidate held-out test split.
        group_column: Column whose values must not overlap across splits.

    Raises:
        ValueError: If row counts do not add up, train/test IDs or groups
            overlap, or topic-label proportions drift beyond the allowed limit.
    """

    if len(train_df) + len(test_df) != len(full_df):
        raise ValueError("Train and test row counts do not sum to the full dataset")
    train_ids = set(train_df["news_id"].astype(str))
    test_ids = set(test_df["news_id"].astype(str))
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train/test news_id overlap detected: {len(overlap)} IDs")
    train_groups = set(train_df[group_column].astype(str))
    test_groups = set(test_df[group_column].astype(str))
    group_overlap = train_groups & test_groups
    if group_overlap:
        raise ValueError(f"Train/test group overlap detected: {len(group_overlap)} groups")
    _check_proportion_drift(full_df, train_df, test_df, "topic_label")


def split_audit_table(full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form audit table for full, train, and test distributions.

    Purpose:
        Summarize sample counts, label balance, topic balance, and joint
        topic-label balance so the frozen split can be inspected and reported.

    Args:
        full_df: Complete source feature table.
        train_df: Final training split.
        test_df: Final held-out test split.

    Returns:
        A dataframe with one row per split/category/value containing counts,
        proportions, full-dataset reference proportions, differences, and drift
        warnings.
    """

    frames = {"full": full_df.copy(), "train": train_df.copy(), "test": test_df.copy()}
    for frame in frames.values():
        frame["topic_label"] = stratum_labels(frame)
    rows: list[dict[str, object]] = []
    full_props = _distribution_props(frames["full"])

    for split_name, frame in frames.items():
        rows.extend(_summary_rows(split_name, frame, full_props))

    return pd.DataFrame(rows)


def _summary_rows(split_name: str, df: pd.DataFrame, full_props: dict[tuple[str, str], float]) -> list[dict[str, object]]:
    """Return audit rows for one split dataframe.

    Purpose:
        Convert one split's sample, label, topic, and topic-label distributions
        into the shared row schema used by the split audit CSV.

    Args:
        split_name: Name to store in the audit table, such as `full`, `train`,
            or `test`.
        df: Split dataframe to summarize.
        full_props: Full-dataset proportions keyed by `(category, value)`.

    Returns:
        Audit rows for the requested split.
    """

    rows: list[dict[str, object]] = []
    total = len(df)
    rows.append(_row(split_name, "samples", "all", total, total, full_props))

    label_names = {0: "Real", 1: "Fake"}
    labels = pd.to_numeric(df["labels"], errors="raise").astype(int)
    for label, count in labels.value_counts().reindex([0, 1], fill_value=0).items():
        rows.append(_row(split_name, "label", label_names[int(label)], int(count), total, full_props))

    for topic, count in df["topic"].value_counts().sort_index().items():
        rows.append(_row(split_name, "topic", str(topic), int(count), total, full_props))

    topic_label_counts = (
        df.assign(label_name=labels.map(label_names))
        .groupby(["topic", "label_name"], observed=True)
        .size()
        .sort_index()
    )
    for (topic, label_name), count in topic_label_counts.items():
        rows.append(_row(split_name, "topic_label", f"{topic}__{label_name}", int(count), total, full_props))

    return rows


def _row(
    split_name: str,
    category: str,
    value: str,
    count: int,
    denominator: int,
    full_props: dict[tuple[str, str], float],
) -> dict[str, object]:
    """Build one distribution audit row.

    Purpose:
        Centralize the audit row format and compute the row's proportion,
        percentage, difference from the full dataset, and warning flag.

    Args:
        split_name: Split name to record.
        category: Distribution category, such as `label`, `topic`, or
            `topic_label`.
        value: Category value being counted.
        count: Number of rows with this value in the split.
        denominator: Total number of rows used to compute the proportion.
        full_props: Full-dataset proportions keyed by `(category, value)`.

    Returns:
        One audit-table row as a dictionary.
    """

    proportion = 0 if denominator == 0 else count / denominator
    full_prop = full_props.get((category, value))
    return {
        "split": split_name,
        "category": category,
        "value": value,
        "count": count,
        "proportion": proportion,
        "percentage": 100 * proportion,
        "full_proportion": full_prop,
        "difference_from_full": None if full_prop is None else proportion - full_prop,
        "warning": bool(full_prop is not None and abs(proportion - full_prop) > 0.03),
    }


def _distribution_props(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Calculate full-dataset proportions for audited distributions.

    Purpose:
        Provide baseline proportions that train and test summaries can compare
        against when flagging distribution drift.

    Args:
        df: Feature table containing `labels` and `topic`.

    Returns:
        Proportions keyed by `(category, value)` for labels, topics, and joint
        topic-label values.
    """

    props: dict[tuple[str, str], float] = {}
    labels = pd.to_numeric(df["labels"], errors="raise").astype(int)
    label_names = {0: "Real", 1: "Fake"}
    total = len(df)
    for label, count in labels.value_counts().reindex([0, 1], fill_value=0).items():
        props[("label", label_names[int(label)])] = int(count) / total
    for topic, count in df["topic"].value_counts().items():
        props[("topic", str(topic))] = int(count) / total
    grouped = df.assign(label_name=labels.map(label_names)).groupby(["topic", "label_name"], observed=True).size()
    for (topic, label_name), count in grouped.items():
        props[("topic_label", f"{topic}__{label_name}")] = int(count) / total
    return props


def _check_proportion_drift(full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> None:
    """Raise when train or test distribution drift is too large.

    Purpose:
        Guard against a split whose topic-label strata differ meaningfully from
        the full dataset, even if the split operation itself completed.

    Args:
        full_df: Complete source feature table.
        train_df: Candidate training split.
        test_df: Candidate held-out test split.
        column: Temporary column name used to store stratum labels.

    Raises:
        ValueError: If the maximum absolute proportion difference for any
            observed stratum exceeds 0.05 in train or test.
    """

    frames = [full_df.copy(), train_df.copy(), test_df.copy()]
    for frame in frames:
        frame[column] = stratum_labels(frame)
    full_props = frames[0][column].value_counts(normalize=True)
    for frame in frames[1:]:
        props = frame[column].value_counts(normalize=True)
        drift = (props - full_props).abs().dropna()
        if not drift.empty and drift.max() > 0.05:
            raise ValueError(f"Large {column} split drift detected: {drift.max():.3f}")


def _group_aware_split(df: pd.DataFrame, group_column: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows with stratification while keeping groups intact.

    Purpose:
        Use `StratifiedGroupKFold` to reserve one of five folds as the held-out
        test split when repeated group IDs mean row-level splitting would leak
        the same event across train and test.

    Args:
        df: Feature table to split.
        group_column: Column identifying groups that must remain in one split.
        random_state: Seed used by the shuffled fold splitter.

    Returns:
        Training and test dataframes copied from the selected fold indices.
    """

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    strata = stratum_labels(df)
    groups = df[group_column].astype(str)
    train_idx, test_idx = next(splitter.split(df, strata, groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
