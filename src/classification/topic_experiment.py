from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.classification.constants import TOPIC_GROUPS
from src.classification.features import add_topic_column
from src.classification.split import create_frozen_split, validate_minimum_schema
from src.classification.workflow import run_training_workflow


@dataclass(frozen=True)
class TopicDataset:
    topic: str
    slug: str
    dataset_path: Path
    train_path: Path
    test_path: Path
    results_dir: Path


@dataclass(frozen=True)
class TopicExperimentSummary:
    summary_table: pd.DataFrame
    summary_table_path: Path
    summary_report_path: Path
    topic_datasets: list[TopicDataset]


def run_topic_experiments(input_path: Path, data_root: Path, results_root: Path) -> TopicExperimentSummary:
    df = add_topic_column(pd.read_csv(input_path, low_memory=False))
    validate_minimum_schema(df)
    topic_datasets = create_topic_datasets(df, data_root, results_root)

    summary_rows: list[dict[str, object]] = []
    for topic_dataset in topic_datasets:
        split_outputs = create_frozen_split(
            topic_dataset.dataset_path,
            topic_dataset.train_path,
            topic_dataset.test_path,
            topic_dataset.results_dir,
        )
        workflow_result = run_training_workflow(
            topic_dataset.train_path,
            topic_dataset.results_dir,
            split_summary_path=Path(split_outputs["split_summary"]),
            group_report_path=Path(split_outputs["group_report"]),
        )
        summary_rows.append(topic_summary_row(topic_dataset, workflow_result.report_path))

    summary_table = pd.DataFrame(summary_rows).sort_values("topic")
    results_root.mkdir(parents=True, exist_ok=True)
    summary_table_path = results_root / "topic_experiment_summary.csv"
    summary_table.to_csv(summary_table_path, index=False)
    summary_report_path = write_topic_summary_report(summary_table, results_root / "topic_experiment_summary.md")
    return TopicExperimentSummary(
        summary_table=summary_table,
        summary_table_path=summary_table_path,
        summary_report_path=summary_report_path,
        topic_datasets=topic_datasets,
    )


def create_topic_datasets(df: pd.DataFrame, data_root: Path, results_root: Path) -> list[TopicDataset]:
    data_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    topics = list(TOPIC_GROUPS)
    datasets: list[TopicDataset] = []

    for topic in topics:
        topic_df = df[df["topic"] == topic].sort_index().copy()
        if topic_df.empty:
            raise ValueError(f"No rows found for configured topic: {topic}")
        label_counts = pd.to_numeric(topic_df["labels"], errors="raise").astype(int).value_counts()
        missing_labels = sorted({0, 1} - set(label_counts.index))
        if missing_labels:
            raise ValueError(f"Topic {topic} is missing labels required for binary classification: {missing_labels}")

        slug = slugify_topic(topic)
        topic_data_dir = data_root / slug
        topic_results_dir = results_root / slug
        topic_data_dir.mkdir(parents=True, exist_ok=True)
        topic_results_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = topic_data_dir / "dataset.csv"
        topic_df.to_csv(dataset_path, index=False)
        datasets.append(
            TopicDataset(
                topic=topic,
                slug=slug,
                dataset_path=dataset_path,
                train_path=topic_data_dir / "train.csv",
                test_path=topic_data_dir / "test.csv",
                results_dir=topic_results_dir,
            )
        )

    return datasets


def topic_summary_row(topic_dataset: TopicDataset, report_path: Path) -> dict[str, object]:
    split_summary = pd.read_csv(topic_dataset.results_dir / "split_summary.csv")
    model_comparison = pd.read_csv(topic_dataset.results_dir / "model_comparison.csv")
    best = model_comparison.iloc[0]

    labels = split_summary[split_summary["category"] == "label"].set_index(["split", "value"])
    full_rows = _split_count(split_summary, "full")
    train_rows = _split_count(split_summary, "train")
    test_rows = _split_count(split_summary, "test")
    full_real = labels.loc[("full", "Real")]
    full_fake = labels.loc[("full", "Fake")]
    train_real = labels.loc[("train", "Real")]
    train_fake = labels.loc[("train", "Fake")]

    return {
        "topic": topic_dataset.topic,
        "dataset_path": topic_dataset.dataset_path,
        "train_path": topic_dataset.train_path,
        "test_path": topic_dataset.test_path,
        "results_dir": topic_dataset.results_dir,
        "report_path": report_path,
        "full_rows": full_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "full_real_count": int(full_real["count"]),
        "full_fake_count": int(full_fake["count"]),
        "full_real_percentage": float(full_real["percentage"]),
        "full_fake_percentage": float(full_fake["percentage"]),
        "train_real_percentage": float(train_real["percentage"]),
        "train_fake_percentage": float(train_fake["percentage"]),
        "selected_model": best["model"],
        "f1_mean": float(best["f1_mean"]),
        "f1_std": float(best["f1_std"]),
        "balanced_accuracy_mean": float(best["balanced_accuracy_mean"]),
        "roc_auc_mean": float(best["roc_auc_mean"]),
        "pr_auc_mean": float(best["pr_auc_mean"]),
    }


def write_topic_summary_report(summary_table: pd.DataFrame, output_path: Path) -> Path:
    lines = [
        "# Per-Topic Classification Experiments",
        "",
        "Each topic was materialized as its own engineered-feature dataset, split into train/test with the same label proportions as that topic's full dataset, and modeled using the existing train-only classification workflow.",
        "",
        "The held-out `test.csv` for each topic was created and frozen but not evaluated. Reported metrics are development metrics from cross-validation on that topic's `train.csv` only.",
        "",
        "## Topic Results",
        "",
        _markdown_table(
            summary_table[
                [
                    "topic",
                    "full_rows",
                    "full_real_percentage",
                    "full_fake_percentage",
                    "selected_model",
                    "f1_mean",
                    "f1_std",
                    "balanced_accuracy_mean",
                    "roc_auc_mean",
                    "pr_auc_mean",
                ]
            ]
        ),
        "",
        "## Artifacts",
        "",
        _markdown_table(summary_table[["topic", "dataset_path", "train_path", "test_path", "report_path"]]),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    if not slug:
        raise ValueError(f"Could not create a filesystem slug for topic: {topic!r}")
    return slug


def _split_count(split_summary: pd.DataFrame, split: str) -> int:
    row = split_summary[(split_summary["split"] == split) & (split_summary["category"] == "samples")].iloc[0]
    return int(row["count"])


def _markdown_table(df: pd.DataFrame) -> str:
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
    rows = ["| " + " | ".join(_escape_cell(row[column]) for column in columns) + " |" for _, row in out.iterrows()]
    return "\n".join([header, separator, *rows])


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
