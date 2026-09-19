from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.classification.modeling import ExperimentArtifacts, run_train_only_experiment
from src.classification.plots import write_figures
from src.classification.report import write_report


@dataclass(frozen=True)
class WorkflowResult:
    artifacts: ExperimentArtifacts
    report_path: Path


def run_training_workflow(
    train_path: Path,
    results_dir: Path,
    *,
    split_summary_path: Path | None = None,
    group_report_path: Path | None = None,
) -> WorkflowResult:
    artifacts = run_train_only_experiment(train_path, results_dir)
    write_figures(
        results_dir,
        artifacts.model_comparison,
        artifacts.ablation_results,
        artifacts.confusion,
        artifacts.roc_curve,
        artifacts.pr_curve,
        artifacts.feature_importance,
    )
    report_path = write_report(
        results_dir,
        train_path=train_path,
        split_summary_path=split_summary_path or results_dir / "split_summary.csv",
        group_report_path=group_report_path or results_dir / "grouping_and_split_checks.txt",
        model_comparison=artifacts.model_comparison,
        ablation_results=artifacts.ablation_results,
        train_diagnostics=artifacts.train_diagnostics,
        best_model_config=artifacts.best_model_config,
        selected_features=artifacts.selected_features,
        excluded_features=artifacts.excluded_features,
        high_correlations=artifacts.high_correlations,
        feature_importance=artifacts.feature_importance,
    )
    return WorkflowResult(artifacts=artifacts, report_path=report_path)
