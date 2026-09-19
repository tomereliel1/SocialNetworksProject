from __future__ import annotations

import argparse
from pathlib import Path

from src.classification.constants import (
    DEFAULT_CLASSIFICATION_DATA_DIR,
    DEFAULT_FEATURE_TABLE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
)
from src.classification.correlation_experiment import run_correlation_experiment
from src.classification.split import create_frozen_split
from src.classification.topic_experiment import run_topic_experiments
from src.classification.workflow import run_training_workflow


def run_split(args: argparse.Namespace) -> None:
    outputs = create_frozen_split(
        args.input,
        args.train,
        args.test,
        args.results,
    )
    print("Created frozen split")
    for key, value in outputs.items():
        print(f"{key}: {value}")


def run_experiment(args: argparse.Namespace) -> None:
    result = run_training_workflow(args.train, args.results)
    print(f"Wrote classification report: {result.report_path}")
    print("Experiment used train.csv only. No test.csv evaluation was performed.")


def run_all(args: argparse.Namespace) -> None:
    split_args = argparse.Namespace(input=args.input, train=args.train, test=args.test, results=args.results)
    run_split(split_args)
    experiment_args = argparse.Namespace(train=args.train, results=args.results)
    run_experiment(experiment_args)


def run_correlation(args: argparse.Namespace) -> None:
    artifacts = run_correlation_experiment(args.train, args.results)
    print(f"Wrote correlation experiment report: {args.results / 'correlation_experiment_report.md'}")
    print(f"Selected development configuration: {artifacts.best_configuration['selected_configuration']}")
    print("Experiment used train.csv only. No test.csv evaluation was performed.")


def run_topics(args: argparse.Namespace) -> None:
    summary = run_topic_experiments(args.input, args.data_root, args.results)
    print(f"Wrote topic experiment summary: {summary.summary_report_path}")
    print(f"Wrote topic comparison table: {summary.summary_table_path}")
    print("Each topic experiment used its own train.csv only. No topic test.csv was evaluated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC-Fake classification split and train-only model selection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="Create train.csv/test.csv and split audit; do not train models.")
    split.add_argument("--input", type=Path, default=DEFAULT_FEATURE_TABLE)
    split.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    split.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    split.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    split.set_defaults(func=run_split)

    experiment = subparsers.add_parser("experiment", help="Run model selection using train.csv only.")
    experiment.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    experiment.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    experiment.set_defaults(func=run_experiment)

    correlation = subparsers.add_parser(
        "correlation-experiment",
        help="Run train-only correlation pruning and sparse Logistic Regression follow-up.",
    )
    correlation.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    correlation.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR / "correlation_experiment")
    correlation.set_defaults(func=run_correlation)

    topics = subparsers.add_parser(
        "topic-experiments",
        help="Create per-topic datasets and run train-only classification for each topic.",
    )
    topics.add_argument("--input", type=Path, default=DEFAULT_FEATURE_TABLE)
    topics.add_argument("--data-root", type=Path, default=DEFAULT_CLASSIFICATION_DATA_DIR / "by_topic")
    topics.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR / "by_topic")
    topics.set_defaults(func=run_topics)

    all_cmd = subparsers.add_parser("all", help="Create the split, then run train-only model selection.")
    all_cmd.add_argument("--input", type=Path, default=DEFAULT_FEATURE_TABLE)
    all_cmd.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    all_cmd.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    all_cmd.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    all_cmd.set_defaults(func=run_all)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
