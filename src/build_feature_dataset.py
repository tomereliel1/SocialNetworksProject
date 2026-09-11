from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.cascade_builder import build_interaction_graph
from src.cascade_features import extract_interaction_features
from src.cascade_validation import validate_cascade
from src.relation_parser import parse_interaction_relations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "MC_Fake_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "MC_Fake_dataset_features.csv"
RETWEET_COLUMN_CANDIDATES = ("retweet_relations", "retweet_relation")
REPLY_COLUMN_CANDIDATES = ("reply_relations", "reply_relation")
RAW_RELATION_CANDIDATES = (*RETWEET_COLUMN_CANDIDATES, *REPLY_COLUMN_CANDIDATES)
DEFAULT_PROGRESS_INTERVAL = 1_000


def build_feature_dataset(
    input_path: Path,
    output_path: Path,
    *,
    retweet_column: str | None = None,
    reply_column: str | None = None,
    keep_raw_relations: bool = False,
    limit: int | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> pd.DataFrame:
    """Build the Step 3 one-row-per-event feature dataset.

    Args:
        input_path: Path to the source MC-Fake CSV.
        output_path: Path where the feature CSV should be written.
        retweet_column: Optional explicit retweet relation column name. When
            omitted, the column is detected from known candidates.
        reply_column: Optional explicit reply relation column name. When
            omitted, the column is detected from known candidates.
        keep_raw_relations: Whether to preserve raw relation columns in the
            output dataset.
        limit: Optional maximum number of input rows to process.
        progress_interval: Number of rows between progress messages. Values
            less than or equal to zero disable periodic progress messages.

    Returns:
        The generated feature dataset.

    Raises:
        ValueError: If required relation columns cannot be detected, validation
            fails, or output integrity checks fail.
    """

    log_progress(f"Loading input CSV: {input_path}")
    df = pd.read_csv(input_path, dtype=str, nrows=limit)
    log_progress(f"Loaded {len(df):,} event rows")

    retweet_column = detect_relation_column(df.columns, retweet_column, RETWEET_COLUMN_CANDIDATES, "retweet")
    reply_column = detect_relation_column(df.columns, reply_column, REPLY_COLUMN_CANDIDATES, "reply")
    log_progress(f"Using retweet column: {retweet_column}")
    log_progress(f"Using reply column: {reply_column}")

    preserved_df = (
        df.copy()
        if keep_raw_relations
        else df.drop(columns=raw_columns_to_drop(df.columns), errors="ignore")
    )
    records: list[dict[str, object]] = []
    status_counts: dict[str, Counter[str]] = {"rt": Counter(), "reply": Counter()}
    cyclic_counts = {"rt": 0, "reply": 0}

    total_rows = len(df)
    log_progress(f"Extracting graph features for {total_rows:,} events")
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        record = preserved_df.iloc[index - 1].to_dict()
        for relation_type, column, prefix in [
            ("retweet", retweet_column, "rt"),
            ("reply", reply_column, "reply"),
        ]:
            parsed = parse_interaction_relations(row[column], strict=False)
            graph = build_interaction_graph(
                parsed.relations,
                interaction_type=relation_type,
                direction="propagation",
            )
            validation = validate_cascade(
                graph,
                parse_error_count=len(parsed.malformed),
                duplicate_relations=parsed.duplicate_relations,
            )
            features = extract_interaction_features(
                graph,
                prefix=prefix,
                validation=validation,
                parse_result=parsed,
            )
            record.update(features)
            status_counts[prefix][validation.graph_status] += 1
            cyclic_counts[prefix] += int(validation.has_cycle)

        records.append(record)
        if progress_interval > 0 and (index % progress_interval == 0 or index == total_rows):
            log_progress(f"Processed {index:,} / {total_rows:,} events")

    log_progress("Validating output integrity")
    output_df = pd.DataFrame(records)
    validate_output_integrity(df, output_df, keep_raw_relations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_progress(f"Writing feature dataset: {output_path}")
    output_df.to_csv(output_path, index=False)
    print_summary(df, output_df, output_path, status_counts, cyclic_counts)
    return output_df


def detect_relation_column(
    columns: pd.Index,
    explicit_column: str | None,
    candidates: tuple[str, ...],
    relation_name: str,
) -> str:
    """Detect or validate the raw relation column for one interaction type.

    Args:
        columns: Columns available in the input dataset.
        explicit_column: User-provided column name, if any.
        candidates: Candidate column names to use for automatic detection.
        relation_name: Human-readable relation name for error messages.

    Returns:
        The selected relation column name.

    Raises:
        ValueError: If the explicit column is missing, no candidates are found,
            or multiple candidates match.
    """

    if explicit_column is not None:
        if explicit_column not in columns:
            raise ValueError(f"{relation_name} relation column not found: {explicit_column}")
        return explicit_column

    matches = [candidate for candidate in candidates if candidate in columns]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        joined = ", ".join(candidates)
        raise ValueError(f"Could not find {relation_name} relation column. Tried: {joined}")

    joined = ", ".join(matches)
    raise ValueError(
        f"Multiple possible {relation_name} relation columns found: {joined}. "
        f"Specify --{relation_name}-column explicitly."
    )


def raw_columns_to_drop(columns: pd.Index) -> list[str]:
    """Return raw relation columns present in the dataset.

    Args:
        columns: Columns available in the input dataset.

    Returns:
        Raw retweet/reply relation columns that should be removed by default.
    """

    return [column for column in RAW_RELATION_CANDIDATES if column in columns]


def validate_output_integrity(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    keep_raw_relations: bool,
) -> None:
    """Run row-count, ordering, and column sanity checks before writing.

    Args:
        input_df: Source dataset loaded from the input CSV.
        output_df: Generated feature dataset.
        keep_raw_relations: Whether raw relation columns were intentionally
            preserved.

    Raises:
        ValueError: If row counts differ, required preserved columns are
            missing or reordered, output column names are duplicated, or raw
            relation columns were retained unexpectedly.
    """

    if len(output_df) != len(input_df):
        raise ValueError(f"Output row count {len(output_df)} does not match input row count {len(input_df)}")

    for column in ["news_id", "labels"]:
        if column in input_df.columns:
            if column not in output_df.columns:
                raise ValueError(f"Required preserved column missing from output: {column}")
            if not input_df[column].reset_index(drop=True).equals(output_df[column].reset_index(drop=True)):
                raise ValueError(f"Output column order/content changed for {column}")

    if len(output_df.columns) != len(set(output_df.columns)):
        raise ValueError("Output feature column names are not unique")

    if not keep_raw_relations:
        retained_raw_columns = [
            column for column in raw_columns_to_drop(input_df.columns) if column in output_df.columns
        ]
        if retained_raw_columns:
            joined = ", ".join(retained_raw_columns)
            raise ValueError(f"Raw relation columns were retained unexpectedly: {joined}")


def print_summary(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    output_path: Path,
    status_counts: dict[str, Counter[str]],
    cyclic_counts: dict[str, int],
) -> None:
    """Print a compact processing summary.

    Args:
        input_df: Source dataset loaded from the input CSV.
        output_df: Generated feature dataset.
        output_path: Path where `output_df` was written.
        status_counts: Graph status counts keyed by interaction prefix.
        cyclic_counts: Cyclic graph counts keyed by interaction prefix.
    """

    print(f"Input rows: {len(input_df)}")
    print(f"Output rows: {len(output_df)}")
    for prefix, label in [("rt", "Retweet"), ("reply", "Reply")]:
        print(f"{label} EMPTY: {status_counts[prefix].get('EMPTY', 0)}")
        print(f"{label} MALFORMED: {status_counts[prefix].get('MALFORMED', 0)}")
        print(f"{label} cyclic: {cyclic_counts[prefix]}")
    print(f"Output path: {output_path}")


def log_progress(message: str) -> None:
    """Print lightweight progress updates immediately.

    Args:
        message: Progress message to print to stderr.
    """

    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Step 3 feature extraction.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Build MC-Fake event-level graph feature dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retweet-column")
    parser.add_argument("--reply-column")
    parser.add_argument("--keep-raw-relations", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_PROGRESS_INTERVAL)
    return parser.parse_args()


def main() -> None:
    """Run the Step 3 dataset builder.

    Parses command-line arguments and writes the generated feature dataset.
    """

    args = parse_args()
    build_feature_dataset(
        args.input,
        args.output,
        retweet_column=args.retweet_column,
        reply_column=args.reply_column,
        keep_raw_relations=args.keep_raw_relations,
        limit=args.limit,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
