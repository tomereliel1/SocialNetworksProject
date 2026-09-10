from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "MC_Fake_dataset.csv"
LABEL_MEANINGS = {
    0: "real / true news",
    1: "fake / false news",
}
TOPIC_GROUPS = {
    "Politics": ["politifact", "RealPolitics"],
    "Entertainment": ["gossipcop"],
    "Health": ["HealthStory", "HealthRelease", "RealHealth"],
    "Covid": ["FakeCovid", "FakeCovidClaimFiltered", "RealCovid"],
    "Syria War": ["FA-KES", "RealSyria"],
}


def get_value_type(value) -> str:
    if pd.isna(value):
        return "missing"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "double"

    if isinstance(value, str):
        if "," in value:
            return "list/string"
        return "string"

    return type(value).__name__


def print_classification_summary(df: pd.DataFrame) -> None:
    print("classification summary:")
    print("Each row is classified by the 'labels' column.")
    print("In this dataset, the label meanings are:")
    for label, meaning in LABEL_MEANINGS.items():
        print(f"  {label}: {meaning}")
    print()

    print("possible label values:")
    print(df["labels"].value_counts(dropna=False).sort_index())
    print()

    print("possible data_name values:")
    print(df["data_name"].value_counts(dropna=False).sort_index())
    print()

    print("topic counts:")
    topic_rows = []
    for topic, data_names in TOPIC_GROUPS.items():
        topic_df = df[df["data_name"].isin(data_names)]
        label_counts = topic_df["labels"].value_counts()
        topic_rows.append(
            {
                "topic": topic,
                "fake": label_counts.get(1, 0),
                "real": label_counts.get(0, 0),
                "total": len(topic_df),
                "data_names": ", ".join(data_names),
            }
        )
    print(pd.DataFrame(topic_rows).to_string(index=False))
    print()

    print("data_name meaning:")
    print("'data_name' tells which subset/source/topic the row came from.")
    print("It is not always the final fake/real classification, because some subsets contain both labels.")
    print("Use 'labels' for the row classification.")
    print()

    print("labels by data_name:")
    print(pd.crosstab(df["data_name"], df["labels"], dropna=False))
    print()


def print_missing_values_summary(df: pd.DataFrame) -> None:
    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / len(df) * 100).round(2)
    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percent": missing_percent,
        }
    )

    print("missing values summary:")
    print(missing_summary.to_string())
    print()

    print("columns with missing values:")
    print(missing_summary[missing_summary["missing_count"] > 0].to_string())
    print()

    print("columns without missing values:", (missing_counts == 0).sum())
    print()


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    print("dimensions:", df.shape)
    print("rows:", df.shape[0])
    print("columns:", df.shape[1])
    print("column keys:", df.columns.tolist())
    print()

    print(df.head(10))
    print()

    print_classification_summary(df)
    print_missing_values_summary(df)

    first_row = df.iloc[0]
    print("first row details:")
    for column, value in first_row.items():
        print("key:", column)
        print("type:", get_value_type(value))
        print("data:", value)
        print()


if __name__ == "__main__":
    main()
