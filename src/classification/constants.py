from __future__ import annotations

from pathlib import Path

try:
    from src.analyze_data import TOPIC_GROUPS
except Exception:  # pragma: no cover - defensive fallback for standalone use
    TOPIC_GROUPS = {
        "Politics": ["politifact", "RealPolitics"],
        "Entertainment": ["gossipcop"],
        "Health": ["HealthStory", "HealthRelease", "RealHealth"],
        "Covid": ["FakeCovid", "FakeCovidClaimFiltered", "RealCovid"],
        "Syria War": ["FA-KES", "RealSyria"],
    }


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_TABLE = PROJECT_ROOT / "data" / "MC_Fake_dataset_features.csv"
DEFAULT_CLASSIFICATION_DATA_DIR = PROJECT_ROOT / "data" / "classification"
DEFAULT_TRAIN_PATH = DEFAULT_CLASSIFICATION_DATA_DIR / "train.csv"
DEFAULT_TEST_PATH = DEFAULT_CLASSIFICATION_DATA_DIR / "test.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "classification"
RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = 1
LABEL_NAMES = {0: "Real", 1: "Fake"}
N_SPLITS = 5

