from pathlib import Path
from src.data.preprocess import load_dataset, FEATURE_COLUMNS


def test_demo_dataset_schema():
    path = Path("data/raw/demo_patients.csv")
    if not path.exists():
        return
    df = load_dataset(path)
    assert all(c in df.columns for c in FEATURE_COLUMNS)
    assert "readmitted_30d" in df.columns
