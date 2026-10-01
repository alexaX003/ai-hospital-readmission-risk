from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

_model = None
_base_model = None
_metadata = None


def load_artifacts():
    global _model, _base_model, _metadata

    if _model is None:
        calibrated_path = MODEL_DIR / "calibrated_model.joblib"
        base_path = MODEL_DIR / "base_model.joblib"
        metadata_path = MODEL_DIR / "metadata.json"

        if not calibrated_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run: python scripts/train_models.py"
            )

        _model = joblib.load(calibrated_path)
        _base_model = joblib.load(base_path)
        _metadata = json.loads(metadata_path.read_text())

    return _model, _base_model, _metadata


def predict_one(payload: dict):
    model, _, metadata = load_artifacts()
    features = metadata["features"]

    row = {feature: payload[feature] for feature in features}
    X = pd.DataFrame([row], columns=features)

    probability = float(model.predict_proba(X)[0, 1])

    if probability < 0.30:
        band = "Low"
    elif probability < 0.60:
        band = "Medium"
    else:
        band = "High"

    return probability, band, X
