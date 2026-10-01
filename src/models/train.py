from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data.preprocess import FEATURE_COLUMNS, TARGET, load_dataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/demo_patients.csv"
MODEL_DIR = ROOT / "models"


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            verbosity=-1,
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_val, y_val)

    # Select using validation AUC; report all models rather than hiding the comparison.
    best_name = max(results, key=lambda n: results[n]["auc_roc"])
    best_base = models[best_name]

    # Calibrate the selected model so predicted probabilities can be interpreted
    # as probabilities more appropriately than raw tree scores.
    calibrated = CalibratedClassifierCV(best_base, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    test_metrics = evaluate(calibrated, X_test, y_test)

    joblib.dump(best_base, MODEL_DIR / "base_model.joblib")
    joblib.dump(calibrated, MODEL_DIR / "calibrated_model.joblib")

    metadata = {
        "best_model": best_name,
        "features": FEATURE_COLUMNS,
        "validation_results": results,
        "test_results": test_metrics,
        "synthetic_demo": True,
    }

    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nValidation results:")
    print(pd.DataFrame(results).T.round(4))
    print(f"\nSelected model: {best_name}")
    print("\nHeld-out test metrics:")
    print(pd.Series(test_metrics).round(4))


if __name__ == "__main__":
    main()
