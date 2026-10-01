from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.models.predict import load_artifacts


def explain_shap(X: pd.DataFrame):
    _, base_model, metadata = load_artifacts()
    features = metadata["features"]

    # The selected model is currently a sklearn Pipeline:
    # StandardScaler -> LogisticRegression
    if hasattr(base_model, "named_steps") and "model" in base_model.named_steps:
        estimator = base_model.named_steps["model"]
        scaler = base_model.named_steps.get("scaler")

        # Transform the input using the same scaler used during training.
        X_transformed = scaler.transform(X) if scaler is not None else X

        # Logistic Regression explanation.
        explainer = shap.LinearExplainer(
            estimator,
            X_transformed,
        )

        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 2:
            row_values = shap_values[0]
        else:
            row_values = shap_values

        result = pd.DataFrame({
            "feature": features,
            "value": X.iloc[0].values,
            "shap_value": row_values,
        })

    else:
        # Tree-based models such as XGBoost / LightGBM.
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 2:
            row_values = shap_values[0]
        else:
            row_values = shap_values

        result = pd.DataFrame({
            "feature": features,
            "value": X.iloc[0].values,
            "shap_value": row_values,
        })

    result["abs_shap"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap", ascending=False)

    return result


def explain_lime(X: pd.DataFrame):
    # Imported lazily because LIME is only needed when this endpoint is used.
    from lime.lime_tabular import LimeTabularExplainer

    _, base_model, metadata = load_artifacts()
    features = metadata["features"]

    rng = np.random.default_rng(42)

    background = np.tile(
        X.iloc[0].values,
        (100, 1)
    )

    noise = rng.normal(
        0,
        0.03,
        background.shape
    )

    background = background * (1 + noise)

    explainer = LimeTabularExplainer(
        background,
        feature_names=features,
        class_names=["No readmission", "Readmission"],
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        X.iloc[0].values,
        base_model.predict_proba,
        num_features=8,
    )

    return [
        {
            "feature_rule": rule,
            "weight": float(weight)
        }
        for rule, weight in explanation.as_list(label=1)
    ]