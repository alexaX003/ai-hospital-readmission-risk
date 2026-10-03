from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.models.predict import load_artifacts


# Project root directory
ROOT = Path(__file__).resolve().parents[2]


def _load_background_data(features):
    """
    Load the synthetic demo dataset used for model training
    and keep only the model input features.
    """
    data_path = ROOT / "data" / "raw" / "demo_patients.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Background dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df[features]


def _get_shap_values(explainer, X):
    """
    Handle different SHAP output formats.
    """
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.asarray(shap_values)


def explain_shap(X: pd.DataFrame):
    """
    Generate a patient-level SHAP explanation.

    Returns the contribution of every feature for the
    first patient in X.
    """
    _, base_model, metadata = load_artifacts()
    features = metadata["features"]

    background = _load_background_data(features)

    # Current selected model:
    # StandardScaler -> LogisticRegression
    if (
        hasattr(base_model, "named_steps")
        and "model" in base_model.named_steps
    ):
        estimator = base_model.named_steps["model"]
        scaler = base_model.named_steps.get("scaler")

        # Apply the same scaler used during training.
        background_transformed = (
            scaler.transform(background)
            if scaler is not None
            else background
        )

        X_transformed = (
            scaler.transform(X)
            if scaler is not None
            else X
        )

        explainer = shap.LinearExplainer(
            estimator,
            background_transformed,
        )

        shap_values = _get_shap_values(
            explainer,
            X_transformed,
        )

    else:
        # Tree-based models such as XGBoost / LightGBM.
        explainer = shap.TreeExplainer(base_model)

        shap_values = _get_shap_values(
            explainer,
            X,
        )

    if shap_values.ndim == 2:
        row_values = shap_values[0]
    else:
        row_values = shap_values

    result = pd.DataFrame(
        {
            "feature": features,
            "value": X.iloc[0].values,
            "shap_value": row_values,
        }
    )

    result["abs_shap"] = result["shap_value"].abs()

    result = result.sort_values(
        "abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    return result


def explain_shap_global():
    """
    Generate global SHAP feature importance.

    Uses mean absolute SHAP value across the complete
    synthetic demo dataset.
    """
    _, base_model, metadata = load_artifacts()
    features = metadata["features"]

    background = _load_background_data(features)

    if (
        hasattr(base_model, "named_steps")
        and "model" in base_model.named_steps
    ):
        estimator = base_model.named_steps["model"]
        scaler = base_model.named_steps.get("scaler")

        X_transformed = (
            scaler.transform(background)
            if scaler is not None
            else background
        )

        explainer = shap.LinearExplainer(
            estimator,
            X_transformed,
        )

        shap_values = _get_shap_values(
            explainer,
            X_transformed,
        )

    else:
        explainer = shap.TreeExplainer(base_model)

        shap_values = _get_shap_values(
            explainer,
            background,
        )

    # Mean absolute contribution for every feature.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    result = result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    return result


def explain_lime(X: pd.DataFrame):
    """
    Generate a patient-level LIME explanation.
    """
    from lime.lime_tabular import LimeTabularExplainer

    _, base_model, metadata = load_artifacts()
    features = metadata["features"]

    background = _load_background_data(features)

    explainer = LimeTabularExplainer(
        background.values,
        feature_names=features,
        class_names=[
            "No readmission",
            "Readmission",
        ],
        mode="classification",
        random_state=42,
    )

    # LIME produces NumPy arrays without column names.
    # Convert them back to a DataFrame before passing
    # them to the sklearn pipeline.
    def predict_proba_for_lime(data):
        data_df = pd.DataFrame(
            data,
            columns=features,
        )

        return base_model.predict_proba(data_df)

    explanation = explainer.explain_instance(
        X.iloc[0].values,
        predict_proba_for_lime,
        num_features=8,
    )

    return [
        {
            "feature_rule": rule,
            "weight": float(weight),
        }
        for rule, weight in explanation.as_list(label=1)
    ]
def compare_shap_lime(X: pd.DataFrame):
    """
    Compare SHAP and LIME explanations for one patient.

    The comparison is based on the direction of each explanation:
        positive -> pushes toward readmission
        negative -> pushes away from readmission

    LIME only returns its top features, so the agreement percentage
    is calculated only among features explained by both methods.
    """

    # -----------------------------
    # 1. Get SHAP explanation
    # -----------------------------
    shap_result = explain_shap(X)

    # -----------------------------
    # 2. Get LIME explanation
    # -----------------------------
    lime_result = explain_lime(X)

    # -----------------------------
    # 3. Convert LIME rules into
    #    feature -> weight mapping
    # -----------------------------
    lime_weights = {}

    for item in lime_result:
        rule = item["feature_rule"]
        weight = item["weight"]

        matched_feature = None

        for feature in shap_result["feature"]:
            if feature in rule:
                matched_feature = feature
                break

        if matched_feature is not None:
            lime_weights[matched_feature] = weight

    # -----------------------------
    # 4. Compare SHAP and LIME
    # -----------------------------
    comparison = []

    for _, row in shap_result.iterrows():

        feature = row["feature"]
        shap_value = float(row["shap_value"])

        lime_weight = lime_weights.get(feature)

        # LIME did not explain this feature
        if lime_weight is None:
            agreement = None

        # Ignore zero contributions
        elif shap_value == 0 or lime_weight == 0:
            agreement = None

        else:
            # Check whether SHAP and LIME
            # have the same direction.
            agreement = (
                np.sign(shap_value)
                == np.sign(lime_weight)
            )

        comparison.append(
            {
                "feature": feature,
                "shap_value": shap_value,
                "lime_weight": lime_weight,
                "direction_agreement": agreement,
            }
        )

    result = pd.DataFrame(comparison)

    # -----------------------------
    # 5. Calculate agreement only
    #    for comparable features
    # -----------------------------
    comparable = result[
        result["direction_agreement"].notna()
    ]

    if len(comparable) > 0:

        agreements = comparable[
            "direction_agreement"
        ].tolist()

        agreement_percentage = (
            sum(agreements) / len(agreements)
        ) * 100

    else:
        agreement_percentage = 0.0

    return result, agreement_percentage