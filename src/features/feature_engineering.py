from __future__ import annotations

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features.

    For the demo dataset, comorbidity_count is used as a lightweight proxy
    for the richer Charlson Comorbidity Index described in the project plan.
    A validated CCI implementation should replace this when an approved
    clinical dataset is integrated.
    """
    out = df.copy()

    out["age_over_65"] = (out["age"] >= 65).astype(int)
    out["long_stay"] = (out["length_of_stay"] >= 7).astype(int)
    out["frequent_admissions"] = (out["prior_admissions"] >= 3).astype(int)
    out["polypharmacy"] = (out["medication_count"] >= 10).astype(int)

    return out
