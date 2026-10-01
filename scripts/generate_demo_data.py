from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N = 2500

rng = np.random.default_rng(RANDOM_SEED)

age = rng.integers(18, 91, N)
length_of_stay = np.clip(rng.gamma(2.2, 2.5, N).round(), 1, 30).astype(int)
prior_admissions = np.clip(rng.poisson(1.2, N), 0, 8)
medication_count = np.clip(rng.poisson(5.5, N), 0, 20)
comorbidity_count = np.clip(rng.poisson(2.2, N), 0, 10)

heart_rate = np.clip(rng.normal(82, 14, N), 45, 150)
systolic_bp = np.clip(rng.normal(128, 20, N), 75, 210)
oxygen_saturation = np.clip(rng.normal(96, 3, N), 75, 100)
creatinine = np.clip(rng.lognormal(mean=np.log(1.0), sigma=0.45, size=N), 0.4, 8)
glucose = np.clip(rng.normal(125, 45, N), 50, 400)
hemoglobin = np.clip(rng.normal(13.0, 2.0, N), 6, 18)

has_diabetes = rng.binomial(1, 0.28, N)
has_copd = rng.binomial(1, 0.12, N)
has_heart_failure = rng.binomial(1, 0.10, N)
has_renal_disease = rng.binomial(1, 0.12, N)

# A synthetic risk-generating process for software demonstration only.
logit = (
    -4.2
    + 0.025 * (age - 50)
    + 0.16 * length_of_stay
    + 0.32 * prior_admissions
    + 0.08 * medication_count
    + 0.20 * comorbidity_count
    + 0.35 * has_diabetes
    + 0.50 * has_copd
    + 0.70 * has_heart_failure
    + 0.45 * has_renal_disease
    + 0.015 * np.maximum(heart_rate - 90, 0)
    + 0.012 * np.maximum(110 - systolic_bp, 0)
    + 0.11 * np.maximum(94 - oxygen_saturation, 0)
    + 0.30 * np.maximum(creatinine - 1.2, 0)
    + 0.002 * np.maximum(glucose - 140, 0)
    - 0.06 * np.maximum(hemoglobin - 13, 0)
)

probability = 1 / (1 + np.exp(-logit))
readmitted_30d = rng.binomial(1, probability)

df = pd.DataFrame({
    "patient_id": [f"P{i:05d}" for i in range(1, N + 1)],
    "age": age,
    "length_of_stay": length_of_stay,
    "prior_admissions": prior_admissions,
    "medication_count": medication_count,
    "comorbidity_count": comorbidity_count,
    "heart_rate": heart_rate.round(1),
    "systolic_bp": systolic_bp.round(1),
    "oxygen_saturation": oxygen_saturation.round(1),
    "creatinine": creatinine.round(2),
    "glucose": glucose.round(1),
    "hemoglobin": hemoglobin.round(1),
    "has_diabetes": has_diabetes,
    "has_copd": has_copd,
    "has_heart_failure": has_heart_failure,
    "has_renal_disease": has_renal_disease,
    "readmitted_30d": readmitted_30d,
})

out = Path("data/raw/demo_patients.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"Created {out} with {len(df)} rows.")
print(f"Readmission rate: {df['readmitted_30d'].mean():.2%}")
