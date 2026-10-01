from fastapi import FastAPI, HTTPException
import pandas as pd
from src.api.schemas import PatientInput, PredictionResponse
from src.models.predict import predict_one
from src.explainability.explainer import explain_shap, explain_lime

app = FastAPI(
    title="Hospital Readmission Risk API",
    version="0.1.0",
    description="Academic prototype API for explainable 30-day readmission risk prediction.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        probability, band, X = predict_one(patient.model_dump())
        shap_df = explain_shap(X)

        top = []
        for _, row in shap_df.head(5).iterrows():
            top.append({
                "feature": row["feature"],
                "value": float(row["value"]),
                "impact": float(row["shap_value"]),
                "direction": "increases risk" if row["shap_value"] > 0 else "decreases risk",
            })

        return PredictionResponse(
            risk_probability=probability,
            risk_band=band,
            top_factors=top,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/explain/lime")
def lime_explanation(payload: PatientInput):
    try:
        X = pd.DataFrame([payload.model_dump()])
        result = explain_lime(X)

        return {
            "explanation": result
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
