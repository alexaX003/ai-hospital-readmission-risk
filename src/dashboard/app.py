import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Readmission Risk",
    page_icon="🏥",
    layout="wide",
)

st.title("AI-Driven Hospital Readmission Risk")
st.caption("Academic prototype — synthetic demo data only. Not for clinical use.")

with st.sidebar:
    st.header("Patient Data")

    age = st.number_input("Age", 18, 100, 65)
    length_of_stay = st.number_input("Length of stay", 1, 60, 7)
    prior_admissions = st.number_input("Prior admissions", 0, 20, 3)
    medication_count = st.number_input("Medication count", 0, 30, 8)
    comorbidity_count = st.number_input("Comorbidity count", 0, 15, 3)

    st.subheader("Vitals")
    heart_rate = st.number_input("Heart rate", 40, 200, 92)
    systolic_bp = st.number_input("Systolic BP", 70, 220, 120)
    oxygen_saturation = st.number_input("Oxygen saturation", 70.0, 100.0, 94.0)
    creatinine = st.number_input("Creatinine", 0.1, 10.0, 1.5)
    glucose = st.number_input("Glucose", 40.0, 500.0, 150.0)
    hemoglobin = st.number_input("Hemoglobin", 4.0, 20.0, 12.0)

    st.subheader("Conditions")
    has_diabetes = int(st.checkbox("Diabetes"))
    has_copd = int(st.checkbox("COPD"))
    has_heart_failure = int(st.checkbox("Heart failure"))
    has_renal_disease = int(st.checkbox("Renal disease"))

    predict_clicked = st.button("Predict readmission risk", type="primary")

payload = {
    "age": age,
    "length_of_stay": length_of_stay,
    "prior_admissions": prior_admissions,
    "medication_count": medication_count,
    "comorbidity_count": comorbidity_count,
    "heart_rate": heart_rate,
    "systolic_bp": systolic_bp,
    "oxygen_saturation": oxygen_saturation,
    "creatinine": creatinine,
    "glucose": glucose,
    "hemoglobin": hemoglobin,
    "has_diabetes": has_diabetes,
    "has_copd": has_copd,
    "has_heart_failure": has_heart_failure,
    "has_renal_disease": has_renal_disease,
}

if predict_clicked:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)

        if response.status_code != 200:
            st.error(response.text)
        else:
            result = response.json()
            probability = result["risk_probability"]

            c1, c2 = st.columns(2)
            c1.metric("30-day readmission risk", f"{probability:.1%}")
            c2.metric("Risk band", result["risk_band"])

            st.subheader("Top contributing factors")

            factors = pd.DataFrame(result["top_factors"])
            if not factors.empty:
                factors["label"] = (
                    factors["feature"]
                    + " — "
                    + factors["direction"]
                )

                fig = px.bar(
                    factors,
                    x="impact",
                    y="feature",
                    orientation="h",
                    title="SHAP feature contributions",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    factors[["feature", "value", "impact", "direction"]],
                    use_container_width=True,
                    hide_index=True,
                )

            st.info(
                "Prototype interpretation only. Any suggested intervention or "
                "clinical decision must be made by qualified healthcare professionals."
            )

    except requests.RequestException:
        st.error(
            "Could not connect to the FastAPI server. Start it with:\n\n"
            "`uvicorn src.api.main:app --reload`"
        )
else:
    st.info("Enter patient values in the sidebar and click Predict.")
