# AI-Driven Explainable Hospital Readmission Risk Prediction System

Academic capstone prototype for predicting 30-day hospital readmission risk from EHR-style data and providing patient-level explanations.

> **Important:** This repository is an academic/demo system. It is not a clinical decision-support device and must not be used for real patient care.

## Architecture

```text
EHR-style data
      |
      v
Data validation + preprocessing
      |
      v
Feature engineering
      |
      v
Logistic Regression / XGBoost / LightGBM
      |
      v
Probability calibration
      |
      +------------------+
      |                  |
      v                  v
   SHAP               LIME
      |                  |
      +--------+---------+
               v
        FastAPI backend
               |
               v
       Streamlit dashboard
```

## Team ownership

- Alexes — ML training, evaluation, integration
- Allen — data ingestion, preprocessing, feature engineering
- Athul — SHAP/LIME explainability
- Azharuddin — FastAPI backend
- Lutfi — Streamlit clinician dashboard

## Local setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ai-hospital-readmission-risk
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate demo data

```bash
python scripts/generate_demo_data.py
```

### 5. Train the models

```bash
python scripts/train_models.py
```

This creates model artifacts in `models/`.

### 6. Start the API

```bash
uvicorn src.api.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the dashboard

Open another VS Code terminal:

```bash
streamlit run src/dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

## Project structure

```text
ai-hospital-readmission-risk/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── explainability/
│   ├── features/
│   └── models/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Git workflow

Do not push directly to `main`.

Create a branch:

```bash
git checkout -b feature/your-module
```

Commit:

```bash
git add .
git commit -m "Add preprocessing pipeline"
```

Push:

```bash
git push -u origin feature/your-module
```

Then open a Pull Request on GitHub.

## Dataset

The project is structured so the demo generator can later be replaced with an approved de-identified dataset such as MIMIC-III/IV. Do not commit restricted patient-level datasets to GitHub.

## Current prototype limitations

- Demo data is synthetic.
- Clinical validity has not been established.
- The model is not validated for deployment.
- Suggested actions are illustrative UI content only.
