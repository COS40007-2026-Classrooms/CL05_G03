# Zone 1 Real-Time Power Consumption Prediction — CL05_G03

A real-time, regression-based system that predicts **Zone 1 power consumption** in a smart-grid environment and maps the prediction into demand categories (Low, Medium, High, Very High) so grid operators can interpret system state at a glance and react to critical demand.

Built for **COS40007 — Artificial Intelligence Engineering** ( Group CL05_G03).

---

## Overview

The system estimates Zone 1 power demand from environmental variables, time-based features, and real-time **Zone 2 & Zone 3** consumption values. It is a real-time **estimation (nowcasting)** system — it estimates the present state from currently available inputs rather than forecasting the future.

- **Model:** Random Forest Regressor (selected over Linear Regression, Decision Tree, SVR, and MLP)
- **Target:** `zone1_power`
- **Metrics:** MAE, RMSE, R²
- **Categories:** equal-frequency binning into Low / Medium / High / Very High
- **Latest performance:** MAE ≈ 662.5 W, RMSE ≈ 1045.8 W, R² ≈ 0.979 (8,787 test samples, 9 features)

---


## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull DVC-tracked data and artifacts

```bash
dvc pull
```

### 3. Reproduce the full pipeline

```bash
dvc repro
```

This runs all stages in order: `generate_new_data → preprocess → monitor → train → evaluate`.

### 4. Run the prediction app

```bash
streamlit run app.py
```

The Streamlit UI accepts environmental and Zone 2 & 3 inputs, predicts Zone 1 consumption, shows the demand category, and raises an alert for **Very High** demand.

---

## DVC pipeline

| Stage | Script | Output |
|---|---|---|
| `generate_new_data` | `src/generate_new_data.py` | `data/new_data.csv` |
| `preprocess` | `src/preprocess_new_data.py` | `artifacts/processed_train.csv` |
| `monitor` | `src/monitor.py` | drift report, alerts, monitoring log |
| `train` | `src/model.py` | `models/best_model.joblib` |
| `evaluate` | `src/evaluate.py` | `metrics.json`, `metrics.txt`, `model_results.png` |

Key parameters (`params.yaml`): `n_estimators=100`, `random_state=42`, target `zone1_power`, retraining thresholds `min_new_rows=50`, `new_rows=350`.

---

## CI/CD — retraining & monitoring

GitHub Actions (`.github/workflows/train.yml`) automatically retrains and monitors the model. It triggers on:

- **push** to `main` affecting `data/`, `train/`, `test/`, `src/`, `params.yaml`, `dvc.yaml`, or the workflow file
- **workflow_dispatch** — manual run from the Actions UI
- **schedule** — every Sunday at 00:00 UTC

The workflow pulls DVC data from the DagsHub remote, runs `dvc repro`, compares metrics against the previous commit, pushes updated artifacts, and uploads the model, metrics, and monitoring reports as downloadable artifacts.

> Requires repository secrets `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` for the DVC remote.

---

## Monitoring & drift detection

`src/monitor.py` performs:

- **Data drift** — compares incoming feature distributions against training data
- **Concept drift** — watches the input→output relationship via rising MAE/RMSE
- **Performance & data-quality checks** — feeds the retrain decision

Retraining is triggered by a performance drop, detected drift, or accumulation of enough new rows. The update process is **retrain → validate → compare → deploy only if improved**, preventing performance regression.

---

## Team

| Member | Role |
|---|---|
| Vihanga Peiris | Project Lead / ML Engineer / Scrum Master |
| Sithum Sirimanna | Data Engineer / MLOps & DevOps / Frontend |
| Pasindu Balasooriya | Data Engineer / MLOps & DevOps / Report Lead |
| Adriel Subi | MLOps & DevOps / Frontend |

---

## Tools

GitHub (version control) · DVC + DagsHub (data & pipeline versioning) · GitHub Actions (CI/CD) · Streamlit (UI) · Trello (task management) · Confluence (documentation) · Microsoft Teams & WhatsApp (communication)
