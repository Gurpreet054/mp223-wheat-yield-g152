
# Group G152 — MP223

## Problem: Wheat Yield per Acre (Guwahati)

**Client:** Guwahati Kisan Producer Company  
**Course:** SCSE3040 MLOps | **Batch:** 63

---

### Team Members

| Enrolment No. | Name | Role |
| :--- | :--- | :--- |
| **S24CSEU2313** | Gurpreet Nagar | Data Lead |
| **S24CSEU2594** | Rhea Mathur | Ops Lead |
| **S24CSEU2241** | Ojas Vats | |
| **S24CSEU2199** | Pratham Raj | |

---

### Project Overview

An end-to-end MLOps pipeline for predicting wheat yield per acre (`yield_qtl_per_acre`) for Guwahati Kisan Producer Company using agronomic, weather, and regional features (`rainfall_mm`, `fertilizer_kg`, `sowing_delay_days`, `irrigation_count`, `soil_ph`, `farm_acres`, `seed_variety`, and `tehsil`).

* **Data Versioning:** DVC (`data/raw/train.csv.dvc`, `data/raw/later.csv.dvc`)
* **Data Quality & Cleaning:** Automated audit and remediation of 7 data anomalies in `src/mp223/data.py`, documented in `reports/findings.yaml`
* **Experiment Tracking & Registry:** MLflow (`mlflow.db`) with `@champion` model promotion
* **Prediction Service:** FastAPI (`app/main.py`) with Pydantic validation, `pytest` suite, Docker, and Kubernetes manifests
* **Monitoring & Drift:** PSI drift detection, group MAE by `tehsil`, and automated v2 retraining (`src/mp223/monitor.py`)

---

### Setup for Teammates

1. Clone the repository and install dependencies: `pip install -r requirements.txt`
2. Place the original `train.csv` and `later.csv` inside `data/raw/` (do **not** edit/save them in Excel, and do **not** force-add them to Git).
3. Import `clean_train_data()` from `src/mp223/data.py` to load the cleaned training dataset.
EOF
