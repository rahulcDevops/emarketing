# Support Triage

Support Triage is a machine learning service that automates customer-support ticket classification. When a user submits ticket text, the system predicts the issue category along with an associated confidence score.

The project is structured to evolve into a full-lifecycle MLOps portfolio project.

---

## Project Goal

Automate triage workflows in customer service by routing incoming tickets to appropriate teams (e.g., Billing, Technical Support, Account Access, Hardware) using an NLP text classifier and delivering low-latency inference with confidence estimations.

---

## Planned Architecture

```
                       +-----------------------+
                       |  Streamlit Web UI     |
                       |       (ui/)           |
                       +-----------+-----------+
                                   |
                             HTTP / REST
                                   |
                                   v
                       +-----------------------+
                       |   FastAPI Service     |
                       |       (api/)          |
                       +-----------+-----------+
                                   |
                             Loads Artifacts
                                   |
                                   v
                       +-----------------------+
                       | Serialized Pipeline   |
                       |    (artifacts/)       |
                       +-----------+-----------+
                                   ^
                              Trained By
                                   |
+---------------------+ +----------------------+
|  Raw / Processed    | | Training Pipeline    |
|      (data/)        | |    (training/)       |
+---------------------+ +----------------------+
```

1. **Training (`training/`):** Data loading, text cleaning, vectorization/feature extraction, and model training routines. Saves serialized model pipelines into `artifacts/`.
2. **Inference API (`api/`):** A lightweight FastAPI application with Pydantic request/response schemas to serve real-time predictions and health checks.
3. **User Interface (`ui/`):** An interactive Streamlit dashboard allowing users to input ticket text and view predicted categories and confidence scores.
4. **Testing (`tests/`):** Unit and integration tests covering model pipelines, schema validations, and API endpoints using `pytest` and `httpx`.
5. **Storage (`data/` & `artifacts/`):** Local directories for dataset samples and exported model binaries (e.g., `.joblib` files).

---

## Day 1 Scope

The current stage establishes the foundation and project layout only:
- Python 3.11+ directory structure (`api`, `ui`, `training`, `tests`, `data`, `artifacts`).
- Package markers (`__init__.py`) where applicable.
- Clean dependency definitions in `requirements.txt`.
- `.gitignore` configured for Python virtual environments, OS/IDE files, local datasets, and binary model artifacts.
- No model training, API routes, or production orchestration tooling (Docker, MLflow, CI/CD, DB, K8s) are implemented yet.

---

## Prerequisites

- Python 3.11 or higher
- `pip` or preferred virtual environment manager (`venv`)

---

## Setup

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install initial dependencies
pip install -r requirements.txt
```
