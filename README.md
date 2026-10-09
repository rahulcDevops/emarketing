# Emarketing
Production-oriented MLOps for customer-intent classification.

Emarketing is an NLP platform that classifies customer emails and feedback using a Scikit-Learn pipeline. It combines reproducible data, ephemeral Kubernetes training, MLflow quality gates, and GitOps delivery to safely promote better models without requiring persistent training infrastructure.

## Architecture

    Git push
       │
       ▼
    GitHub Actions
    • hydrate DVC dataset
    • build training image
       │
       ▼
    Ephemeral Kind cluster
    • sideload image with kind load docker-image
    • run resource-limited Kubernetes Job
       │
       ▼
    MLflow quality gate
    • Macro-F1 ≥ 0.85
    • regression ≤ 1% vs Champion
       │
       ├── Fail → stop pipeline; production unchanged
       │
       ▼
    GitOps commit
    • update serving image tag
    • commit to main with [skip ci]
       │
       ▼
    Argo CD sync
       │
       ▼
    KServe InferenceService
    • FastAPI predictor
    • autoscaling: 1–5 replicas
    • production model access via service account

## Highlights

- **Zero-cost continuous training** — Kubernetes training runs on an ephemeral Kind cluster inside GitHub Actions, avoiding always-on cloud compute.
- **Data versioning** — DVC tracks dataset pointers while CI hydrates the required `.parquet` data at runtime.
- **Governed model promotion** — MLflow evaluates every challenger against the active champion before release.
- **Safe production delivery** — CI commits declarative image changes; Argo CD pulls and reconciles them into the cluster.
- **Serverless inference** — KServe manages the FastAPI predictor with concurrency-based autoscaling from one to five replicas.

## Model governance

A candidate model is promoted only when it satisfies both release gates:

| Gate | Requirement |
| --- | --- |
| Absolute quality floor | Macro-F1 `>= 0.85` |
| Degradation guard | Score drop `<= 1.0%` versus the active Champion |

Passing challengers receive the MLflow `@champion` alias, while the previous champion is archived. Failed candidates retain their experiment data, but the pipeline exits with an error and does not alter production.

## Repository layout

    .
    ├── .github/workflows/
    │   └── train-on-k8s.yml           # CI training, validation, and GitOps workflow
    ├── api/
    │   ├── main.py                    # FastAPI inference application
    │   └── schemas.py                 # API request and response models
    ├── training/
    │   ├── dataset.py                 # Data validation and ingestion
    │   ├── governance.py              # MLflow Champion/Challenger promotion
    │   ├── publisher.py               # Versioned model-artifact publishing
    │   ├── retrain.py                 # End-to-end retraining workflow
    │   └── train.py                   # Scikit-Learn training pipeline
    ├── k8s/
    │   ├── argocd/
    │   │   └── application.yaml       # Argo CD GitOps application
    │   ├── serving/
    │   │   ├── configmap.yaml         # Serving configuration
    │   │   ├── deployment.yaml        # Kubernetes API deployment manifest
    │   │   ├── inferenceservice.yaml  # KServe production inference service
    │   │   └── service.yaml           # API service definition
    │   ├── mlflow-deployment.yaml     # MLflow tracking server
    │   ├── postgres-statefulset.yaml  # Persistent MLflow metadata store
    │   ├── training-job.yaml.template # Base Kubernetes training Job
    │   └── training-job.yaml.tmpl     # CI-parameterized training Job
    ├── monitoring/                    # Prometheus configuration
    ├── ui/                            # Streamlit feedback interface
    ├── docker-compose.yml             # Local development stack
    └── requirements.txt               # Python dependencies

## Example retraining run

    [CI] Creating ephemeral Kind cluster...
    [CI] Loading training image into Kind...
    [DVC] Hydrating data/emails.parquet

    [K8s] Starting training Job...
    [TRAIN] Loaded 18,420 labeled messages
    [TRAIN] Challenger Macro-F1: 0.8924

    [MLflow] Champion Macro-F1:   0.8871
    [MLflow] Quality floor:       0.8500  PASS
    [MLflow] Regression check:    +0.60%  PASS
    [MLflow] Promoting challenger to @champion

    [GitOps] Updating k8s/serving/inferenceservice.yaml
    [GitOps] Commit created with [skip ci]
    [Argo CD] Reconciling production InferenceService

## Quick start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- DVC
- kubectl and Kind for local Kubernetes validation
- Access to Kubernetes, Argo CD, and KServe for production deployment

### Run locally

    git clone https://github.com/<your-org>/emarketing.git
    cd emarketing

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    docker compose up --build -d

Stop the local environment:

    docker compose down

### Run retraining locally

    dvc pull
    python -m training.retrain

## Delivery and rollback

Argo CD continuously watches the serving manifests in Git and reconciles the desired state into the production cluster. This provides deployment history, drift correction, and a simple rollback path:

    git revert <deployment-commit>
    git push origin main

Argo CD detects the reverted manifest and restores the previous KServe deployment state.