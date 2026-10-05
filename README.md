Emarketing is a production-oriented MLOps platform for customer-intent classification. It combines automated retraining, model governance, quality gates, immutable artifact releases, and zero-downtime inference deployment.

The platform separates rapid local development from cloud-native training workloads, using Kubernetes Jobs for isolated retraining and MLflow aliases to enforce Champion/Challenger promotion policies.

## Key capabilities

- **Automated retraining** — runs isolated, resource-constrained Kubernetes training Jobs.
- **Model governance** — promotes models through MLflow `champion` and `challenger` aliases.
- **Quality gates** — blocks model releases that fall below absolute or historical quality thresholds.
- **Data protection** — sanitizes feedback data by removing invalid labels, malformed text, and duplicates.
- **Immutable releases** — packages validated model artifacts and metadata into versioned Amazon S3 bundles.
- **Production serving** — exposes predictions through a FastAPI service designed for blue/green rollouts.
- **Observability** — exports API latency, request volume, and prediction-distribution metrics to Prometheus.

## Architecture

```text
Inbound data and user feedback
            │
            ▼
┌───────────────────────────────┐
│ Data Defense Pipeline         │
│ • label validation            │
│ • text-boundary checks        │
│ • deduplication               │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Kubernetes Training Job        │
│ • ephemeral batch/v1 Job      │
│ • CPU and memory limits       │
└───────────────┬───────────────┘
                │
                ├──────────► MLflow Tracking + PostgreSQL
                │              metrics, lineage, and registry
                ▼
┌───────────────────────────────┐
│ Evaluation Gate               │
│ Candidate Macro-F1 must meet: │
│ • absolute floor: ≥ 0.85      │
│ • champion tolerance: ≥ -0.01 │
└───────────────┬───────────────┘
          fail  │  pass
                │
                ▼
┌───────────────────────────────┐
│ Artifact Release Engine       │
│ Versioned .tar.gz bundle      │
│ stored in Amazon S3           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ FastAPI Serving Layer         │
│ Blue/green rollout + dynamic  │
│ artifact retrieval            │
└───────────────────────────────┘

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── train-on-k8s.yml        # CI/CD workflow for EKS training Jobs
├── api/
│   ├── main.py                     # FastAPI application and routes
│   └── schemas.py                  # Pydantic request/response contracts
├── training/
│   ├── dataset.py                  # Data ingestion and schema definitions
│   ├── governance.py               # MLflow alias and promotion management
│   ├── publisher.py                # S3 artifact packaging and publication
│   ├── retrain.py                  # Retraining pipeline and evaluation gate
│   └── train.py                    # Baseline model training
├── k8s/
│   ├── mlflow-deployment.yaml      # MLflow deployment manifest
│   ├── postgres-stateful.yaml      # PostgreSQL StatefulSet
│   └── training-job.yaml.tmpl      # Parameterized Kubernetes Job template
├── monitoring/                     # Prometheus configuration
├── ui/                             # Streamlit feedback and testing interface
├── Dockerfile.api                  # Inference API image
├── Dockerfile.mlflow               # MLflow tracking-server image
├── Dockerfile.training             # Training-job image
├── Dockerfile.ui                   # Streamlit UI image
├── docker-compose.yml              # Local development stack
└── requirements.txt                # Python dependencies
```

## Quick start

### Prerequisites

- Docker and Docker Compose
- Python 3.10 or later
- AWS credentials and Kubernetes access for cloud deployment

### Run locally

Start the complete local development environment:

docker compose up --build -d

Once the services are running:

| Service | URL |
| --- | --- |
| Inference API documentation | `http://localhost:8000/docs` |
| MLflow | `http://localhost:5000` |
| Feedback UI | `http://localhost:8501` |
| Prometheus | `http://localhost:9090` |

Stop the local stack when finished:

docker compose down

## Retraining workflow

Run the retraining pipeline locally:

python -m training.retrain

In production, GitHub Actions builds and publishes the training image to Amazon ECR, then dispatches a resource-governed Kubernetes Job to Amazon EKS.

The retraining lifecycle is:

1. **Sanitize feedback data** by validating labels, removing malformed text, and deduplicating records.
2. **Train a candidate model** in an isolated Kubernetes Job with defined CPU and memory limits.
3. **Track and register** the run in MLflow as a `challenger`.
4. **Evaluate the candidate** against the active `champion`.
5. **Promote and publish** only when the candidate satisfies both release gates:
   - `Macro-F1 >= 0.85`
   - `Candidate Macro-F1 >= Champion Macro-F1 - 0.01`
6. **Reject failed candidates** by preserving their MLflow telemetry while stopping the deployment pipeline with a non-zero exit code.

## Model release contract

Approved models are distributed as immutable, versioned `.tar.gz` bundles in Amazon S3. Each bundle contains the trained model, validation metadata, and the environment contract required for reproducible serving.

The inference service retrieves the approved artifact and can be deployed through blue/green rollout strategies to minimize disruption.

## Monitoring

LeadSentry exposes Prometheus-compatible metrics for:

- Prediction request volume
- Inference latency distributions
- Response outcomes
- Prediction-class distribution drift
