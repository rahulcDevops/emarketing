"""
api/main.py
LeadSentry: High-performance inbound email intent routing API
with Champion / Challenger (Shadow & Canary) deployment capabilities.
"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
import mlflow
import numpy as np
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from api.db import init_db, save_feedback
from api.logger import setup_logger
from api.security import calculate_sha256
from training.governance import (
    get_model_version_by_alias,
    REGISTERED_MODEL_NAME,
    ALIAS_CHAMPION,
    ALIAS_CHALLENGER,
    DATA_DIR,
)

logger = setup_logger("leadsentry-api")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"

# Model runtime storage
model_state: Dict[str, Any] = {
    "champion": None,
    "champion_metadata": {},
    "challenger": None,
    "challenger_metadata": {},
    "sha256": None,
}

ACTION_DISPATCH = {
    "INTERESTED_DEMO": {"priority": "HIGH", "action": "ESCALATE_SLACK_CRM"},
    "HARD_UNSUBSCRIBE": {"priority": "CRITICAL", "action": "SUPPRESS_DOMAIN_BLACKLIST"},
    "OUT_OF_OFFICE": {"priority": "LOW", "action": "AUTO_SNOOZE_14_DAYS"},
    "NOT_INTERESTED": {"priority": "LOW", "action": "ARCHIVE_COLD_LEAD"},
    "WRONG_PERSON": {"priority": "MEDIUM", "action": "FLAG_FOR_NEW_CONTACT"},
    "OBJECTION_PRICING": {"priority": "MEDIUM", "action": "ROUTE_OBJECTION_PLAYBOOK"},
}

PREDICTIONS_COUNTER = Counter(
    "leadsentry_predictions_total",
    "Total inbound email intent predictions",
    ["category", "model_version", "deployment_variant"],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "leadsentry_prediction_confidence",
    "Distribution of classification confidence scores",
    ["model_version", "deployment_variant"],
    buckets=[0.2, 0.4, 0.6, 0.75, 0.85, 0.95, 1.0],
)

SHADOW_DISCREPANCIES = Counter(
    "leadsentry_shadow_discrepancies_total",
    "Discrepancies where Champion and Challenger disagree on intent",
    ["champion_category", "challenger_category"],
)


def load_model_from_registry(model_name: str, alias: str):
    """Loads a model directly from MLflow tracking store by its alias."""
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.sklearn.load_model(model_uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        logger.error("Base artifacts missing on disk.")
        raise RuntimeError("Model artifacts missing. Run training pipeline first.")

    computed_hash = calculate_sha256(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 1. Load Champion
    champ_version = get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
    try:
        model_state["champion"] = load_model_from_registry(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
        model_state["champion_metadata"] = {
            "version": champ_version or metadata.get("model_version", "v1"),
            "alias": "champion",
        }
        logger.info(f"Loaded @champion model v{champ_version} into memory.")
    except Exception as exc:
        logger.warning(f"Could not load @champion from MLflow URI ({exc}). Using disk fallback.")
        import joblib
        model_state["champion"] = joblib.load(MODEL_PATH)
        model_state["champion_metadata"] = {"version": metadata.get("model_version", "v1"), "alias": "champion"}

    # 2. Check and Load Challenger (if present)
    challenger_version = get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER)
    if challenger_version:
        try:
            model_state["challenger"] = load_model_from_registry(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER)
            model_state["challenger_metadata"] = {
                "version": challenger_version,
                "alias": "challenger",
            }
            logger.info(f"Loaded @challenger model v{challenger_version} into memory for shadow/canary.")
        except Exception as exc:
            logger.warning(f"Failed to load @challenger model: {exc}")
            model_state["challenger"] = None
    else:
        logger.info("No active @challenger registered. Operating in single-champion mode.")

    model_state["sha256"] = computed_hash
    init_db()
    yield
    model_state.clear()


app = FastAPI(
    title="LeadSentry Email Routing Engine",
    version="1.1.0",
    lifespan=lifespan,
)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")


class TracingAndLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = correlation_id

        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "correlation_id": correlation_id,
                "duration_ms": duration_ms,
                "status_code": response.status_code,
            },
        )
        return response


app.add_middleware(TracingAndLoggingMiddleware)


class EmailRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10000)


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    priority: str
    action: str
    model_version: str
    deployment_variant: str
    artifact_hash: str


class HealthResponse(BaseModel):
    status: str
    champion_version: str
    challenger_version: Optional[str]
    artifact_hash: str
    classes: List[str]


class FeedbackRequest(BaseModel):
    ticket_text: str = Field(..., min_length=3, max_length=10000)
    predicted_category: str
    corrected_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    is_correct: bool


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


def _predict_with_pipeline(pipeline, text: str):
    cleaned = text.strip()
    probabilities = pipeline.predict_proba([cleaned])[0]
    max_idx = int(np.argmax(probabilities))
    category = str(pipeline.classes_[max_idx])
    confidence = round(float(probabilities[max_idx]), 4)
    return category, confidence


def run_shadow_scoring(text: str, champ_category: str, champ_conf: float):
    """Executes challenger inference asynchronously in background."""
    challenger = model_state.get("challenger")
    if not challenger:
        return

    try:
        chal_category, chal_conf = _predict_with_pipeline(challenger, text)
        version = model_state["challenger_metadata"]["version"]
        PREDICTIONS_COUNTER.labels(category=chal_category, model_version=version, deployment_variant="shadow").inc()
        CONFIDENCE_HISTOGRAM.labels(model_version=version, deployment_variant="shadow").observe(chal_conf)

        if chal_category != champ_category:
            SHADOW_DISCREPANCIES.labels(champion_category=champ_category, challenger_category=chal_category).inc()
            logger.info(
                f"[Shadow Discrepancy] Champ: {champ_category} ({champ_conf:.2f}) vs Chal: {chal_category} ({chal_conf:.2f})"
            )
    except Exception as e:
        logger.error(f"Shadow scoring failed: {e}")


@app.get("/health", response_model=HealthResponse)
def health_check():
    champ = model_state.get("champion")
    if champ is None:
        raise HTTPException(status_code=503, detail="Champion model unavailable")

    classes = list(champ.classes_) if hasattr(champ, "classes_") else []
    return HealthResponse(
        status="healthy",
        champion_version=str(model_state["champion_metadata"].get("version", "unknown")),
        challenger_version=model_state["challenger_metadata"].get("version"),
        artifact_hash=model_state["sha256"][:12] if model_state["sha256"] else "unknown",
        classes=classes,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_intent(
    email: EmailRequest,
    background_tasks: BackgroundTasks,
    x_model_variant: Optional[str] = Header(default=None, alias="X-Model-Variant"),
):
    champion = model_state.get("champion")
    challenger = model_state.get("challenger")

    if champion is None:
        raise HTTPException(status_code=503, detail="Champion model unavailable")

    # Canary routing: If requested via header and challenger is present
    if x_model_variant == "challenger" and challenger is not None:
        category, confidence = _predict_with_pipeline(challenger, email.text)
        version = model_state["challenger_metadata"]["version"]
        variant = "canary"
    else:
        # Standard route: Champion
        category, confidence = _predict_with_pipeline(champion, email.text)
        version = model_state["champion_metadata"]["version"]
        variant = "champion"

        # If a challenger exists, trigger shadow scoring asynchronously
        if challenger is not None:
            background_tasks.add_task(run_shadow_scoring, email.text, category, confidence)

    dispatch = ACTION_DISPATCH.get(category, {"priority": "MEDIUM", "action": "MANUAL_REVIEW"})

    PREDICTIONS_COUNTER.labels(category=category, model_version=version, deployment_variant=variant).inc()
    CONFIDENCE_HISTOGRAM.labels(model_version=version, deployment_variant=variant).observe(confidence)

    return PredictionResponse(
        category=category,
        confidence=confidence,
        priority=dispatch["priority"],
        action=dispatch["action"],
        model_version=f"v{version}" if not str(version).startswith("v") else str(version),
        deployment_variant=variant,
        artifact_hash=model_state["sha256"][:12] if model_state["sha256"] else "unknown",
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest):
    feedback_id = save_feedback(
        ticket_text=feedback.ticket_text,
        predicted_category=feedback.predicted_category,
        corrected_category=feedback.corrected_category,
        confidence=feedback.confidence,
        model_version=feedback.model_version,
        is_correct=feedback.is_correct,
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)
