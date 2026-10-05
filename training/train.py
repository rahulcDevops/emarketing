"""
training/train.py
Trains the baseline model and registers it with the formal @champion alias.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from training.dataset import generate_email_dataset
from training.governance import (
    promote_challenger_to_champion,
    print_registry_status,
    REGISTERED_MODEL_NAME,
)

RANDOM_STATE = 42
MODEL_VERSION_TAG = "v1"
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
MLFLOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "mlflow"
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"
EXPERIMENT_NAME = "leadsentry-email-triage"


def setup_mlflow() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Priority: MLFLOW_TRACKING_URI env var -> local SQLite fallback
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.resolve()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def build_pipeline(max_features: int = 5000, c_param: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    C=c_param,
                ),
            ),
        ]
    )


def train_and_evaluate() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracking_uri = setup_mlflow()
    print(f"[MLflow] Connected to tracking backend: {tracking_uri}")

    train_df, test_df, label_names = generate_email_dataset()

    X_train, y_train = train_df["text"], train_df["label_text"]
    X_test, y_test = test_df["text"], test_df["label_text"]

    with mlflow.start_run(run_name=f"baseline-{MODEL_VERSION_TAG}") as run:
        mlflow.log_param("model_version", MODEL_VERSION_TAG)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("classes", label_names)

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("macro_f1", round(macro_f1, 4))

        sig = infer_signature(
            pd.DataFrame({"text": X_test.iloc[:5].tolist()}),
            pd.DataFrame({"category": y_pred[:5]}),
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=sig,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Retrieve registered version number from MLflow using the active tracking URI
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        reg_model = client.get_registered_model(REGISTERED_MODEL_NAME)
        latest_version = reg_model.latest_versions[-1].version

        # Promote directly to @champion
        promote_challenger_to_champion(REGISTERED_MODEL_NAME, latest_version)

        # Local cache for fast CPU startup
        joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")

        metadata = {
            "model_version": f"v{latest_version}",
            "registry_version": latest_version,
            "alias": "champion",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4)},
            "num_classes": len(label_names),
            "classes": label_names,
        }

        with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Success] LeadSentry v{latest_version} registered as @champion. Macro-F1: {macro_f1:.4f}")
        print_registry_status(REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    train_and_evaluate()
