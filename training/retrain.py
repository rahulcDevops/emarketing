"""
training/retrain.py
Continuous retraining with anti-poisoning sanitization,
strict evaluation gates, MLflow @champion/@challenger governance,
and optional S3 model publishing (dual local / prod support).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from training.dataset import generate_email_dataset
from training.governance import (
    register_challenger,
    promote_challenger_to_champion,
    get_model_version_by_alias,
    print_registry_status,
    REGISTERED_MODEL_NAME,
    ALIAS_CHAMPION,
)
from training.train import (
    build_pipeline,
    setup_mlflow,
)

# Optional S3 publisher import (only invoked if MODEL_S3_BUCKET is set)
try:
    from training.publisher import S3ModelPublisher
except ImportError:
    S3ModelPublisher = None

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "feedback.sqlite"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

# Ensure local directories exist (crucial for ephemeral containers)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_feedback_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes human-in-the-loop feedback to defend against data poisoning:
    - Drops empty/null text and labels
    - Rejects degenerate character lengths (< 5 or > 5000 characters)
    - Deduplicates identical ticket text to prevent replay bias
    """
    if raw_df.empty:
        return raw_df

    initial_count = len(raw_df)

    df = raw_df.dropna(subset=["text", "label_text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label_text"] = df["label_text"].astype(str).str.strip()

    # Length guard
    df = df[(df["text"].str.len() >= 5) & (df["text"].str.len() <= 5000)]

    # Exact string deduplication
    df = df.drop_duplicates(subset=["text"])

    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"[Sanitization Guard] Filtered out {dropped} poisoned/duplicate feedback records.")

    return df


def load_feedback_data() -> pd.DataFrame:
    """Loads feedback from SQLite if present, otherwise returns empty DataFrame."""
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["text", "label_text"])

    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT ticket_text AS text, corrected_category AS label_text
        FROM feedback
        WHERE ticket_text IS NOT NULL AND corrected_category IS NOT NULL
        """
        raw_df = pd.read_sql_query(query, conn)

    return sanitize_feedback_data(raw_df)


def get_champion_f1_baseline(client: mlflow.tracking.MlflowClient, champion_version: str) -> float:
    """
    Retrieves the champion's baseline Macro-F1 score.
    Prefers reading from local metadata.json; falls back to MLflow Run metrics for prod/K8s environments.
    """
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                champion_metadata = json.load(f)
            return float(champion_metadata["metrics"]["macro_f1"])
        except (KeyError, json.JSONDecodeError):
            pass

    # Fallback to MLflow Run tracking
    model_version_details = client.get_model_version(
        name=REGISTERED_MODEL_NAME, version=champion_version
    )
    run = client.get_run(model_version_details.run_id)
    return float(run.data.metrics.get("macro_f1", 0.85))


def retrain_and_gate() -> None:
    tracking_uri = setup_mlflow()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # 1. Fetch current Champion metadata
    champion_version = get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
    if not champion_version:
        raise RuntimeError("No current @champion found in MLflow. Run training.train first.")

    champion_f1 = get_champion_f1_baseline(client, champion_version)
    print(f"[Retrain] Current active @champion is Version {champion_version} (Macro-F1: {champion_f1:.4f})")

    # 2. Ingest base dataset + sanitized feedback
    train_df, test_df, label_names = generate_email_dataset()
    feedback_df = load_feedback_data()
    print(f"[Retrain] Merging {len(feedback_df)} production feedback corrections into training set...")

    if not feedback_df.empty:
        combined_train = pd.concat([train_df[["text", "label_text"]], feedback_df], ignore_index=True)
    else:
        combined_train = train_df[["text", "label_text"]]

    X_train, y_train = combined_train["text"], combined_train["label_text"]
    X_test, y_test = test_df["text"], test_df["label_text"]

    candidate_pipeline = build_pipeline(max_features=8000, c_param=1.2)

    with mlflow.start_run(run_name="retrain-challenger-candidate") as run:
        candidate_run_id = run.info.run_id

        candidate_pipeline.fit(X_train, y_train)
        y_pred = candidate_pipeline.predict(X_test)

        candidate_acc = float(accuracy_score(y_test, y_pred))
        candidate_f1 = float(f1_score(y_test, y_pred, average="macro"))

        mlflow.log_metric("accuracy", round(candidate_acc, 4))
        mlflow.log_metric("macro_f1", round(candidate_f1, 4))
        mlflow.log_metric("champion_baseline_f1", round(champion_f1, 4))

        sig = infer_signature(
            pd.DataFrame({"text": X_test.iloc[:5].tolist()}),
            pd.DataFrame({"category": y_pred[:5]}),
        )

        mlflow.sklearn.log_model(
            sk_model=candidate_pipeline,
            name="model",
            signature=sig,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Retrieve new version number using the configured client
        reg_model = client.get_registered_model(REGISTERED_MODEL_NAME)
        challenger_version = reg_model.latest_versions[-1].version

        # Register candidate as @challenger
        register_challenger(REGISTERED_MODEL_NAME, challenger_version)

        # 3. Quality Gate Evaluation
        passed_f1_gate = candidate_f1 >= (champion_f1 - 0.01)
        passed_floor_gate = candidate_f1 >= 0.85
        passed_promotion_gate = passed_f1_gate and passed_floor_gate

        mlflow.log_param("passed_promotion_gate", passed_promotion_gate)

        print("\n----------------- EVALUATION GATE REPORT -----------------")
        print(f" • Champion (v{champion_version}) Macro-F1   : {champion_f1:.4f}")
        print(f" • Challenger (v{challenger_version}) Macro-F1 : {candidate_f1:.4f}")
        print(f" • Relative Tolerance Check (>= -0.01) : {'PASS' if passed_f1_gate else 'FAIL'}")
        print(f" • Absolute Quality Floor Check (>= 0.85) : {'PASS' if passed_floor_gate else 'FAIL'}")
        print("----------------------------------------------------------\n")

        if passed_promotion_gate:
            promote_challenger_to_champion(REGISTERED_MODEL_NAME, challenger_version)
            joblib.dump(candidate_pipeline, MODEL_PATH)

            new_metadata = {
                "model_version": f"v{challenger_version}",
                "registry_version": challenger_version,
                "alias": "champion",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": {"accuracy": round(candidate_acc, 4), "macro_f1": round(candidate_f1, 4)},
                "num_classes": len(label_names),
                "classes": label_names,
            }
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(new_metadata, f, indent=2)

            print(f"[Gate PASSED] Candidate v{challenger_version} promoted to active @champion.")

            # 4. Optional S3 Publishing (Triggered only if S3 bucket environment variable is defined)
            s3_bucket = os.getenv("MODEL_S3_BUCKET")
            if s3_bucket:
                if S3ModelPublisher is None:
                    raise ImportError("boto3 / S3ModelPublisher missing. Ensure training/publisher.py exists.")
                
                print(f"[Production S3] Publishing model bundle to bucket: {s3_bucket}...")
                publisher = S3ModelPublisher(bucket_name=s3_bucket)
                s3_uri = publisher.package_and_publish(
                    model_path=str(MODEL_PATH),
                    metrics={"accuracy": candidate_acc, "macro_f1": candidate_f1},
                    classes=label_names,
                    run_id=candidate_run_id,
                )
                print(f"[Production S3] Artifact successfully uploaded: {s3_uri}")
            else:
                print("[Local Mode] MODEL_S3_BUCKET not set. Model saved locally at artifacts/model.joblib.")
        else:
            print(f"[Gate REJECTED] Candidate v{challenger_version} did not meet SLA. @champion unchanged.")
            # If running in CI/CD pipeline, fail non-zero so GitHub Actions catches the SLA drop
            if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
                raise RuntimeError("Candidate model failed governance quality gate. Halting CI/CD promotion.")

        print_registry_status(REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    retrain_and_gate()
