"""
tests/test_mlflow.py
Unit tests to ensure MLflow experiment tracking operates properly.
"""

from pathlib import Path
import mlflow
import pytest
from training.train import build_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"


def test_mlflow_local_tracking(tmp_path: Path):
    test_db = tmp_path / "test_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{test_db}")
    mlflow.set_experiment("test-experiment")

    with mlflow.start_run(run_name="unit-test-run") as run:
        mlflow.log_param("test_param", 100)
        mlflow.log_metric("test_metric", 0.95)

    client = mlflow.tracking.MlflowClient()
    logged_run = client.get_run(run.info.run_id)

    assert logged_run.data.params["test_param"] == "100"
    assert logged_run.data.metrics["test_metric"] == 0.95
