"""
tests/test_training.py
Fast unit tests to ensure pipeline integrity and serialization compatibility.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
from training.train import build_pipeline


def test_pipeline_fit_predict(tmp_path: Path):
    # Minimal synthetic dataset
    train_df = pd.DataFrame(
        {
            "text": [
                "I lost my card",
                "Where is my credit card?",
                "Transfer money abroad",
                "Wire cash overseas",
            ],
            "label_text": ["lost_card", "lost_card", "transfer", "transfer"],
        }
    )

    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["label_text"])

    # Prediction checks
    sample = ["I need to cancel my lost card"]
    prediction = pipeline.predict(sample)
    probabilities = pipeline.predict_proba(sample)

    assert len(prediction) == 1
    assert prediction[0] in ["lost_card", "transfer"]
    assert probabilities.shape == (1, 2)

    # Artifact serialization verification
    test_model_file = tmp_path / "model.joblib"
    joblib.dump(pipeline, test_model_file)
    assert test_model_file.exists()

    loaded_pipeline = joblib.load(test_model_file)
    reloaded_pred = loaded_pipeline.predict(sample)
    assert reloaded_pred[0] == prediction[0]
import sqlite3
from training.retrain import load_feedback_data


def test_load_feedback_data(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "feedback.sqlite"
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY,
                ticket_text TEXT,
                corrected_category TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO feedback (ticket_text, corrected_category) VALUES ('Test query', 'test_intent')"
        )
        conn.commit()

    # Point DB_PATH in retrain module to the temporary SQLite file
    monkeypatch.setattr("training.retrain.DB_PATH", test_db)

    df = load_feedback_data()
    assert len(df) == 1
    assert list(df.columns) == ["text", "label_text"]
    assert df.iloc[0]["text"] == "Test query"
    assert df.iloc[0]["label_text"] == "test_intent"
