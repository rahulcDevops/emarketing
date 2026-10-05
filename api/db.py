"""
api/db.py
SQLite database operations for persisting agent feedback.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "feedback.sqlite"


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dictionary-like row access."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the feedback table schema if it does not already exist."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_text TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                corrected_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            """
        )
        conn.commit()


def save_feedback(
    ticket_text: str,
    predicted_category: str,
    corrected_category: str,
    confidence: float,
    model_version: str,
    is_correct: bool,
) -> int:
    """
    Inserts an agent feedback record into the database.
    Returns: generated row id.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO feedback (
                ticket_text,
                predicted_category,
                corrected_category,
                confidence,
                model_version,
                is_correct,
                created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_text,
                predicted_category,
                corrected_category,
                confidence,
                model_version,
                1 if is_correct else 0,
                created_at,
            ),
        )
        conn.commit()
        return cursor.lastrowid
