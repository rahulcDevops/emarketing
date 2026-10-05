"""
api/logger.py
Structured JSON logger for production observability.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            log_payload["correlation_id"] = record.correlation_id
        if hasattr(record, "duration_ms"):
            log_payload["duration_ms"] = record.duration_ms
        if hasattr(record, "status_code"):
            log_payload["status_code"] = record.status_code
        return json.dumps(log_payload)


def setup_logger(name: str = "leadsentry-api") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger
