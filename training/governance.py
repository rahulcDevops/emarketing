"""
training/governance.py
Enterprise Model Registry Governance for LeadSentry.
Manages @champion, @challenger, and @archived aliases using MLflowClient.
Supports local SQLite and centralized HTTP tracking servers via MLFLOW_TRACKING_URI.
"""

import os
from pathlib import Path
import sys
from typing import Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "leadsentry-intent-model"
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"
ALIAS_ARCHIVED = "archived"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"


def get_client() -> MlflowClient:
    """
    Returns an active MLflow tracking client.
    Priority: MLFLOW_TRACKING_URI environment variable -> local SQLite fallback.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.resolve()}")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def get_model_version_by_alias(model_name: str, alias: str) -> Optional[str]:
    """Retrieves the model version associated with a specific alias."""
    client = get_client()
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
        return model_version.version
    except Exception:
        return None


def register_challenger(model_name: str, version: str) -> None:
    """Tags a newly trained model version as the active @challenger."""
    client = get_client()
    client.set_registered_model_alias(model_name, ALIAS_CHALLENGER, version)
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="deployment_status",
        value="challenger_evaluation",
    )
    print(f"[Governance] Model version {version} tagged as @{ALIAS_CHALLENGER}.")


def promote_challenger_to_champion(model_name: str, candidate_version: str) -> Tuple[str, Optional[str]]:
    """
    Promotes the candidate to @champion.
    Archives the previous champion for deterministic rollback capability.
    Returns: (new_champion_version, previous_champion_version)
    """
    client = get_client()
    previous_champion = get_model_version_by_alias(model_name, ALIAS_CHAMPION)

    # 1. Archive the existing champion if one exists
    if previous_champion and previous_champion != candidate_version:
        client.set_registered_model_alias(model_name, ALIAS_ARCHIVED, previous_champion)
        client.set_model_version_tag(
            name=model_name,
            version=previous_champion,
            key="deployment_status",
            value="archived",
        )
        print(f"[Governance] Previous champion v{previous_champion} moved to @{ALIAS_ARCHIVED}.")

    # 2. Promote candidate to champion
    client.set_registered_model_alias(model_name, ALIAS_CHAMPION, candidate_version)
    try:
        client.delete_registered_model_alias(model_name, ALIAS_CHALLENGER)
    except Exception:
        pass

    client.set_model_version_tag(
        name=model_name,
        version=candidate_version,
        key="deployment_status",
        value="production_champion",
    )
    print(f"[Governance] Model version {candidate_version} promoted to @{ALIAS_CHAMPION}.")
    return candidate_version, previous_champion


def rollback_to_archived(model_name: str = REGISTERED_MODEL_NAME) -> Optional[str]:
    """Emergency automated rollback: Reinstates the @archived model as @champion."""
    client = get_client()
    archived_version = get_model_version_by_alias(model_name, ALIAS_ARCHIVED)

    if not archived_version:
        print("[Governance Error] No archived model found for rollback.")
        return None

    current_champion = get_model_version_by_alias(model_name, ALIAS_CHAMPION)
    if current_champion:
        client.set_model_version_tag(
            name=model_name,
            version=current_champion,
            key="deployment_status",
            value="reverted_incident",
        )

    client.set_registered_model_alias(model_name, ALIAS_CHAMPION, archived_version)
    client.delete_registered_model_alias(model_name, ALIAS_ARCHIVED)
    client.set_model_version_tag(
        name=model_name,
        version=archived_version,
        key="deployment_status",
        value="reinstated_champion",
    )
    print(f"[Governance Rollback] Reinstated v{archived_version} as @{ALIAS_CHAMPION}.")
    return archived_version


def print_registry_status(model_name: str = REGISTERED_MODEL_NAME) -> None:
    """Pretty prints the current status of all registered model aliases."""
    client = get_client()
    print(f"\n================ MODEL REGISTRY: {model_name} ================")
    try:
        reg_model = client.get_registered_model(model_name)
        aliases = reg_model.aliases
        if not aliases:
            print("No aliases currently configured.")
        for alias, version in aliases.items():
            print(f" • @{alias:<12} -> Version {version}")
    except Exception as e:
        print(f"Error fetching registry status: {e}")
    print("=================================================================\n")


if __name__ == "__main__":
    if "--status" in sys.argv:
        print_registry_status()
    elif "--rollback" in sys.argv:
        reverted = rollback_to_archived(REGISTERED_MODEL_NAME)
        if reverted:
            print(f"Successfully rolled back to version {reverted}")
        print_registry_status()
    else:
        print("Usage: python -m training.governance [--status | --rollback]")
