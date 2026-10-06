import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
import boto3


class S3ModelPublisher:
    """Packages and publishes verified model artifacts to AWS S3."""

    def __init__(self, bucket_name: str | None = None):
        self.bucket_name = bucket_name or os.getenv("MODEL_S3_BUCKET")
        self.region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
        self._s3_client = None

    @property
    def s3_client(self):
        if not self._s3_client:
            self._s3_client = boto3.client("s3", region_name=self.region)
        return self._s3_client

    def package_and_publish(
        self,
        model_path: str,
        metrics: dict,
        classes: list[str],
        run_id: str,
    ) -> str:
        if not self.bucket_name:
            raise ValueError("MODEL_S3_BUCKET environment variable is required.")

        metadata = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "classes": classes,
            "metrics": metrics,
            "git_commit": os.getenv("GITHUB_SHA", "local-dev"),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            tarball_name = f"leadsentry-{run_id}.tar.gz"
            tarball_path = os.path.join(temp_dir, tarball_name)

            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(model_path, arcname="model.joblib")
                tar.add(metadata_file, arcname="metadata.json")

            versioned_key = f"models/leadsentry/releases/{tarball_name}"
            champion_key = "models/leadsentry/releases/latest-champion.tar.gz"

            # Mock / Dry-run check for CI validation
            if self.bucket_name in ("mock-local-bucket", "none", "") or not os.getenv("AWS_ACCESS_KEY_ID"):
                print(f"[S3 Publisher] Mock/Offline mode detected for bucket '{self.bucket_name}'.")
                print(f"[S3 Publisher] Verified release bundle packaged locally at: {tarball_path}")
                print(f"[S3 Publisher] S3 upload skipped for ephemeral CI validation.")
                return f"s3://{self.bucket_name}/{versioned_key}"

            print(f"[S3 Publisher] Uploading release to s3://{self.bucket_name}/{versioned_key}")
            self.s3_client.upload_file(tarball_path, self.bucket_name, versioned_key)

            print(f"[S3 Publisher] Updating champion pointer at s3://{self.bucket_name}/{champion_key}")
            self.s3_client.upload_file(tarball_path, self.bucket_name, champion_key)

            return f"s3://{self.bucket_name}/{versioned_key}"
