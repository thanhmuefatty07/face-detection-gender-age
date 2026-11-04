import os
import subprocess
import sys
from pathlib import Path

import pytest


from scripts.kaggle_pipeline import audit_and_sync_manifest as audit
from scripts.kaggle_pipeline import kaggle_identity_split as identity_split


REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def test_kaggle_runtime_markers_exist_without_reading_token_values(monkeypatch):
    if KAGGLE_INPUT.is_dir() and KAGGLE_WORKING.is_dir() and "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        assert os.environ.get("KAGGLE_KERNEL_RUN_TYPE") in {"Interactive", "Batch"}
        assert "KAGGLE_USER_SECRETS_TOKEN" in os.environ
        assert audit._is_kaggle_runtime()
        assert identity_split._is_kaggle_runtime()
    else:
        assert not audit._is_kaggle_runtime()
        assert not identity_split._is_kaggle_runtime()
        actual_is_dir = Path.is_dir

        def with_kaggle_mounts(path):
            if path in {KAGGLE_INPUT, KAGGLE_WORKING}:
                return True
            return actual_is_dir(path)

        monkeypatch.setattr(Path, "is_dir", with_kaggle_mounts)
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
        monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", "synthetic-token")
        assert audit._is_kaggle_runtime()
        assert identity_split._is_kaggle_runtime()


@pytest.mark.parametrize("missing_marker", ["KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_USER_SECRETS_TOKEN"])
@pytest.mark.parametrize("entrypoint", ["audit", "identity_split"])
def test_cli_rejects_unmarked_runtime_when_kaggle_directories_exist(
    tmp_path, monkeypatch, missing_marker, entrypoint
):
    actual_is_dir = Path.is_dir

    def with_kaggle_mounts(path):
        if path in {KAGGLE_INPUT, KAGGLE_WORKING}:
            return True
        return actual_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", with_kaggle_mounts)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", "synthetic-token")
    monkeypatch.delenv(missing_marker, raising=False)
    assert KAGGLE_INPUT.is_dir()
    assert KAGGLE_WORKING.is_dir()
    assert not audit._is_kaggle_runtime()
    assert not identity_split._is_kaggle_runtime()

    env = os.environ.copy()
    env.pop(missing_marker, None)

    if entrypoint == "audit":
        report = tmp_path / "must_not_be_written.md"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "kaggle_pipeline" / "audit_and_sync_manifest.py"),
            "--manifest",
            str(tmp_path / "missing-synthetic-manifest.csv"),
            "--report",
            str(report),
        ]
    else:
        report = KAGGLE_WORKING / "identity_grouped_dataset" / "curation_summary.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "kaggle_pipeline" / "kaggle_identity_split.py"),
        ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "Kaggle-only" in result.stderr
    assert not report.exists()

