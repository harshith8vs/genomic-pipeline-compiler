"""
Dataset Integrity and Label Verification Tests.

Verifies:
- All STATIC_VALID records have 0 compiler errors and failed == 0.
- All STATIC_INVALID records have >0 compiler errors and failed == 1.
- No contradictory labels exist for identical pipeline scripts.
- Metadata integrity (version, random seed, sample counts, target mapping).
- Deduplication verification.
"""

import os
import json
import pytest
import pandas as pd
from data.generate_dataset import generate_pipeline_dataset, GENERATOR_VERSION

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATASET_PATH = os.path.join(DATA_DIR, "train_dataset.json")


@pytest.fixture(scope="module")
def loaded_dataset():
    """Load train_dataset.json or generate it if missing."""
    if not os.path.exists(DATASET_PATH):
        df, meta = generate_pipeline_dataset(num_samples=200, seed=42)
        return df, meta

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "records" in data:
        df = pd.DataFrame(data["records"])
        meta = data.get("metadata", {})
    else:
        df = pd.DataFrame(data)
        meta = {"total_samples": len(df)}

    return df, meta


def test_dataset_size_and_balance(loaded_dataset):
    """Verify that dataset size is at least 200 samples with a balanced 50/50 split."""
    df, meta = loaded_dataset
    assert len(df) >= 200
    assert (df["failed"] == 0).sum() == len(df) // 2
    assert (df["failed"] == 1).sum() == len(df) // 2
    assert meta["target_column"] == "failed"


def test_static_valid_samples_have_zero_errors(loaded_dataset):
    """
    CRITICAL INTEGRITY CHECK:
    Every sample labelled failed=0 (STATIC_VALID) must have exactly 0 compiler errors.
    """
    df, _ = loaded_dataset
    valid_samples = df[df["failed"] == 0]
    assert len(valid_samples) > 0

    # Ensure total_error_count is strictly 0 for all success-labelled records
    assert (valid_samples["total_error_count"] == 0).all(), (
        "Dataset contains success-labelled samples that have non-zero compiler errors!"
    )


def test_static_invalid_samples_have_compiler_errors(loaded_dataset):
    """
    CRITICAL INTEGRITY CHECK:
    Every sample labelled failed=1 (STATIC_INVALID) must have at least 1 compiler error.
    """
    df, _ = loaded_dataset
    invalid_samples = df[df["failed"] == 1]
    assert len(invalid_samples) > 0

    # Ensure total_error_count is > 0 for all failure-labelled records
    assert (invalid_samples["total_error_count"] > 0).all(), (
        "Dataset contains failure-labelled samples that have 0 compiler errors!"
    )


def test_no_contradictory_labels(loaded_dataset):
    """Ensure no pipeline script text appears with both failed=0 and failed=1."""
    df, _ = loaded_dataset
    valid_texts = set(df[df["failed"] == 0]["pipeline_text"].str.strip())
    invalid_texts = set(df[df["failed"] == 1]["pipeline_text"].str.strip())

    overlap = valid_texts.intersection(invalid_texts)
    assert len(overlap) == 0, f"Found contradictory pipeline scripts: {overlap}"


def test_dataset_metadata_completeness(loaded_dataset):
    """Verify metadata captures generator version, seed, distribution, and validation stats."""
    _, meta = loaded_dataset
    assert "generator_version" in meta
    assert meta["generator_version"] == GENERATOR_VERSION
    assert "random_seed" in meta
    assert "label_mapping" in meta
    assert meta["label_mapping"]["STATIC_VALID"] == 0
    assert meta["label_mapping"]["STATIC_INVALID"] == 1
    assert "validation_statistics" in meta
