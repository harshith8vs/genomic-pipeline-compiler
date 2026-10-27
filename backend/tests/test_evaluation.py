"""
Regression and Quality Assurance Tests for ML Evaluation Methodology.

Covers:
1. Group separation (zero template overlap across train and test splits).
2. Automated target leakage detection (evaluating tautological relationship of total_error_count).
3. Reproducible splits with fixed seeds.
4. Feature schema consistency between training, serialization, and inference.
5. Baseline comparison integrity (Deterministic Rule, Majority Class, Ablated GLM, Full GLM).
6. Independent benchmark isolation (OOD test set contains unique templates).
7. Model calibration and metric sanity checks.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from app.features import FEATURE_COLUMNS, validate_feature_dict
from app.model import GLMModelManager, DEFAULT_MODEL_PATH
from app.compiler import PipelineCompiler
from app.validator import PipelineValidator

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TRAIN_DATASET_PATH = os.path.join(DATA_DIR, "train_dataset.json")
INDEP_DATASET_PATH = os.path.join(DATA_DIR, "independent_eval_set.json")


@pytest.fixture(scope="module")
def train_df():
    """Load train_dataset.json."""
    assert os.path.exists(TRAIN_DATASET_PATH), "Training dataset file must exist."
    with open(TRAIN_DATASET_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        return pd.DataFrame(data["records"])
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def independent_data():
    """Load independent_eval_set.json."""
    assert os.path.exists(INDEP_DATASET_PATH), "Independent evaluation set must exist."
    with open(INDEP_DATASET_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def loaded_model():
    """Load trained GLM model artifact."""
    mgr = GLMModelManager(model_path=DEFAULT_MODEL_PATH)
    assert mgr.load_model() is True, "Trained GLM model artifact must load successfully."
    return mgr


def test_group_separation_and_zero_overlap(train_df):
    """
    CRITICAL EVALUATION CHECK:
    Verify that GroupShuffleSplit strictly isolates templates so no template/group crosses splits.
    """
    assert "group_id" in train_df.columns, "Dataset must have group_id column for group-aware splitting."
    groups = train_df["group_id"].values
    X = train_df[FEATURE_COLUMNS]
    y = train_df["failed"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    train_groups = set(train_df.iloc[train_idx]["group_id"].unique())
    test_groups = set(train_df.iloc[test_idx]["group_id"].unique())

    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Group leakage detected! Overlapping groups: {overlap}"
    assert len(test_groups) >= 2, "Test set must contain at least 2 held-out template groups."


def test_reproducible_splitting(train_df):
    """Verify that group-aware splitting is 100% deterministic given the same random seed."""
    groups = train_df["group_id"].values
    X = train_df[FEATURE_COLUMNS]
    y = train_df["failed"].values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    t1, te1 = next(gss1.split(X, y, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    t2, te2 = next(gss2.split(X, y, groups=groups))

    assert np.array_equal(t1, t2), "Splits are not reproducible across identical seed calls."
    assert np.array_equal(te1, te2), "Test splits are not reproducible across identical seed calls."


def test_automated_target_leakage_detection(train_df):
    """
    LEAKAGE AUDIT CHECK:
    Automate detection of whether total_error_count deterministically encodes the target label.
    """
    matches = ((train_df["total_error_count"] > 0) == (train_df["failed"] == 1)).sum()
    leakage_rate = matches / len(train_df)

    # In static compiler validity, this is 100% by definition
    assert leakage_rate == 1.0, (
        f"Target label diverges from compiler validation state: {leakage_rate:.2f}"
    )


def test_independent_benchmark_isolation(train_df, independent_data):
    """
    INDEPENDENT EVALUATION CHECK:
    Verify that pipelines in independent_eval_set.json are genuinely novel and
    do not exist in the training dataset.
    """
    train_pipelines = set(train_df["pipeline_text"].str.strip())
    indep_samples = independent_data.get("samples", [])
    assert len(indep_samples) >= 10, "Independent set should contain at least 10 benchmarks."

    for s in indep_samples:
        indep_text = s["pipeline_text"].strip()
        assert indep_text not in train_pipelines, (
            f"Independent sample '{s['id']}' leaks into the training dataset!"
        )


def test_schema_parity_training_vs_inference(loaded_model):
    """Verify that feature names and count match exactly between training artifact and schema."""
    assert loaded_model.feature_names == FEATURE_COLUMNS
    assert len(loaded_model.feature_names) == 13

    # Test dictionary validation against schema
    sample_feat_dict = {f: 0.0 for f in FEATURE_COLUMNS}
    validated = validate_feature_dict(sample_feat_dict)
    assert list(validated.keys()) == FEATURE_COLUMNS


def test_model_artifact_metadata_contains_baseline_comparisons(loaded_model):
    """Verify that model artifact metadata records full evaluation audit and baseline results."""
    meta = loaded_model.metadata
    assert "baselines" in meta, "Metadata must record baseline comparison."
    baselines = meta["baselines"]

    assert "rule_based_compiler" in baselines
    assert "majority_class" in baselines
    assert "glm_ablated_without_total_error_count" in baselines
    assert "glm_full" in baselines

    # Rule baseline on static validity must be 100%
    assert baselines["rule_based_compiler"]["accuracy"] == 1.0

    # Majority class on balanced test set must be 50%
    assert baselines["majority_class"]["accuracy"] == 0.5

    # Ablated GLM must achieve non-trivial performance (> 70%) on held-out groups
    assert baselines["glm_ablated_without_total_error_count"]["accuracy"] >= 0.70

    # Verify group isolation is recorded
    assert meta.get("group_overlap_percent") == 0.0
    assert len(meta.get("train_groups", [])) > 0
    assert len(meta.get("test_groups", [])) > 0


def test_independent_evaluation_compiler_diagnostics(independent_data):
    """
    Verify compiler diagnostic accuracy on the independent evaluation set.
    Assert zero compiler false positives and false negatives on standard genomic pipelines.
    """
    compiler = PipelineCompiler()
    validator = PipelineValidator()

    for s in independent_data["samples"]:
        pipeline_text = s["pipeline_text"]
        expected_valid = s["expected_valid"]

        parsed_steps, syntax_errors = compiler.parse(pipeline_text)
        report = validator.validate(steps=parsed_steps, syntax_errors=syntax_errors)

        assert report.is_valid == expected_valid, (
            f"Compiler diagnostic discrepancy on sample '{s['id']}': "
            f"expected valid={expected_valid}, got valid={report.is_valid} with errors: {report.errors}"
        )
