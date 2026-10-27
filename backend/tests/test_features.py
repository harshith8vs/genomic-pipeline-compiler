"""
Feature Mapping and Input Validation Tests.

Verifies:
- Accurate mapping of each compiler diagnostic to its designated feature category.
- Separation of unresolved references from reference conflicts.
- Separation of extension errors from parameter value format errors.
- Strict validation of input feature dictionaries before inference (rejection of NaN, inf, negatives, bad bounds).
- Correct column order and schema dimensions.
"""

import pytest
from app.compiler import PipelineCompiler
from app.validator import PipelineValidator
from app.features import (
    extract_features,
    validate_feature_dict,
    FEATURE_COLUMNS,
)


@pytest.fixture
def compiler():
    return PipelineCompiler()


@pytest.fixture
def validator():
    return PipelineValidator()


def test_feature_columns_schema_order():
    """Verify that FEATURE_COLUMNS defines exactly 13 unique canonical features."""
    assert len(FEATURE_COLUMNS) == 13
    assert len(set(FEATURE_COLUMNS)) == 13
    assert FEATURE_COLUMNS[0] == "syntax_error_count"
    assert FEATURE_COLUMNS[-1] == "unindexed_bam_count"


def test_unresolved_reference_count_strictly_isolated(compiler, validator):
    """
    Verify that unresolved_reference_count counts UNRESOLVED_FILE_REFERENCE,
    and does NOT include OUTPUT_REFERENCE_CONFLICT (Issue 6).
    """
    # Pipeline with in-place conflict only
    script_conflict = "SAMTOOLS SORT --in sample.bam --out sample.bam"
    steps1, syntax1 = compiler.parse(script_conflict)
    report1 = validator.validate(steps1, syntax1, declared_inputs=["sample.bam"])
    feat1 = extract_features(report1, steps1)

    assert feat1.unresolved_reference_count == 0
    assert feat1.total_error_count >= 1  # OUTPUT_REFERENCE_CONFLICT is counted in total errors

    # Pipeline with unresolved reference only
    script_unresolved = "SAMTOOLS SORT --in missing_reads.bam --out sorted.bam"
    steps2, syntax2 = compiler.parse(script_unresolved)
    report2 = validator.validate(steps2, syntax2)
    feat2 = extract_features(report2, steps2)

    assert feat2.unresolved_reference_count == 1
    assert feat2.total_error_count >= 1


def test_file_extension_error_count_strictly_isolated(compiler, validator):
    """
    Verify that file_extension_error_count counts INVALID_FILE_EXTENSION only,
    and does NOT include INVALID_PARAMETER_VALUE numeric errors (Issue 6).
    """
    # Pipeline with numeric value format error only (--threads -4)
    script_numeric = "FASTQC --in reads.fq --threads -4"
    steps1, syntax1 = compiler.parse(script_numeric)
    report1 = validator.validate(steps1, syntax1, declared_inputs=["reads.fq"])
    feat1 = extract_features(report1, steps1)

    assert feat1.file_extension_error_count == 0
    assert feat1.total_error_count >= 1

    # Pipeline with invalid file extension only (.txt instead of .bam)
    script_ext = "SAMTOOLS SORT --in reads.sam --out reads.txt"
    steps2, syntax2 = compiler.parse(script_ext)
    report2 = validator.validate(steps2, syntax2, declared_inputs=["reads.sam"])
    feat2 = extract_features(report2, steps2)

    assert feat2.file_extension_error_count == 1


def test_validate_valid_feature_dict():
    """A well-formed feature dict should validate cleanly."""
    valid_dict = {col: 0.0 for col in FEATURE_COLUMNS}
    valid_dict["pipeline_length"] = 5.0
    valid_dict["has_qc_step"] = 1.0
    valid_dict["tool_diversity_ratio"] = 0.8

    sanitized = validate_feature_dict(valid_dict)
    assert sanitized["pipeline_length"] == 5.0
    assert sanitized["tool_diversity_ratio"] == 0.8


def test_reject_nan_and_inf():
    """validate_feature_dict must reject NaN and infinity."""
    base = {col: 0.0 for col in FEATURE_COLUMNS}

    nan_dict = dict(base, total_error_count=float("nan"))
    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        validate_feature_dict(nan_dict)

    inf_dict = dict(base, total_error_count=float("inf"))
    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        validate_feature_dict(inf_dict)


def test_reject_negative_counts():
    """Negative count features must be rejected."""
    base = {col: 0.0 for col in FEATURE_COLUMNS}
    neg_dict = dict(base, syntax_error_count=-2.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_feature_dict(neg_dict)


def test_reject_invalid_binary_indicators():
    """Binary stage features must be strictly 0 or 1."""
    base = {col: 0.0 for col in FEATURE_COLUMNS}
    bad_binary = dict(base, has_qc_step=2.0)
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        validate_feature_dict(bad_binary)


def test_reject_out_of_bounds_diversity_ratio():
    """tool_diversity_ratio must be between 0.0 and 1.0."""
    base = {col: 0.0 for col in FEATURE_COLUMNS}
    bad_ratio = dict(base, tool_diversity_ratio=1.5)
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        validate_feature_dict(bad_ratio)


def test_reject_unexpected_and_missing_keys():
    """Unexpected keys or missing keys must be rejected with helpful messages."""
    base = {col: 0.0 for col in FEATURE_COLUMNS}

    # Extra key
    extra = dict(base, unknown_feature=12.0)
    with pytest.raises(ValueError, match="Unexpected feature key"):
        validate_feature_dict(extra)

    # Missing key
    incomplete = {k: v for k, v in base.items() if k != "pipeline_length"}
    with pytest.raises(ValueError, match="Missing required feature"):
        validate_feature_dict(incomplete)
