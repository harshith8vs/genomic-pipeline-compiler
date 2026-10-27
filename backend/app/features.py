"""
Feature Extraction and Input Validation for Genomic Pipeline GLM Classifier.

Extracts structured quantitative features from compiler validation reports
to feed into the Generalized Linear Model (Logistic Regression).
Ensures strict schema adherence and input validation for both training and inference.
"""

import math
from typing import List, Dict, Any
import pandas as pd
from app.schemas import ParsedStep, ValidationReport, PipelineFeatures

# Authoritative single source of truth for the 13-dimensional GLM feature schema
FEATURE_COLUMNS = [
    "syntax_error_count",
    "missing_parameter_count",
    "sequence_violation_count",
    "unresolved_reference_count",
    "file_extension_error_count",
    "total_error_count",
    "warning_count",
    "pipeline_length",
    "has_qc_step",
    "has_alignment_step",
    "has_variant_calling_step",
    "tool_diversity_ratio",
    "unindexed_bam_count",
]

# Error types classified under syntax/lexical category
SYNTAX_ERROR_TYPES = {
    "SYNTAX_ERROR",
    "EMPTY_PIPELINE",
    "UNKNOWN_COMMAND",
    "UNKNOWN_SUBCOMMAND",
    "MISSING_SUBCOMMAND",
    "UNEXPECTED_ARGUMENT",
    "DUPLICATE_PARAMETER",
    "MISSING_PARAMETER_VALUE",
}


def extract_features(report: ValidationReport, steps: List[ParsedStep]) -> PipelineFeatures:
    """
    Extract exact features from a validation report and parsed AST steps.
    Adheres strictly to the error classification policy without double-counting.

    Args:
        report: The ValidationReport emitted by PipelineValidator.
        steps: List of ParsedSteps produced by PipelineCompiler.

    Returns:
        PipelineFeatures object adhering strictly to the ML schema.
    """
    # 1. Syntax errors: tokenization, malformed flags, unknown commands/subcommands
    syntax_error_count = sum(
        1 for e in report.errors if e.error_type in SYNTAX_ERROR_TYPES and e.severity == "ERROR"
    )

    # 2. Missing parameter errors: mandatory parameters omitted
    missing_parameter_count = sum(
        1 for e in report.errors if e.error_type == "MISSING_PARAMETER" and e.severity == "ERROR"
    )

    # 3. Sequence violation errors: out-of-order workflow stages
    sequence_violation_count = sum(
        1 for e in report.errors if e.error_type == "INVALID_COMMAND_SEQUENCE" and e.severity == "ERROR"
    )

    # 4. Unresolved reference errors: intermediate files referenced before generation
    unresolved_reference_count = sum(
        1 for e in report.errors if e.error_type == "UNRESOLVED_FILE_REFERENCE" and e.severity == "ERROR"
    )

    # 5. File extension errors: invalid format / file type mismatches
    file_extension_error_count = sum(
        1 for e in report.errors if e.error_type == "INVALID_FILE_EXTENSION" and e.severity == "ERROR"
    )

    # 6. Unindexed BAM errors: BAM used in GATK without prior indexing
    unindexed_bam_count = sum(
        1 for e in report.errors if e.error_type == "UNINDEXED_BAM_INPUT" and e.severity == "ERROR"
    )

    # Total compiler errors and warnings
    total_error_count = sum(1 for e in report.errors if e.severity == "ERROR")
    warning_count = len(report.warnings)
    pipeline_length = len(steps)

    # Domain stage indicators
    has_qc_step = 1 if any(s.tool in ("FASTQC", "TRIMMOMATIC") for s in steps) else 0
    has_alignment_step = 1 if any(s.tool == "BWA" for s in steps) else 0
    has_variant_calling_step = 1 if any(s.tool == "GATK" for s in steps) else 0

    unique_tools = len(set(s.tool for s in steps))
    tool_diversity_ratio = round(unique_tools / pipeline_length, 4) if pipeline_length > 0 else 0.0

    return PipelineFeatures(
        syntax_error_count=syntax_error_count,
        missing_parameter_count=missing_parameter_count,
        sequence_violation_count=sequence_violation_count,
        unresolved_reference_count=unresolved_reference_count,
        file_extension_error_count=file_extension_error_count,
        total_error_count=total_error_count,
        warning_count=warning_count,
        pipeline_length=pipeline_length,
        has_qc_step=has_qc_step,
        has_alignment_step=has_alignment_step,
        has_variant_calling_step=has_variant_calling_step,
        tool_diversity_ratio=tool_diversity_ratio,
        unindexed_bam_count=unindexed_bam_count,
    )


def validate_feature_dict(feature_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Validate arbitrary dictionary inputs before feeding into the GLM model.
    Rejects NaN, inf, unexpected keys, missing keys, non-numerics, negative counts,
    and out-of-range indicators/ratios.

    Returns:
        Sanitized feature dictionary matching FEATURE_COLUMNS order.
    """
    if not isinstance(feature_dict, dict):
        raise ValueError("Feature vector payload must be a JSON dictionary.")

    # Check for unexpected feature keys
    extra_keys = set(feature_dict.keys()) - set(FEATURE_COLUMNS)
    if extra_keys:
        raise ValueError(
            f"Unexpected feature key(s): {sorted(list(extra_keys))}. Allowed keys: {FEATURE_COLUMNS}"
        )

    # Check for missing feature keys
    missing_keys = set(FEATURE_COLUMNS) - set(feature_dict.keys())
    if missing_keys:
        raise ValueError(
            f"Missing required feature(s): {sorted(list(missing_keys))}. All 13 schema features must be provided."
        )

    sanitized: Dict[str, float] = {}

    count_features = {
        "syntax_error_count",
        "missing_parameter_count",
        "sequence_violation_count",
        "unresolved_reference_count",
        "file_extension_error_count",
        "total_error_count",
        "warning_count",
        "pipeline_length",
        "unindexed_bam_count",
    }
    binary_features = {"has_qc_step", "has_alignment_step", "has_variant_calling_step"}

    for col in FEATURE_COLUMNS:
        raw_val = feature_dict[col]
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{col}' must be a numeric value, got '{raw_val}'.")

        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Feature '{col}' cannot be NaN or infinite.")

        if col in count_features:
            if val < 0:
                raise ValueError(f"Count feature '{col}' cannot be negative (got {val}).")
            if not val.is_integer():
                raise ValueError(f"Count feature '{col}' must be a non-negative integer (got {val}).")

        if col in binary_features:
            if val not in (0.0, 1.0):
                raise ValueError(f"Binary indicator '{col}' must be exactly 0 or 1 (got {val}).")

        if col == "tool_diversity_ratio":
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Feature 'tool_diversity_ratio' must be between 0.0 and 1.0 (got {val}).")

        sanitized[col] = val

    return sanitized


def features_to_dict(features: PipelineFeatures) -> Dict[str, Any]:
    """Convert PipelineFeatures to dictionary adhering to FEATURE_COLUMNS order."""
    d = features.model_dump()
    return {col: d[col] for col in FEATURE_COLUMNS}


def features_to_dataframe(features: PipelineFeatures) -> pd.DataFrame:
    """Format single feature instance into DataFrame with correct schema columns."""
    return pd.DataFrame([features_to_dict(features)], columns=FEATURE_COLUMNS)


def dict_to_dataframe(feature_dict: Dict[str, Any]) -> pd.DataFrame:
    """Validate and format arbitrary dict into DataFrame with FEATURE_COLUMNS."""
    sanitized = validate_feature_dict(feature_dict)
    return pd.DataFrame([sanitized], columns=FEATURE_COLUMNS)
