"""
Pydantic schemas for the Intelligent Genomic Pipeline Compiler and GLM Model (Version 2.0.0).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ValidationError(BaseModel):
    """Represents a syntax, semantic, sequence, or reference error."""
    line_number: int = Field(..., description="1-indexed line number in pipeline script (0 for global errors)")
    command: Optional[str] = Field(None, description="The command or tool where error occurred")
    error_type: str = Field(..., description="Classification of the error")
    severity: str = Field(default="ERROR", description="ERROR or WARNING")
    message: str = Field(..., description="Detailed explanation of the issue")
    suggestion: Optional[str] = Field(None, description="Actionable correction recommendation")


class ParsedStep(BaseModel):
    """Represents an intermediate representation (IR) of a parsed pipeline command."""
    line_number: int
    raw_command: str
    tool: str
    subcommand: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    input_files: List[str] = Field(default_factory=list)
    output_files: List[str] = Field(default_factory=list)
    stage: str
    is_valid: bool = Field(default=True, description="False if step encountered syntax/command structure errors")


class ValidationReport(BaseModel):
    """Compiler analysis outcome for a pipeline."""
    is_valid: bool = Field(..., description="True if no ERROR severity issues found")
    status: str = Field(..., description="VALID, FAILED_WITH_ERRORS, or VALID_WITH_WARNINGS")
    total_errors: int
    total_warnings: int
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)
    parsed_steps: List[ParsedStep] = Field(default_factory=list)


class PipelineFeatures(BaseModel):
    """
    Extracted feature vector fed into the Logistic Regression (GLM) model.
    Must maintain identical schema across training and inference.
    """
    syntax_error_count: int = Field(..., ge=0, description="Count of syntax, flag, or command parsing errors")
    missing_parameter_count: int = Field(..., ge=0, description="Count of omitted mandatory arguments")
    sequence_violation_count: int = Field(..., ge=0, description="Count of out-of-order genomic pipeline steps")
    unresolved_reference_count: int = Field(..., ge=0, description="Count of dangling input file references")
    file_extension_error_count: int = Field(..., ge=0, description="Count of invalid format or mismatched file types")
    total_error_count: int = Field(..., ge=0, description="Cumulative compiler errors")
    warning_count: int = Field(..., ge=0, description="Cumulative compiler warnings")
    pipeline_length: int = Field(..., ge=0, description="Total number of executable steps")
    has_qc_step: int = Field(..., ge=0, le=1, description="1 if FASTQC/TRIMMOMATIC present, else 0")
    has_alignment_step: int = Field(..., ge=0, le=1, description="1 if BWA aligner present, else 0")
    has_variant_calling_step: int = Field(..., ge=0, le=1, description="1 if GATK present, else 0")
    tool_diversity_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of distinct tools to total steps")
    unindexed_bam_count: int = Field(..., ge=0, description="Number of BAM files used in GATK without prior indexing")


class GLMPrediction(BaseModel):
    """Prediction output from the Logistic Regression (GLM) failure model."""
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Statistical failure probability (0.0 to 1.0)")
    risk_category: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    predicted_outcome: str = Field(..., description="Predicted pipeline execution outcome: SUCCESS or FAILURE")
    confidence: float = Field(..., description="Statistical confidence score of the predicted class")
    top_risk_factors: List[str] = Field(default_factory=list, description="Dominant features contributing to risk")


class ModelMetadata(BaseModel):
    """Metadata regarding the machine learning model."""
    model_type: str = "Generalized Linear Model (Logistic Regression with Logit Link)"
    framework: str = "scikit-learn"
    training_dataset: str = "Curated synthetic genomic benchmark pipelines (240 verified labelled runs across 12 groups)"
    schema_version: str = "2.1.0"
    limitation_notice: str = (
        "Supplementary statistical risk scoring model trained on synthetic benchmark pipeline mutations. "
        "The deterministic compiler remains the authoritative system for lexical, syntax, semantic, and workflow validation. "
        "GLM predictions indicate supplementary statistical risk on benchmark patterns and do not serve as proof of real-world "
        "genomic cluster runtime execution success or failure."
    )


class AnalyzeRequest(BaseModel):
    """Payload for analyzing a genomic pipeline with reasonable input size limits."""
    pipeline_text: str = Field(
        ...,
        max_length=50000,
        description="Multi-line genomic pipeline script (max 50,000 characters)"
    )
    declared_inputs: Optional[List[str]] = Field(
        default_factory=list,
        max_length=100,
        description="Optional list of preexisting files available on disk (max 100 items)"
    )

    @field_validator("pipeline_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pipeline script text cannot be empty or pure whitespace.")
        return v


class PredictRequest(BaseModel):
    """Payload for direct risk prediction."""
    pipeline_text: Optional[str] = Field(
        None,
        max_length=50000,
        description="Optional raw pipeline script to parse and predict"
    )
    features: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional pre-extracted feature dictionary matching the 13 schema features"
    )


class AnalysisResponse(BaseModel):
    """Combined compiler validation and GLM failure risk assessment."""
    validation: ValidationReport
    features: PipelineFeatures
    prediction: GLMPrediction
    model_info: ModelMetadata


class HealthResponse(BaseModel):
    """API health and readiness check."""
    status: str
    service: str
    model_loaded: bool
    model_version: str
    supported_tools: List[str]
