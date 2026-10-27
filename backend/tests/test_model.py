"""
Unit tests for the GLM Machine Learning Model and Feature Parity.
"""

import pytest
import joblib
from app.compiler import PipelineCompiler
from app.validator import PipelineValidator
from app.features import extract_features, FEATURE_COLUMNS
from app.model import GLMModelManager, DEFAULT_MODEL_PATH


@pytest.fixture
def model_manager():
    mgr = GLMModelManager(model_path=DEFAULT_MODEL_PATH)
    loaded = mgr.load_model()
    assert loaded is True, "Trained model artifact should exist and load successfully."
    return mgr


def test_model_loading_and_metadata(model_manager):
    """Verify that model loads with schema version and cross-validation metadata."""
    assert model_manager.is_loaded() is True
    assert model_manager.feature_names == FEATURE_COLUMNS

    meta = model_manager.metadata
    assert "accuracy" in meta
    assert meta["accuracy"] >= 0.85
    assert "cv_accuracy_mean" in meta
    assert meta["cv_accuracy_mean"] >= 0.85
    assert "brier_score" in meta
    assert "roc_auc" in meta
    assert meta["roc_auc"] >= 0.85
    assert "test_class_support" in meta
    assert "framework" in meta
    assert "scikit-learn" in meta["framework"]


def test_missing_model_file_fails_safely():
    """Attempting to load a non-existent model file must return False without crashing."""
    mgr = GLMModelManager(model_path="/non/existent/path/model.joblib")
    assert mgr.load_model() is False
    assert mgr.is_loaded() is False


def test_corrupted_artifact_rejected_safely(tmp_path):
    """Artifacts with schema mismatch or invalid structure must be safely rejected."""
    bad_artifact_path = str(tmp_path / "bad_model.joblib")

    # Save a dict missing 'pipeline'
    joblib.dump({"corrupted": True}, bad_artifact_path)
    mgr = GLMModelManager(model_path=bad_artifact_path)
    assert mgr.load_model() is False

    # Save a dict with mismatched feature schema
    joblib.dump({
        "pipeline": model_manager.pipeline if "model_manager" in locals() else None,
        "features": ["wrong_feature_1", "wrong_feature_2"],
    }, bad_artifact_path)
    mgr2 = GLMModelManager(model_path=bad_artifact_path)
    assert mgr2.load_model() is False


def test_positive_class_resolution(model_manager):
    """Verify that positive class index is dynamically resolved from classifier.classes_."""
    classifier = model_manager.pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    assert 0 in classes and 1 in classes
    assert len(classes) == 2


def test_valid_pipeline_low_risk_prediction(model_manager):
    """A valid pipeline must yield low failure risk probability (< 0.30)."""
    compiler = PipelineCompiler()
    validator = PipelineValidator()

    valid_script = """
    FASTQC --in reads.fastq --out qc.html
    TRIMMOMATIC --in reads.fastq --out clean.fq --minlen 36 --slidingwindow 4:20
    BWA MEM --ref ref.fa --in clean.fq --out out.sam
    SAMTOOLS SORT --in out.sam --out sorted.bam
    SAMTOOLS INDEX --in sorted.bam
    GATK HaplotypeCaller --ref ref.fa --in sorted.bam --out vars.vcf
    """
    steps, syntax_errs = compiler.parse(valid_script)
    report = validator.validate(steps, syntax_errs, declared_inputs=["reads.fastq", "ref.fa"])
    features = extract_features(report, steps)

    prediction = model_manager.predict_risk(features)
    assert 0.0 <= prediction.failure_probability <= 1.0
    assert prediction.failure_probability < 0.30
    assert prediction.risk_category == "LOW"
    assert prediction.predicted_outcome == "SUCCESS"


def test_faulty_pipeline_high_risk_prediction(model_manager):
    """A pipeline riddled with errors must yield high failure risk probability (>= 0.70)."""
    compiler = PipelineCompiler()
    validator = PipelineValidator()

    bad_script = """
    GATK HaplotypeCaller --ref ref.fa --in missing_unindexed.bam --out out.vcf
    SAMTOOLS SORT --in missing_intermediate.bam --out missing_intermediate.bam
    UNKNOWN_TOOL --in bad.txt
    """
    steps, syntax_errs = compiler.parse(bad_script)
    report = validator.validate(steps, syntax_errs)
    features = extract_features(report, steps)

    prediction = model_manager.predict_risk(features)
    assert 0.0 <= prediction.failure_probability <= 1.0
    assert prediction.failure_probability >= 0.70
    assert prediction.risk_category == "HIGH"
    assert prediction.predicted_outcome == "FAILURE"
    assert len(prediction.top_risk_factors) > 0


def test_predict_from_dict_and_schema_validation(model_manager):
    """Verify inference from raw dictionary and error on missing features."""
    valid_feature_dict = {col: 0.0 for col in FEATURE_COLUMNS}
    valid_feature_dict["pipeline_length"] = 5.0
    valid_feature_dict["has_qc_step"] = 1.0
    valid_feature_dict["has_alignment_step"] = 1.0
    valid_feature_dict["tool_diversity_ratio"] = 0.8

    pred = model_manager.predict_from_dict(valid_feature_dict)
    assert 0.0 <= pred.failure_probability <= 1.0

    # Test missing feature key
    incomplete_dict = {"syntax_error_count": 2}
    with pytest.raises(ValueError, match="Missing required feature"):
        model_manager.predict_from_dict(incomplete_dict)
