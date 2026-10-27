"""
Integration and API endpoint security tests.

Covers:
- Standard /health, /analyze, /predict, /samples, /docs endpoints
- Payload size limit validation
- Feature vector schema validation
- CORS headers and origin restrictions
- Structured clean error responses
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.features import FEATURE_COLUMNS


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """GET /health should return 200 with service and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "FASTQC" in data["supported_tools"]


def test_analyze_valid_pipeline(client):
    """POST /analyze with a valid pipeline returns is_valid=True and LOW risk."""
    payload = {
        "pipeline_text": """
        FASTQC --in sample_raw.fastq --out qc.html
        TRIMMOMATIC --in sample_raw.fastq --out sample_trimmed.fq --minlen 36 --slidingwindow 4:20
        BWA MEM --ref reference.fa --in sample_trimmed.fq --out sample_aligned.sam
        SAMTOOLS VIEW --in sample_aligned.sam --out sample_aligned.bam
        SAMTOOLS SORT --in sample_aligned.bam --out sample_sorted.bam
        SAMTOOLS INDEX --in sample_sorted.bam
        GATK HaplotypeCaller --ref reference.fa --in sample_sorted.bam --out variants.vcf
        """,
        "declared_inputs": ["sample_raw.fastq", "reference.fa"]
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["validation"]["is_valid"] is True
    assert data["validation"]["total_errors"] == 0
    assert data["prediction"]["risk_category"] == "LOW"
    assert data["prediction"]["failure_probability"] < 0.30
    assert "model_type" in data["model_info"]
    assert len(data["validation"]["parsed_steps"]) == 7


def test_analyze_invalid_pipeline(client):
    """POST /analyze with errors returns line-indexed errors, suggestions, and HIGH risk."""
    payload = {
        "pipeline_text": """
        BOWTIE2 --in reads.fq --out aligned.sam
        SAMTOOLS SORT --in missing_reads.bam --out missing_reads.bam
        GATK HaplotypeCaller --ref ref.fa --in unindexed.bam --out vars.vcf
        """
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["validation"]["is_valid"] is False
    assert data["validation"]["total_errors"] > 0

    # Ensure each error has line number, error type, message, and suggestion
    for err in data["validation"]["errors"]:
        assert err["line_number"] > 0
        assert err["error_type"] is not None
        assert err["message"] is not None
        assert err["suggestion"] is not None

    assert data["prediction"]["risk_category"] == "HIGH"
    assert data["prediction"]["failure_probability"] >= 0.70


def test_analyze_empty_request(client):
    """POST /analyze with empty string returns 422 Unprocessable Entity."""
    response = client.post("/analyze", json={"pipeline_text": "   "})
    assert response.status_code == 422


def test_analyze_payload_size_limit(client):
    """POST /analyze with payload exceeding 50,000 characters must be rejected with 422."""
    giant_text = "FASTQC --in reads.fq\n" * 3000  # > 60,000 chars
    response = client.post("/analyze", json={"pipeline_text": giant_text})
    assert response.status_code == 422


def test_predict_endpoint_with_pipeline_text(client):
    """POST /predict with raw pipeline text returns risk evaluation."""
    payload = {
        "pipeline_text": "FASTQC --in reads.fq\nBWA MEM --ref ref.fa --in reads.fq --out out.sam"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    assert data["risk_category"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_endpoint_with_valid_feature_vector(client):
    """POST /predict with valid extracted feature dictionary."""
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    features["total_error_count"] = 5.0
    features["syntax_error_count"] = 2.0
    features["pipeline_length"] = 3.0

    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert data["risk_category"] == "HIGH"


def test_predict_endpoint_rejects_invalid_feature_values(client):
    """POST /predict with negative counts or NaN must return 422."""
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    features["syntax_error_count"] = -5.0  # Invalid negative count

    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "cannot be negative" in response.json()["detail"]


def test_predict_empty_payload(client):
    """POST /predict with neither text nor features returns 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_samples_endpoint(client):
    """GET /samples returns pre-packaged demonstration samples."""
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_docs_endpoint(client):
    """GET /docs returns OpenAPI swagger UI."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_cors_headers_development(client):
    """Verify CORS preflight / response behavior on configured origins."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/analyze", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_frontend_root_served(client):
    """GET / returns the integrated Stitch frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "GENOME / COMPILER" in response.text
    assert "dsl-editor" in response.text
    assert "compilerState" in response.text


def test_frontend_index_served(client):
    """GET /index.html returns the integrated Stitch frontend HTML."""
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "GENOME / COMPILER" in response.text


