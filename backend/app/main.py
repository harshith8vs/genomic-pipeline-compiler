"""
FastAPI Application for Intelligent Genomic Pipeline Compiler and GLM Model (Version 2.0.0).

Endpoints:
- GET /health: Health check, service status, and model readiness
- POST /analyze: Combined compiler validation and GLM failure risk prediction
- POST /predict: Statistical GLM prediction from raw pipeline or feature vector
- GET /samples: Catalog of valid and invalid sample pipelines for demonstration
- GET /docs: Interactive Swagger documentation
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    PredictRequest,
    GLMPrediction,
    HealthResponse,
    ModelMetadata,
)
from app.compiler import PipelineCompiler
from app.validator import PipelineValidator
from app.features import extract_features
from app.model import model_manager

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook: Ensure GLM model is trained and loaded into memory on startup."""
    logger.info("Initializing Intelligent Genomic Pipeline Compiler Backend...")
    loaded = model_manager.load_model()
    if not loaded:
        logger.info("Model artifact not found or outdated. Auto-training verified GLM model...")
        try:
            from data.generate_dataset import generate_pipeline_dataset
            df, _ = generate_pipeline_dataset(num_samples=200, seed=42)
            model_manager.train_and_save(df=df, target_col="failed")
            logger.info("GLM model auto-trained and loaded successfully.")
        except Exception as e:
            logger.error(f"Model auto-training encountered error: {e}", exc_info=True)
    else:
        logger.info("Verified GLM model artifact loaded successfully.")
    yield
    logger.info("Shutting down backend service.")


app = FastAPI(
    title="Intelligent Genomic Pipeline Compiler & GLM Engine",
    description=(
        "Compiler analysis and Generalized Linear Model (Logistic Regression) "
        "failure prediction engine for high-throughput genomic data processing pipelines "
        "(FASTQC, TRIMMOMATIC, BWA, SAMTOOLS, GATK)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Configurable explicit CORS origins for local frontend development and production
# Allows overriding via CORS_ORIGINS environment variable (comma-separated)
default_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
env_origins = os.getenv("CORS_ORIGINS")
allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else default_cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Shield internal stack traces and paths from leaking to clients while logging server-side."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    logger.error(f"Unhandled server exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the genomic pipeline."}
    )


compiler = PipelineCompiler()
validator = PipelineValidator()


def get_dynamic_model_metadata() -> ModelMetadata:
    """Derive ModelMetadata dynamically from verified model manager state."""
    meta = model_manager.metadata or {}
    total_samples = meta.get("train_samples", 0) + meta.get("test_samples", 0)
    dataset_desc = (
        f"Curated synthetic genomic benchmark pipelines ({total_samples} verified labelled runs)"
        if total_samples > 0 else "Curated synthetic genomic benchmark pipelines (200 verified labelled runs)"
    )
    framework_desc = meta.get("framework", "scikit-learn")
    return ModelMetadata(
        model_type="Generalized Linear Model (Logistic Regression with Logit Link)",
        framework=framework_desc,
        training_dataset=dataset_desc,
        schema_version=meta.get("schema_version", "2.1.0"),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and model readiness check",
    tags=["System"]
)
def get_health():
    """Verify API uptime, model readiness, and supported genomic tools."""
    return HealthResponse(
        status="healthy",
        service="genomic-pipeline-compiler-glm",
        model_loaded=model_manager.is_loaded(),
        model_version=f"2.1.0-glm-{model_manager.metadata.get('framework', 'scikit-learn')}",
        supported_tools=["FASTQC", "TRIMMOMATIC", "BWA", "SAMTOOLS", "GATK"],
    )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Perform compiler validation and GLM failure risk prediction",
    tags=["Compiler & ML"]
)
def analyze_pipeline(request: AnalyzeRequest):
    """
    Execute full compiler pipeline analysis:
    1. Lexical and syntax tokenization
    2. Semantic and command sequence validation
    3. DAG file reference resolution
    4. Quantitative feature extraction
    5. Generalized Linear Model (GLM) failure probability calculation
    """
    pipeline_text = request.pipeline_text.strip()
    if not pipeline_text:
        raise HTTPException(
            status_code=422,
            detail="Pipeline text cannot be empty. Please provide one or more genomic commands."
        )

    # 1. Frontend compiler lexer & parser
    parsed_steps, syntax_errors = compiler.parse(pipeline_text)

    # 2. Semantic & sequence validation
    validation_report = validator.validate(
        steps=parsed_steps,
        syntax_errors=syntax_errors,
        declared_inputs=request.declared_inputs or []
    )

    # 3. Extract exact ML features matching training schema
    features = extract_features(validation_report, parsed_steps)

    # 4. GLM statistical risk prediction
    try:
        prediction = model_manager.predict_risk(features)
    except Exception as e:
        logger.error(f"Prediction failed in /analyze: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Statistical machine learning risk prediction failed."
        )

    return AnalysisResponse(
        validation=validation_report,
        features=features,
        prediction=prediction,
        model_info=get_dynamic_model_metadata(),
    )


@app.post(
    "/predict",
    response_model=GLMPrediction,
    summary="Predict pipeline failure risk",
    tags=["ML Prediction"]
)
def predict_failure_risk(request: PredictRequest):
    """
    Predict failure risk either from:
    1. A raw pipeline script string (auto-extracts compiler features)
    2. A dictionary of precomputed numerical features matching the 13 schema features
    """
    if request.features:
        try:
            return model_manager.predict_from_dict(request.features)
        except ValueError as ve:
            raise HTTPException(
                status_code=422,
                detail=str(ve)
            )
        except Exception as e:
            logger.error(f"Prediction failed on feature vector: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Machine learning prediction failed on the provided feature vector."
            )

    if request.pipeline_text and request.pipeline_text.strip():
        parsed_steps, syntax_errors = compiler.parse(request.pipeline_text)
        report = validator.validate(steps=parsed_steps, syntax_errors=syntax_errors)
        features = extract_features(report, parsed_steps)
        return model_manager.predict_risk(features)

    raise HTTPException(
        status_code=422,
        detail="Either 'pipeline_text' or 'features' must be provided for risk prediction."
    )


@app.get(
    "/samples",
    summary="Get sample pipelines for demonstration",
    tags=["Demonstration Data"]
)
def get_samples():
    """Retrieve curated valid and invalid genomic pipeline samples for testing."""
    samples_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_pipelines.json")
    )
    if os.path.exists(samples_file):
        with open(samples_file, "r") as f:
            return json.load(f)
    return {"samples": []}


# Frontend Integration: Serve Stitch Frontend
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse(status_code=404, content={"detail": "Frontend index.html not found"})

    @app.get("/index.html", include_in_schema=False)
    def serve_frontend_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse(status_code=404, content={"detail": "Frontend index.html not found"})

