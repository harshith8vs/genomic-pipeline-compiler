# Intelligent Compiler for Detecting Errors in Genomic Data Processing Pipelines Using GLM (v2.1.0)

An intelligent compiler engine paired with a Generalized Linear Model (GLM) machine learning classifier to perform static syntax/semantic validation, workflow sequence auditing, DAG dataflow reference verification, and supplementary failure risk scoring on high-throughput genomic data pipelines.

---

## 1. Project Overview

Genomic pipelines (e.g. Next-Generation Sequencing workflows) chain together specialized tools such as read quality controllers, short-read aligners, coordinate sort/index utilities, and probabilistic variant callers. When scripts contain syntactic errors, missing arguments, misplaced intermediate files, or inverted execution stages (e.g. calling GATK before BWA alignment), compute jobs fail hours or days into execution on High-Performance Computing (HPC) clusters.

This project delivers:
1. **Authoritative Deterministic Compiler Frontend & Semantic Validator**:
   - The primary and authoritative system for lexical tokenization, syntax validation, semantic rules, DAG file reference resolution, and biological workflow sequence auditing across bioinformatics tools (`FASTQC`, `TRIMMOMATIC`, `BWA`, `SAMTOOLS`, `GATK`).
   - Handles tokenization edge cases: quoted paths with spaces, `--flag=value` syntax, negative numbers as parameters (`--threads -4`), duplicate parameter flags, missing flag values, and inline comments (`#` and `//`).
   - Enforces workflow order transitions across QC, reference indexing, alignment, processing, and variant calling.
   - Accurately treats `GATK MarkDuplicates` `--metrics` as an output artifact rather than an input file.
   - Emits line numbers, error classifications, detailed human-readable explanations, and actionable suggestions.
2. **Supplementary GLM Probabilistic Risk Scorer**:
   - Implements scikit-learn's `LogisticRegression` (binomial GLM with logit link function) and `StandardScaler` inside a verified pipeline.
   - Extracts a strictly isolated 13-dimensional quantitative feature vector without conflation or double-counting.
   - Evaluates performance using group-isolated holdout splits, 3-fold StratifiedGroupKFold cross-validation on training groups, and baseline comparisons.
   - Dynamically resolves positive class index from `classifier.classes_`.
   - Used strictly for **supplementary continuous risk scoring, probability calibration, and factor attribution**; does not serve as proof of real-world cluster execution failure or replace deterministic validation.
3. **Production-Ready FastAPI REST Backend**:
   - Clean RESTful endpoints (`/health`, `/analyze`, `/predict`, `/samples`, `/docs`).
   - Configurable explicit CORS origin whitelist (defaulting to local development ports 3000, 5173, 8000; overridable via `CORS_ORIGINS`).
   - Request size guards (max 50,000 characters) and server-side exception logging without leaking stack traces or internal filesystem paths.


---

## 2. Architecture

```
[ Genomic Pipeline Script ] (Multi-line text / DSL)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. COMPILER ENGINE (compiler.py)                           │
│    - Lexer & Parser: Tokenization, flags, comments          │
│    - Handles spaces in quotes, --flag=val, negative params │
│    - Abstract Syntax Tree (AST): ParsedStep IR (is_valid)   │
│    - Command-specific file registration (MarkDuplicates)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SEMANTIC & RULE VALIDATOR (validator.py)                 │
│    - Parameter requirements & format validation             │
│    - Directed Acyclic Graph (DAG) file resolution flow       │
│    - Biological workflow transition order matrix            │
│    - In-place overwrite and unindexed BAM checks            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE EXTRACTION (features.py)                         │
│    - 13 canonical features isolated without conflation      │
│    - Strict input validation against NaN, inf, negatives    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. GLM CLASSIFIER (model.py - LogisticRegression)           │
│    - StandardScaler + L2 Logit Link Model                   │
│    - 3-Fold StratifiedGroupKFold CV & Brier Scoring         │
│    - Dynamic positive class resolution                      │
│    - Outputs Failure Probability, Risk Tier & Top Factors   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. FASTAPI REST API (main.py)                               │
│    - Endpoints: /health, /analyze, /predict, /samples       │
│    - Configurable explicit CORS & request size guards       │
│    - Interactive Swagger OpenAPI Docs (/docs)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Web API**: FastAPI `0.162.0` & Uvicorn `0.54.0`
- **Machine Learning (GLM)**: scikit-learn `1.9.1` (`LogisticRegression`, `StandardScaler`, `Pipeline`)
- **Data Manipulation**: pandas `3.1.0`, NumPy `2.5.4`
- **Model Serialization**: joblib `1.6.0`
- **Testing**: pytest `9.1.1`, httpx `0.28.1`
- **Data Validation**: Pydantic `2.14.1`

*Zero paid APIs, external LLM APIs, cloud dependencies, or GPUs required.*

---

## 4. Supported Genomic Tools & Domain Rules

| Tool | Subcommands | Required Parameters | Expected Formats | Pipeline Stage |
|---|---|---|---|---|
| `FASTQC` | None | `--in` | `--in`: `.fastq`, `.fq`, `.fastq.gz`; `--out`: `.html`, `.zip` | Quality Control (QC) |
| `TRIMMOMATIC` | None | `--in`, `--out` | `--in`, `--out`: FASTQ; `--slidingwindow`: `window:qual` (e.g. `4:20`); `--minlen`: positive int | Quality Control (QC) |
| `BWA` | `MEM`, `INDEX` | `MEM`: `--ref`, `--in`, `--out`<br>`INDEX`: `--ref` | `--ref`: `.fa`, `.fasta`; `--in`: FASTQ; `--out`: `.sam`, `.bam` | Read Alignment |
| `SAMTOOLS` | `VIEW`, `SORT`, `INDEX` | `VIEW`: `--in`, `--out`<br>`SORT`: `--in`, `--out`<br>`INDEX`: `--in` | `SORT`: `--out` must be `.bam`<br>`INDEX`: `--in` must be `.bam` | Alignment Processing |
| `GATK` | `HaplotypeCaller`, `Mutect2`, `MarkDuplicates` | `HaplotypeCaller`: `--ref`, `--in`, `--out`<br>`MarkDuplicates`: `--in`, `--out` | `--in`: coordinate-sorted, indexed `.bam`<br>`--out`: `.vcf`, `.vcf.gz`<br>`--metrics`: `.txt`, `.metrics` (output artifact) | Variant Discovery / Processing |

---

## 5. Directory Structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app and API route definitions
│   │   ├── compiler.py          # Lexer, tokenizer, AST step intermediate representation
│   │   ├── validator.py         # Semantic, sequence, and file reference validator
│   │   ├── features.py          # Feature extraction and input dictionary validation
│   │   ├── model.py             # LogisticRegression GLM manager, CV, training, inference
│   │   └── schemas.py           # Pydantic data schemas
│   ├── data/
│   │   ├── sample_pipelines.json # Benchmark valid and invalid pipeline examples
│   │   ├── generate_dataset.py   # Mutation-based verified training data generator v2.0.0
│   │   ├── train_dataset.json    # Verified labelled training set (200 pipelines)
│   │   └── train_dataset.csv     # CSV representation of benchmark dataset
│   ├── models/
│   │   └── glm_pipeline_model.joblib # Serialized scikit-learn pipeline (Scaler + Model)
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_compiler.py     # Compiler & validator unit tests (23 tests)
│   │   ├── test_model.py        # ML training, feature alignment, prediction tests (7 tests)
│   │   ├── test_api.py          # FastAPI endpoint integration tests (12 tests)
│   │   ├── test_dataset.py      # Dataset integrity and label verification tests (5 tests)
│   │   └── test_features.py     # Feature mapping isolation and validation tests (6 tests)
│   ├── requirements.txt         # Pinned dependencies specification
│   ├── train.py                 # Standalone script to train & evaluate the model
│   └── README.md
├── requirements.txt
├── CODE_REVIEW_AND_FIX_REPORT.md # Comprehensive 13-section audit and repair report
└── README.md
```

---

## 6. Installation & Setup

### Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

## 7. Model Training & Rigorous Evaluation (v2.1.0)

To audit the dataset, verify group-isolated holdout splits, and train the GLM with baseline comparisons:
```bash
source venv/bin/activate
python backend/train.py
```

### Evaluation Methodology & Leakage Audit Findings:
- **Tautological Labeling & Target Leakage**: In synthetic dataset generation for static validity, `failed = (total_error_count > 0)`. The aggregate feature `total_error_count` directly encodes the target label. Any linear classifier with access to this feature will achieve 100% test accuracy.
- **Group Separation**: To prevent mutated pipeline clones from crossing between train and test sets, the dataset is partitioned into 12 distinct architectural groups (`GRP_01` to `GRP_12`). Using `GroupShuffleSplit`, 9 groups are assigned to training (180 samples) and 3 complete groups are held out for testing (60 samples), guaranteeing **0.00% template overlap**.
- **3-Fold StratifiedGroupKFold Cross-Validation (Training Split)**: Mean CV Accuracy = **96.67%** ($\pm 0.0272$).

### Baseline Comparison on Group-Held-Out Test Set (60 samples, 0% group overlap):
| Model / Baseline | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| **1. Deterministic Compiler Rule (`errors > 0`)** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | N/A | N/A |
| **2. Majority-Class Baseline (`DummyClassifier`)** | **0.5000** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| **3. Ablated GLM (Excl. `total_error_count`)** | **0.9333** | 1.0000 | 0.8667 | 0.9286 | 0.9444 | 0.0808 |
| **4. Full GLM Model (All 13 Features)** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0154 |

### Independent Out-of-Distribution Benchmark Evaluation (15 isolated templates):
Evaluated on completely novel pipeline templates from `backend/data/independent_eval_set.json` that never appeared in any generator group:
- **Compiler Rule Baseline**: 100.00% accuracy (0 compiler false positives, 0 false negatives on static validity)
- **Full GLM Model**: 100.00% accuracy, F1 = 1.0000
- **Ablated GLM Model (without `total_error_count`)**: **80.00%** accuracy, F1 = **0.8421**

### Learned Feature Coefficients (Full GLM Log-Odds Impact):
- `total_error_count`: $+4.1956$
- `syntax_error_count`: $+1.0599$
- `unindexed_bam_count`: $+0.9282$
- `missing_parameter_count`: $+0.4912$
- `unresolved_reference_count`: $+0.4194$
- `file_extension_error_count`: $+0.3419$
- `tool_diversity_ratio`: $+0.3183$
- `warning_count`: $+0.2321$
- `has_alignment_step`: $-0.1640$
- `pipeline_length`: $-0.1621$
- `has_variant_calling_step`: $+0.0895$
- `has_qc_step`: $-0.0788$
- `sequence_violation_count`: $-0.0085$

---

## 8. Running the API Server

Start the FastAPI application with Uvicorn:
```bash
source venv/bin/activate
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## 9. API Endpoint Reference

### 1. `GET /health`
```bash
curl http://127.0.0.1:8000/health
```
```json
{
  "status": "healthy",
  "service": "genomic-pipeline-compiler-glm",
  "model_loaded": true,
  "model_version": "2.1.0-glm-scikit-learn==1.9.1",
  "supported_tools": ["FASTQC", "TRIMMOMATIC", "BWA", "SAMTOOLS", "GATK"]
}
```

### 2. `POST /analyze`
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_text": "FASTQC --in reads.fq --out qc.html\nBWA MEM --ref ref.fa --in reads.fq --out out.sam\nSAMTOOLS SORT --in out.sam --out sorted.bam\nSAMTOOLS INDEX --in sorted.bam\nGATK HaplotypeCaller --ref ref.fa --in sorted.bam --out vars.vcf",
    "declared_inputs": ["reads.fq", "ref.fa"]
  }'
```
Returns complete structured JSON with `validation`, `features`, `prediction` (probability, category, top factors), and `model_info`.

### 3. `POST /predict`
Supports either raw script or feature dictionaries:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "syntax_error_count": 0,
      "missing_parameter_count": 0,
      "sequence_violation_count": 0,
      "unresolved_reference_count": 0,
      "file_extension_error_count": 0,
      "total_error_count": 0,
      "warning_count": 0,
      "pipeline_length": 5,
      "has_qc_step": 1,
      "has_alignment_step": 1,
      "has_variant_calling_step": 1,
      "tool_diversity_ratio": 0.8,
      "unindexed_bam_count": 0
    }
  }'
```

---

## 10. Web Frontend Interface

The application features an interactive web interface:
- **Hero & Interactive 3D Visualization**: Real-time Three.js WebGL 3D DNA Double Helix scene responding to cursor movements and window resizing.
- **Dark Slate IDE Workspace**: Line numbers gutter, dynamic character/line counters, Tab indentation support, and keyboard shortcut (`⌘B` / `Ctrl+B`) for compilation.
- **Benchmark Sample Selector**: Populated dynamically from `GET /samples` (7 curated benchmark pipelines), plus quick one-click loaders for Clean WGS and Error Samples. Loading an example populates the editor with an informational notification banner without pretending that it has already been compiled.
- **Live Backend Health Telemetry**: Live ping indicator querying `GET /health` with dropdown displaying FastAPI status, port, and loaded model version (`2.1.0-glm-scikit-learn==1.9.1`).
- **Deterministic Diagnostics View**: Highlights syntax errors, semantic errors, parameter violations, line numbers, code snippets, and actionable compiler repair suggestions.
- **Topologically Sorted Execution Tree (DAG)**: Visualizes linearized workflow stages, tool names, subcommands, input files, and output artifacts.
- **GLM Operational Risk Telemetry**: Displays failure probability gauge (0.00 to 1.00 index), risk category badges (LOW, MEDIUM, HIGH), dynamic contributing factor weights, and a clear scientific disclaimer.
- **Integrated Serving**: Served directly by the FastAPI backend at `http://localhost:8000/` and `http://localhost:8000/index.html`.

---

## 11. Automated Testing

Run the automated pytest suite (62 tests across 6 test suites):
```bash
source venv/bin/activate
PYTHONPATH=backend pytest backend/tests -v
```

---

## 12. Known Limitations & Architectural Boundaries

- **Separation of Concerns**: The compiler frontend and semantic validator are the authoritative components responsible for lexical, syntax, semantic, and workflow validation. The GLM machine learning component is strictly supplementary, providing probabilistic risk scoring, probability calibration, and factor attribution based on extracted compiler metrics.
- **Not Proof of Real-World Cluster Execution Failure**: Labels in the dataset correspond to static syntactic, semantic, and workflow order validity. GLM predictions should not be construed as empirical proof or predictions of real-world cluster execution failures (such as cluster out-of-memory errors, disk quota exhaustion, compute node faults, or corrupt FASTA/BAM binaries).
- **Synthetic Training Domain**: The model is trained on synthetic benchmark pipeline mutations. While group-isolated holdout evaluations and independent benchmark tests confirm genuine learning on sub-diagnostics (achieving 93.33% on held-out groups and 80.00% on unseen tool combinations), it remains a demonstration model calibrated to synthetic benchmark distributions.
- **Simplified Tool Interfaces**: Command syntax represents an abstracted DSL mirroring the core flags (`--in`, `--out`, `--ref`, `--threads`, `--slidingwindow`, `--minlen`, `--metrics`) of standard bioinformatics tools. Full proprietary flags of native C/Java binaries are not exhaustively modeled.
- **Static Analysis Scope**: Compiler validation verifies syntactical structure, file references, parameter bounds, and workflow sequences statically without running underlying multi-gigabyte aligners or variant callers.

