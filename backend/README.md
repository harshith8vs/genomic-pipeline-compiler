# Intelligent Compiler for Detecting Errors in Genomic Data Processing Pipelines Using GLM (v2.1.0)

A modular bioinformatics pipeline analysis system combining:
1. **Authoritative Compiler & Semantic Engine**: Responsible for lexical tokenization, syntax validation, semantic rules, DAG file reference resolution, and biological workflow sequence auditing across standard genomic tools (`FASTQC`, `TRIMMOMATIC`, `BWA`, `SAMTOOLS`, `GATK`).
2. **Supplementary GLM Classifier**: A Generalized Linear Model (Logistic Regression with logit link) used strictly for **supplementary continuous risk scoring, probability calibration, and factor attribution** on extracted pipeline metrics. It does not serve as proof of real-world cluster execution failure or substitute for deterministic compiler checks.

See root `README.md` for complete architecture diagrams, leakage audit findings, group-aware cross-validation, baseline comparisons, and API documentation.
