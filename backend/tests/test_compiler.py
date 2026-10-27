"""
Comprehensive Unit Tests for the Genomic Pipeline Compiler and Semantic Validator.

Covers:
- Standard valid pipelines
- Syntax errors, unknown tools, unknown and missing subcommands
- Tokenization edge cases: spaces in quotes, negative values, --flag=val, duplicate flags, missing values
- Inline and full-line comments
- GATK MarkDuplicates metrics output artifact handling
- Table-driven workflow order transitions
- DAG file reference resolution and circular overwrites
- Unindexed BAM detection
"""

import pytest
from app.compiler import PipelineCompiler
from app.validator import PipelineValidator


@pytest.fixture
def compiler():
    return PipelineCompiler()


@pytest.fixture
def validator():
    return PipelineValidator()


def test_valid_genomic_pipeline(compiler, validator):
    """Test that a fully valid WGS pipeline passes compiler and validator cleanly."""
    script = """
    FASTQC --in sample_raw.fastq --out qc_report.html
    TRIMMOMATIC --in sample_raw.fastq --out sample_trimmed.fq --minlen 36 --slidingwindow 4:20
    BWA MEM --ref reference.fa --in sample_trimmed.fq --out sample_aligned.sam --threads 8
    SAMTOOLS VIEW --in sample_aligned.sam --out sample_aligned.bam --threads 4
    SAMTOOLS SORT --in sample_aligned.bam --out sample_sorted.bam --threads 4
    SAMTOOLS INDEX --in sample_sorted.bam
    GATK HaplotypeCaller --ref reference.fa --in sample_sorted.bam --out raw_variants.vcf
    """
    steps, syntax_errors = compiler.parse(script)
    assert len(syntax_errors) == 0
    assert len(steps) == 7

    report = validator.validate(steps, syntax_errors, declared_inputs=["sample_raw.fastq", "reference.fa"])
    assert report.is_valid is True
    assert report.total_errors == 0
    assert report.status in ("VALID", "VALID_WITH_WARNINGS")


def test_invalid_and_missing_subcommands(compiler, validator):
    """Test that invalid subcommands and missing subcommands produce clear diagnostics and are marked invalid."""
    script = """
    BWA INVALID_SUB --ref ref.fa --in reads.fq --out out.sam
    SAMTOOLS --in sample.bam --out sorted.bam
    GATK FAKE_COMMAND --ref ref.fa --in in.bam --out out.vcf
    """
    steps, syntax_errors = compiler.parse(script)
    assert len(syntax_errors) >= 3

    # Check error types and actionable suggestions
    err_types = [e.error_type for e in syntax_errors]
    assert "UNKNOWN_SUBCOMMAND" in err_types
    assert "MISSING_SUBCOMMAND" in err_types

    # Ensure steps are marked invalid so validator handles them safely
    assert all(not step.is_valid for step in steps)

    report = validator.validate(steps, syntax_errors)
    assert report.is_valid is False
    assert report.total_errors >= 3


def test_invalid_step_embedded_in_valid_pipeline(compiler, validator):
    """An invalid step embedded in a valid pipeline must be flagged without crashing downstream validation."""
    script = """
    FASTQC --in reads.fq --out qc.html
    BWA NONEXISTENT_SUB --ref ref.fa --in reads.fq --out out.sam
    SAMTOOLS SORT --in out.sam --out sorted.bam
    """
    steps, syntax_errors = compiler.parse(script)
    assert any(e.error_type == "UNKNOWN_SUBCOMMAND" for e in syntax_errors)

    report = validator.validate(steps, syntax_errors, declared_inputs=["reads.fq", "ref.fa"])
    assert report.is_valid is False
    assert any(e.line_number == 3 for e in report.errors)


def test_quoted_paths_with_spaces(compiler, validator):
    """Paths with spaces inside quotes should parse cleanly."""
    script = """
    FASTQC --in "my sample reads.fastq" --out "my qc report.html"
    """
    steps, syntax_errors = compiler.parse(script)
    assert len(syntax_errors) == 0
    assert len(steps) == 1
    assert steps[0].parameters["in"] == "my sample reads.fastq"
    assert steps[0].parameters["out"] == "my qc report.html"


def test_flag_equals_value_syntax(compiler, validator):
    """Syntax like --flag=value must be parsed into flag and value."""
    script = """
    BWA MEM --ref=reference.fa --in=reads.fq --out=aligned.sam --threads=8
    """
    steps, syntax_errors = compiler.parse(script)
    assert len(syntax_errors) == 0
    assert steps[0].parameters["ref"] == "reference.fa"
    assert steps[0].parameters["threads"] == "8"


def test_negative_numeric_parameter(compiler, validator):
    """A negative number like --threads -4 should be tokenized and flagged as invalid integer bound."""
    script = """
    FASTQC --in reads.fq --threads -4
    """
    steps, syntax_errors = compiler.parse(script)
    report = validator.validate(steps, syntax_errors, declared_inputs=["reads.fq"])
    assert report.is_valid is False
    assert any(e.error_type == "INVALID_PARAMETER_VALUE" for e in report.errors)
    assert any("positive integer" in e.message for e in report.errors)


def test_duplicate_parameter_flags(compiler, validator):
    """Specifying the same flag twice must be detected as an error."""
    script = """
    FASTQC --in reads.fq --threads 4 --threads 8
    """
    _, syntax_errors = compiler.parse(script)
    assert any(e.error_type == "DUPLICATE_PARAMETER" for e in syntax_errors)


def test_missing_parameter_value(compiler, validator):
    """Flags that expect a value must not silently become boolean flags if value is omitted."""
    script = """
    FASTQC --in
    """
    _, syntax_errors = compiler.parse(script)
    assert any(e.error_type == "MISSING_PARAMETER_VALUE" for e in syntax_errors)


def test_comments_full_line_and_inline(compiler, validator):
    """Both full-line comments and inline comments (# and //) must be ignored."""
    script = """
    # Full line comment
    FASTQC --in reads.fq --out qc.html # Inline hash comment
    // Another full line comment
    BWA MEM --ref ref.fa --in reads.fq --out sample.sam // Inline slash comment
    """
    steps, syntax_errors = compiler.parse(script)
    assert len(syntax_errors) == 0
    assert len(steps) == 2
    assert steps[0].tool == "FASTQC"
    assert steps[1].tool == "BWA"


def test_gatk_markduplicates_metrics_is_output_artifact(compiler, validator):
    """
    For GATK MarkDuplicates, --metrics must be treated as an output artifact,
    not checked as an input, and made available in the symbol table.
    """
    script = """
    BWA MEM --ref ref.fa --in reads.fq --out sample.sam
    SAMTOOLS SORT --in sample.sam --out sample_sorted.bam
    GATK MarkDuplicates --in sample_sorted.bam --out sample_dedup.bam --metrics dup_report.metrics
    SAMTOOLS INDEX --in sample_dedup.bam
    """
    steps, syntax_errors = compiler.parse(script)
    assert len(syntax_errors) == 0

    # Verify compiler intermediate representation puts metrics in output_files
    markdup_step = steps[2]
    assert markdup_step.tool == "GATK"
    assert markdup_step.subcommand == "MARKDUPLICATES"
    assert "dup_report.metrics" in markdup_step.output_files
    assert "dup_report.metrics" not in markdup_step.input_files

    # Verify validator does not complain about dup_report.metrics being undefined
    report = validator.validate(steps, syntax_errors, declared_inputs=["reads.fq", "ref.fa"])
    assert report.is_valid is True
    assert not any("dup_report.metrics" in e.message for e in report.errors)


@pytest.mark.parametrize(
    "script,expected_valid,error_type_substr",
    [
        # Valid: Trimming -> Alignment -> Sort -> Index -> Variant Calling
        (
            "TRIMMOMATIC --in reads.fq --out clean.fq --minlen 36 --slidingwindow 4:20\n"
            "BWA MEM --ref ref.fa --in clean.fq --out out.sam\n"
            "SAMTOOLS SORT --in out.sam --out sorted.bam\n"
            "SAMTOOLS INDEX --in sorted.bam\n"
            "GATK HaplotypeCaller --ref ref.fa --in sorted.bam --out vars.vcf",
            True,
            None,
        ),
        # Valid: Reference indexing before alignment
        (
            "BWA INDEX --ref ref.fa\n"
            "BWA MEM --ref ref.fa --in reads.fq --out out.sam\n"
            "SAMTOOLS SORT --in out.sam --out sorted.bam\n"
            "SAMTOOLS INDEX --in sorted.bam",
            True,
            None,
        ),
        # Invalid: BWA MEM before BWA INDEX when index is in pipeline
        (
            "BWA MEM --ref ref.fa --in reads.fq --out out.sam\n"
            "BWA INDEX --ref ref.fa\n"
            "SAMTOOLS SORT --in out.sam --out sorted.bam",
            False,
            "INVALID_COMMAND_SEQUENCE",
        ),
        # Invalid: Trimming after alignment
        (
            "BWA MEM --ref ref.fa --in reads.fq --out out.sam\n"
            "TRIMMOMATIC --in reads.fq --out clean.fq --minlen 36",
            False,
            "INVALID_COMMAND_SEQUENCE",
        ),
        # Invalid: GATK MarkDuplicates after variant calling
        (
            "BWA MEM --ref ref.fa --in reads.fq --out out.sam\n"
            "SAMTOOLS SORT --in out.sam --out sorted.bam\n"
            "SAMTOOLS INDEX --in sorted.bam\n"
            "GATK HaplotypeCaller --ref ref.fa --in sorted.bam --out vars.vcf\n"
            "GATK MarkDuplicates --in sorted.bam --out dedup.bam --metrics m.txt",
            False,
            "INVALID_COMMAND_SEQUENCE",
        ),
        # Invalid: SAMTOOLS INDEX before SAMTOOLS SORT
        (
            "BWA MEM --ref ref.fa --in reads.fq --out out.sam\n"
            "SAMTOOLS INDEX --in out.sam\n"
            "SAMTOOLS SORT --in out.sam --out sorted.bam",
            False,
            "INVALID_COMMAND_SEQUENCE",
        ),
    ]
)
def test_workflow_transitions(compiler, validator, script, expected_valid, error_type_substr):
    steps, syntax_errs = compiler.parse(script)
    report = validator.validate(steps, syntax_errs, declared_inputs=["reads.fq", "ref.fa"])
    if expected_valid:
        assert report.is_valid is True
        assert report.total_errors == 0
    else:
        assert report.is_valid is False
        assert any(e.error_type == error_type_substr for e in report.errors)


def test_pipeline_with_missing_parameters(compiler, validator):
    """Test missing mandatory parameters."""
    script = """
    FASTQC --in reads.fastq
    BWA MEM --in reads.fastq
    """
    steps, syntax_errors = compiler.parse(script)
    report = validator.validate(steps, syntax_errors)
    assert report.is_valid is False
    missing_errs = [e for e in report.errors if e.error_type == "MISSING_PARAMETER"]
    assert len(missing_errs) >= 2


def test_pipeline_with_undefined_file_references(compiler, validator):
    """Test unresolved intermediate files."""
    script = """
    FASTQC --in sample.fastq --out qc.html
    SAMTOOLS SORT --in mysterious_nonexistent_reads.bam --out sorted.bam
    """
    steps, syntax_errors = compiler.parse(script)
    report = validator.validate(steps, syntax_errors)
    assert report.is_valid is False
    unresolved = [e for e in report.errors if e.error_type == "UNRESOLVED_FILE_REFERENCE"]
    assert len(unresolved) == 1
    assert "mysterious_nonexistent_reads.bam" in unresolved[0].message


def test_pipeline_with_unindexed_bam_for_gatk(compiler, validator):
    """Test that GATK HaplotypeCaller flags unindexed BAM input."""
    script = """
    BWA MEM --ref ref.fa --in reads.fq --out sample.sam
    SAMTOOLS SORT --in sample.sam --out sample_sorted.bam
    GATK HaplotypeCaller --ref ref.fa --in sample_sorted.bam --out vars.vcf
    """
    steps, syntax_errors = compiler.parse(script)
    report = validator.validate(steps, syntax_errors)
    assert report.is_valid is False
    assert any(e.error_type == "UNINDEXED_BAM_INPUT" for e in report.errors)


def test_pipeline_with_circular_overwrite_conflict(compiler, validator):
    """Test detection of in-place overwrite corruption."""
    script = """
    SAMTOOLS SORT --in alignment.bam --out alignment.bam
    """
    steps, syntax_errors = compiler.parse(script)
    report = validator.validate(steps, syntax_errors)
    assert report.is_valid is False
    assert any(e.error_type == "OUTPUT_REFERENCE_CONFLICT" for e in report.errors)
