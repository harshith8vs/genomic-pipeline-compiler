"""
Genomic Pipeline Semantic and Rule Validator.

Performs deep semantic validation:
- Required parameters per command and subcommand
- Parameter formats, types, and numeric bounds
- Incompatible file extensions
- Workflow sequence / stage order transition matrix
- Symbol table file resolution (DAG dataflow)
- Circular references, output metrics artifacts, and unindexed BAM checks
"""

import os
import re
from typing import List, Dict, Set, Tuple, Optional
from app.schemas import ParsedStep, ValidationError, ValidationReport

# Parameter specifications per tool / subcommand
COMMAND_SPECS = {
    "FASTQC": {
        "required": ["in"],
        "allowed": ["in", "input", "out", "output", "threads", "extract"],
        "formats": {
            "in": {"ext": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "desc": "FASTQ read file"},
            "out": {"ext": [".html", ".zip", ""], "desc": "HTML/ZIP report or directory"},
            "threads": {"type": "pos_int", "desc": "positive integer number of CPU threads"},
        }
    },
    "TRIMMOMATIC": {
        "required": ["in", "out"],
        "allowed": ["in", "out", "threads", "minlen", "slidingwindow", "leading", "trailing", "crop", "adapter"],
        "formats": {
            "in": {"ext": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "desc": "FASTQ read file"},
            "out": {"ext": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "desc": "trimmed FASTQ read file"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
            "minlen": {"type": "pos_int", "desc": "minimum length positive integer (e.g. 36)"},
            "slidingwindow": {"type": "slidingwindow", "desc": "window:quality (e.g. 4:20)"},
            "leading": {"type": "non_neg_int", "desc": "integer quality score"},
            "trailing": {"type": "non_neg_int", "desc": "integer quality score"},
            "crop": {"type": "pos_int", "desc": "positive integer length"},
        }
    },
    "BWA_MEM": {
        "required": ["ref", "in", "out"],
        "allowed": ["ref", "reference", "in", "input", "out", "output", "threads"],
        "formats": {
            "ref": {"ext": [".fa", ".fasta", ".fna"], "desc": "FASTA reference genome file"},
            "in": {"ext": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "desc": "FASTQ read file"},
            "out": {"ext": [".sam", ".bam"], "desc": "SAM/BAM alignment file"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
    "BWA_INDEX": {
        "required": ["ref"],
        "allowed": ["ref", "reference", "threads"],
        "formats": {
            "ref": {"ext": [".fa", ".fasta", ".fna"], "desc": "FASTA reference genome file"},
        }
    },
    "SAMTOOLS_VIEW": {
        "required": ["in", "out"],
        "allowed": ["in", "input", "out", "output", "threads", "b", "h", "s"],
        "formats": {
            "in": {"ext": [".sam", ".bam"], "desc": "SAM or BAM alignment file"},
            "out": {"ext": [".bam", ".sam"], "desc": "BAM or SAM output file"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
    "SAMTOOLS_SORT": {
        "required": ["in", "out"],
        "allowed": ["in", "input", "out", "output", "threads"],
        "formats": {
            "in": {"ext": [".bam", ".sam"], "desc": "SAM or BAM file to sort"},
            "out": {"ext": [".bam"], "desc": "coordinate-sorted BAM file"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
    "SAMTOOLS_INDEX": {
        "required": ["in"],
        "allowed": ["in", "input", "out", "output", "threads"],
        "formats": {
            "in": {"ext": [".bam"], "desc": "coordinate-sorted BAM file to index"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
    "GATK_HAPLOTYPECALLER": {
        "required": ["ref", "in", "out"],
        "allowed": ["ref", "reference", "in", "input", "out", "output", "threads", "stand-call-conf"],
        "formats": {
            "ref": {"ext": [".fa", ".fasta", ".fna"], "desc": "FASTA reference genome file"},
            "in": {"ext": [".bam"], "desc": "coordinate-sorted, indexed BAM file"},
            "out": {"ext": [".vcf", ".vcf.gz", ".g.vcf"], "desc": "VCF variant output file"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
    "GATK_MUTECT2": {
        "required": ["ref", "in", "out"],
        "allowed": ["ref", "reference", "in", "input", "out", "output", "threads"],
        "formats": {
            "ref": {"ext": [".fa", ".fasta", ".fna"], "desc": "FASTA reference genome file"},
            "in": {"ext": [".bam"], "desc": "coordinate-sorted, indexed BAM file"},
            "out": {"ext": [".vcf", ".vcf.gz"], "desc": "VCF somatic variant output file"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
    "GATK_MARKDUPLICATES": {
        "required": ["in", "out"],
        "allowed": ["in", "input", "out", "output", "metrics", "threads"],
        "formats": {
            "in": {"ext": [".bam"], "desc": "BAM alignment file"},
            "out": {"ext": [".bam"], "desc": "deduplicated BAM alignment file"},
            "metrics": {"ext": [".txt", ".metrics"], "desc": "metrics report output artifact"},
            "threads": {"type": "pos_int", "desc": "positive integer CPU threads"},
        }
    },
}


class PipelineValidator:
    """
    Performs semantic validation, workflow transition rules, and DAG file flow analysis.
    """

    def validate(
        self,
        steps: List[ParsedStep],
        syntax_errors: List[ValidationError],
        declared_inputs: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate all steps in a pipeline.

        Args:
            steps: List of parsed pipeline steps.
            syntax_errors: Errors caught during lexing/parsing.
            declared_inputs: Files assumed already available before pipeline runs.

        Returns:
            ValidationReport containing is_valid, errors, warnings, and summary.
        """
        all_errors: List[ValidationError] = list(syntax_errors)
        all_warnings: List[ValidationError] = []

        declared_set = set(declared_inputs or [])
        # Symbol table for tracking produced files: filename -> step_line_number
        available_files: Dict[str, int] = {f: 0 for f in declared_set}
        indexed_bams: Set[str] = set()

        # Track valid tool steps for sequence verification
        tools_seen: List[Tuple[str, int, str, ParsedStep]] = []

        for step in steps:
            # Skip deep semantic inspection if step had raw syntax/subcommand errors
            if not step.is_valid:
                continue

            tool_key = f"{step.tool}_{step.subcommand}" if step.subcommand else step.tool
            stage = self._get_step_stage(step)
            tools_seen.append((tool_key, step.line_number, stage, step))

            # 1. Parameter presence and unrecognized parameters
            param_errors, param_warnings = self._check_parameters(step, tool_key)
            all_errors.extend(param_errors)
            all_warnings.extend(param_warnings)

            # 2. Parameter format and file extension checks
            fmt_errors = self._check_parameter_formats(step, tool_key)
            all_errors.extend(fmt_errors)

            # 3. In-place file conflict check (reading and writing to exact same file)
            conflict_errors = self._check_file_conflicts(step)
            all_errors.extend(conflict_errors)

            # 4. File reference resolution (DAG check)
            ref_errors, ref_warnings = self._check_file_references(
                step, available_files, declared_set
            )
            all_errors.extend(ref_errors)
            all_warnings.extend(ref_warnings)

            # 5. GATK unindexed BAM check
            if tool_key in ("GATK_HAPLOTYPECALLER", "GATK_MUTECT2"):
                bam_in = step.parameters.get("in") or step.parameters.get("input")
                if bam_in and bam_in not in indexed_bams:
                    all_errors.append(
                        ValidationError(
                            line_number=step.line_number,
                            command=step.raw_command,
                            error_type="UNINDEXED_BAM_INPUT",
                            severity="ERROR",
                            message=f"GATK tool '{step.tool} {step.subcommand}' requires an indexed BAM, but '{bam_in}' has not been indexed with 'SAMTOOLS INDEX'.",
                            suggestion=f"Add 'SAMTOOLS INDEX --in {bam_in}' before line {step.line_number}."
                        )
                    )

            # Register files produced by this step into symbol table
            for out_file in step.output_files:
                available_files[out_file] = step.line_number

            # If this was SAMTOOLS INDEX, mark the input bam as indexed
            if tool_key == "SAMTOOLS_INDEX":
                indexed_bam = step.parameters.get("in") or step.parameters.get("input")
                if indexed_bam:
                    indexed_bams.add(indexed_bam)
                    # Register implicit index file (.bai) in symbol table
                    available_files[f"{indexed_bam}.bai"] = step.line_number

        # 6. Workflow sequence transition validation
        seq_errors, seq_warnings = self._check_workflow_sequences(steps, tools_seen)
        all_errors.extend(seq_errors)
        all_warnings.extend(seq_warnings)

        # Classify overall status
        has_errors = any(e.severity == "ERROR" for e in all_errors)
        if has_errors:
            status = "FAILED_WITH_ERRORS"
            is_valid = False
        elif all_warnings:
            status = "VALID_WITH_WARNINGS"
            is_valid = True
        else:
            status = "VALID"
            is_valid = True

        return ValidationReport(
            is_valid=is_valid,
            status=status,
            total_errors=len(all_errors),
            total_warnings=len(all_warnings),
            errors=all_errors,
            warnings=all_warnings,
            parsed_steps=steps
        )

    def _get_step_stage(self, step: ParsedStep) -> str:
        """Determine pipeline stage for a step."""
        tool_key = f"{step.tool}_{step.subcommand}" if step.subcommand else step.tool
        if tool_key == "BWA_INDEX":
            return "INDEX_REF"
        if tool_key == "BWA_MEM":
            return "ALIGNMENT"
        if step.tool in ("FASTQC", "TRIMMOMATIC"):
            return "QC"
        if step.tool == "SAMTOOLS":
            return "PROCESSING"
        if step.tool == "GATK":
            if step.subcommand == "MARKDUPLICATES":
                return "PROCESSING"
            return "VARIANT_CALLING"
        return "UNKNOWN"

    def _check_parameters(
        self, step: ParsedStep, tool_key: str
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate required and unrecognized parameters."""
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        spec = COMMAND_SPECS.get(tool_key)
        if not spec:
            return errors, warnings

        # Required parameters
        alias_map = {
            "in": ["in", "input"],
            "out": ["out", "output"],
            "ref": ["ref", "reference"],
            "metrics": ["metrics"]
        }

        for req in spec["required"]:
            aliases = alias_map.get(req, [req])
            if not any(alias in step.parameters for alias in aliases):
                display_tool = f"{step.tool} {step.subcommand}" if step.subcommand else step.tool
                errors.append(
                    ValidationError(
                        line_number=step.line_number,
                        command=step.raw_command,
                        error_type="MISSING_PARAMETER",
                        severity="ERROR",
                        message=f"Command '{display_tool}' is missing required parameter '--{req}'.",
                        suggestion=f"Add '--{req} <filepath>' to the command on line {step.line_number}."
                    )
                )

        # Unrecognized parameters
        allowed = set(spec.get("allowed", []))
        for param in step.parameters:
            if param not in allowed:
                display_tool = f"{step.tool} {step.subcommand}" if step.subcommand else step.tool
                warnings.append(
                    ValidationError(
                        line_number=step.line_number,
                        command=step.raw_command,
                        error_type="UNRECOGNIZED_PARAMETER",
                        severity="WARNING",
                        message=f"Parameter '--{param}' is not recognized for '{display_tool}'. Recognized parameters: {sorted(list(allowed))}.",
                        suggestion=f"Verify if '--{param}' is intentional or remove it."
                    )
                )

        return errors, warnings

    def _check_parameter_formats(
        self, step: ParsedStep, tool_key: str
    ) -> List[ValidationError]:
        """Validate parameter value formats, integers, and file extensions."""
        errors: List[ValidationError] = []
        spec = COMMAND_SPECS.get(tool_key)
        if not spec or "formats" not in spec:
            return errors

        formats = spec["formats"]
        alias_map = {"input": "in", "output": "out", "reference": "ref"}

        for param_name, param_val in step.parameters.items():
            canonical_name = alias_map.get(param_name, param_name)
            if canonical_name not in formats:
                continue

            rule = formats[canonical_name]

            # 1. File extension validation
            if "ext" in rule and isinstance(param_val, str):
                valid_exts = rule["ext"]
                # Allow directory or extensionless output directory (e.g. fastqc out dir)
                if "" in valid_exts and ("/" in param_val or not "." in os.path.basename(param_val)):
                    continue

                val_lower = param_val.lower()
                matched = any(val_lower.endswith(ext) for ext in valid_exts if ext)
                if not matched:
                    errors.append(
                        ValidationError(
                            line_number=step.line_number,
                            command=step.raw_command,
                            error_type="INVALID_FILE_EXTENSION",
                            severity="ERROR",
                            message=f"Parameter '--{param_name}' specified '{param_val}', which does not match expected file type for {rule.get('desc', 'this parameter')}. Expected extensions: {valid_exts}",
                            suggestion=f"Provide a file with one of the valid extensions: {', '.join(valid_exts)}"
                        )
                    )

            # 2. Format / Type validation
            rule_type = rule.get("type")
            if rule_type == "pos_int":
                try:
                    int_val = int(param_val)
                    if int_val <= 0:
                        raise ValueError()
                except (ValueError, TypeError):
                    errors.append(
                        ValidationError(
                            line_number=step.line_number,
                            command=step.raw_command,
                            error_type="INVALID_PARAMETER_VALUE",
                            severity="ERROR",
                            message=f"Parameter '--{param_name}' value '{param_val}' must be a positive integer (> 0).",
                            suggestion=f"Specify a positive whole number for '--{param_name}', e.g., '--{param_name} 4'."
                        )
                    )
            elif rule_type == "non_neg_int":
                try:
                    int_val = int(param_val)
                    if int_val < 0:
                        raise ValueError()
                except (ValueError, TypeError):
                    errors.append(
                        ValidationError(
                            line_number=step.line_number,
                            command=step.raw_command,
                            error_type="INVALID_PARAMETER_VALUE",
                            severity="ERROR",
                            message=f"Parameter '--{param_name}' value '{param_val}' must be a non-negative integer (>= 0).",
                            suggestion=f"Specify a number >= 0 for '--{param_name}'."
                        )
                    )
            elif rule_type == "slidingwindow":
                if not re.match(r"^\d+:\d+$", str(param_val)):
                    errors.append(
                        ValidationError(
                            line_number=step.line_number,
                            command=step.raw_command,
                            error_type="INVALID_PARAMETER_VALUE",
                            severity="ERROR",
                            message=f"Parameter '--slidingwindow' value '{param_val}' has invalid format. Expected '<window_size>:<required_quality>' (e.g., '4:20').",
                            suggestion="Provide slidingwindow formatted as 'window:quality', for example '--slidingwindow 4:20'."
                        )
                    )

        return errors

    def _check_file_conflicts(self, step: ParsedStep) -> List[ValidationError]:
        """Detect when a command specifies the exact same file as input and output."""
        errors: List[ValidationError] = []
        in_files = set(step.input_files)
        out_files = set(step.output_files)

        overlap = in_files.intersection(out_files)
        if overlap:
            for conf_file in overlap:
                errors.append(
                    ValidationError(
                        line_number=step.line_number,
                        command=step.raw_command,
                        error_type="OUTPUT_REFERENCE_CONFLICT",
                        severity="ERROR",
                        message=f"Command specifies '{conf_file}' as both input and output. In-place overwrite causes pipeline file corruption.",
                        suggestion=f"Write output to a new distinct file (e.g. '{conf_file.rsplit('.', 1)[0]}_processed.{conf_file.rsplit('.', 1)[1]}')."
                    )
                )

        return errors

    def _check_file_references(
        self,
        step: ParsedStep,
        available_files: Dict[str, int],
        declared_set: Set[str]
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Verify that input files referenced exist either as initial inputs or previous outputs."""
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        for in_file in step.input_files:
            if in_file in available_files:
                continue

            file_lower = in_file.lower()
            # 1. Reference genome is always an initial input
            is_ref_genome = any(file_lower.endswith(ext) for ext in [".fa", ".fasta", ".fna"])

            # 2. Raw sequencing reads (fastq) used in preprocessing/QC or initial alignment
            is_raw_read = any(file_lower.endswith(ext) for ext in [".fastq", ".fq", ".fastq.gz", ".fq.gz"])
            is_initial_read = is_raw_read and (step.stage in ("QC", "ALIGNMENT"))

            if is_ref_genome or is_initial_read:
                # Register as an inferred initial input so subsequent steps can reference it
                available_files[in_file] = 0
                if in_file not in declared_set:
                    warnings.append(
                        ValidationError(
                            line_number=step.line_number,
                            command=step.raw_command,
                            error_type="UNDECLARED_INITIAL_INPUT",
                            severity="WARNING",
                            message=f"File '{in_file}' is referenced as input but was not explicitly declared in initial inputs. Inferred as an initial source file.",
                            suggestion=f"Add '{in_file}' to the declared inputs list to ensure reproducible pipeline validation."
                        )
                    )
            else:
                errors.append(
                    ValidationError(
                        line_number=step.line_number,
                        command=step.raw_command,
                        error_type="UNRESOLVED_FILE_REFERENCE",
                        severity="ERROR",
                        message=f"Input file '{in_file}' is undefined and cannot be resolved. It was neither declared as an input nor generated by any preceding pipeline step.",
                        suggestion=f"Ensure an earlier step outputs '{in_file}', or correct the filename if misspelled."
                    )
                )

        return errors, warnings

    def _check_workflow_sequences(
        self, steps: List[ParsedStep], tools_seen: List[Tuple[str, int, str, ParsedStep]]
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """
        Enforce biological workflow sequence rules based on command transitions.
        Allows legitimate branching workflows while blocking invalid operations.
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Find first occurrences of key pipeline milestones
        first_align_line: Optional[int] = None
        first_variant_line: Optional[int] = None
        first_sort_line: Optional[int] = None
        first_index_line: Optional[int] = None
        first_ref_index_line: Optional[int] = None

        for tool_key, line_num, _, _ in tools_seen:
            if tool_key == "BWA_INDEX" and first_ref_index_line is None:
                first_ref_index_line = line_num
            elif tool_key == "BWA_MEM" and first_align_line is None:
                first_align_line = line_num
            elif tool_key in ("GATK_HAPLOTYPECALLER", "GATK_MUTECT2") and first_variant_line is None:
                first_variant_line = line_num
            elif tool_key == "SAMTOOLS_SORT" and first_sort_line is None:
                first_sort_line = line_num
            elif tool_key == "SAMTOOLS_INDEX" and first_index_line is None:
                first_index_line = line_num

        # Check sequence rules for each tool occurrence
        for tool_key, line_num, _, _ in tools_seen:
            # Rule 1: TRIMMOMATIC cannot run after alignment or variant calling
            if tool_key == "TRIMMOMATIC":
                if first_align_line is not None and line_num > first_align_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command="TRIMMOMATIC",
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"Read trimming (TRIMMOMATIC) on line {line_num} cannot be executed after alignment (line {first_align_line}).",
                            suggestion="Move read trimming to the pre-processing stage before sequence alignment."
                        )
                    )
                elif first_variant_line is not None and line_num > first_variant_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command="TRIMMOMATIC",
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"Read trimming (TRIMMOMATIC) on line {line_num} cannot be executed after variant calling (line {first_variant_line}).",
                            suggestion="Move read trimming to the pre-processing stage before variant calling."
                        )
                    )

            # Rule 2: BWA INDEX must run before BWA MEM if BWA INDEX is present
            if tool_key == "BWA_MEM" and first_ref_index_line is not None:
                if line_num < first_ref_index_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command="BWA MEM",
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"Sequence alignment (BWA MEM) on line {line_num} precedes reference indexing (BWA INDEX on line {first_ref_index_line}).",
                            suggestion="Execute 'BWA INDEX' before 'BWA MEM'."
                        )
                    )

            # Rule 3: SAMTOOLS INDEX must follow SAMTOOLS SORT
            if tool_key == "SAMTOOLS_INDEX":
                if first_sort_line is None or line_num < first_sort_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command="SAMTOOLS INDEX",
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"SAMTOOLS INDEX on line {line_num} was invoked before coordinate sorting (SAMTOOLS SORT). An unsorted BAM file cannot be indexed.",
                            suggestion=f"Execute 'SAMTOOLS SORT' before running 'SAMTOOLS INDEX' on line {line_num}."
                        )
                    )

            # Rule 4: GATK MarkDuplicates cannot run after variant calling
            if tool_key == "GATK_MARKDUPLICATES":
                if first_variant_line is not None and line_num > first_variant_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command="GATK MarkDuplicates",
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"Duplicate read marking (GATK MarkDuplicates) on line {line_num} cannot be executed after variant calling (line {first_variant_line}).",
                            suggestion="Execute 'GATK MarkDuplicates' during alignment processing before variant discovery."
                        )
                    )

            # Rule 5: Variant Calling (GATK HaplotypeCaller / Mutect2) requires prior alignment
            if tool_key in ("GATK_HAPLOTYPECALLER", "GATK_MUTECT2"):
                if first_align_line is None or line_num < first_align_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command=tool_key,
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"Variant calling ({tool_key}) on line {line_num} requires aligned reads, but no alignment step (BWA MEM) preceded it.",
                            suggestion=f"Perform sequence alignment using 'BWA MEM' prior to variant calling on line {line_num}."
                        )
                    )
                elif first_sort_line is not None and line_num < first_sort_line:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command=tool_key,
                            error_type="INVALID_COMMAND_SEQUENCE",
                            severity="ERROR",
                            message=f"Variant calling ({tool_key}) on line {line_num} was invoked before coordinate sorting (SAMTOOLS SORT on line {first_sort_line}).",
                            suggestion="Sort alignments with 'SAMTOOLS SORT' before running variant discovery."
                        )
                    )

            # Rule 6: FASTQC after variant calling is unusual
            if tool_key == "FASTQC" and first_variant_line is not None and line_num > first_variant_line:
                warnings.append(
                    ValidationError(
                        line_number=line_num,
                        command="FASTQC",
                        error_type="UNUSUAL_STEP_SEQUENCE",
                        severity="WARNING",
                        message=f"FASTQC is executed on line {line_num} after variant discovery (line {first_variant_line}). Read QC is typically performed upstream.",
                        suggestion="Consider running FASTQC in the initial preprocessing stage."
                    )
                )

        # Warning: variant calling or alignment present without any QC
        has_qc = any(s == "QC" for _, _, s, _ in tools_seen)
        has_alignment = first_align_line is not None
        has_variant = first_variant_line is not None

        if (has_alignment or has_variant) and not has_qc and steps:
            warnings.append(
                ValidationError(
                    line_number=1,
                    command=steps[0].tool,
                    error_type="MISSING_QC_STAGE",
                    severity="WARNING",
                    message="Pipeline performs alignment/variant calling without any prior Quality Control step (FASTQC or TRIMMOMATIC).",
                    suggestion="Add FASTQC and TRIMMOMATIC as initial steps to filter adapter contamination and low-quality bases."
                )
            )

        return errors, warnings
