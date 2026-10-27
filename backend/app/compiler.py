"""
Genomic Pipeline Lexer and Parser (Compiler Engine).

This module tokenizes and parses human-readable genomic pipeline scripts
into structured Intermediate Representations (ParsedStep) and catches raw syntax errors.
"""

import shlex
import re
from typing import List, Tuple, Dict, Any, Optional
from app.schemas import ParsedStep, ValidationError

# Supported genomic tools and their stages
TOOL_STAGES = {
    "FASTQC": "QC",
    "TRIMMOMATIC": "QC",
    "BWA": "ALIGNMENT",
    "SAMTOOLS": "PROCESSING",
    "GATK": "VARIANT_CALLING",
}

# Subcommands recognized per tool
KNOWN_SUBCOMMANDS = {
    "BWA": {"MEM", "INDEX"},
    "SAMTOOLS": {"VIEW", "SORT", "INDEX"},
    "GATK": {"HAPLOTYPECALLER", "MUTECT2", "MARKDUPLICATES"},
}

# Flags that are known standalone boolean switches (do not take values)
KNOWN_BOOLEAN_FLAGS = {"b", "h", "s", "include-header", "extract"}


def is_numeric_literal(token: str) -> bool:
    """Check if token is a positive or negative integer or float."""
    return bool(re.match(r"^[-+]?\d+(\.\d+)?$", token))


class PipelineCompiler:
    """
    Compiler frontend for the simplified genomic pipeline language.
    Parses pipeline script lines into structured AST steps and emits syntax errors.
    """

    def parse(self, script_text: str) -> Tuple[List[ParsedStep], List[ValidationError]]:
        """
        Tokenize and parse a pipeline script.

        Args:
            script_text: Multi-line string containing pipeline commands.

        Returns:
            Tuple of (parsed_steps, syntax_errors)
        """
        parsed_steps: List[ParsedStep] = []
        errors: List[ValidationError] = []

        if not script_text or not script_text.strip():
            errors.append(
                ValidationError(
                    line_number=0,
                    command=None,
                    error_type="EMPTY_PIPELINE",
                    severity="ERROR",
                    message="Pipeline script is empty. Provide at least one genomic command step.",
                    suggestion="Add valid commands such as 'FASTQC --in reads.fq' or 'BWA MEM --ref ref.fa --in reads.fq --out out.sam'."
                )
            )
            return parsed_steps, errors

        lines = script_text.splitlines()

        for line_idx, raw_line in enumerate(lines, start=1):
            line_str = raw_line.strip()

            # Skip empty lines
            if not line_str:
                continue

            # Strip C-style inline comments (//)
            if "//" in line_str:
                line_str = line_str.split("//", 1)[0].strip()
                if not line_str:
                    continue

            # Skip full-line hash comments
            if line_str.startswith("#"):
                continue

            try:
                # shlex with comments=True strips inline '#' comments cleanly
                tokens = shlex.split(line_str, comments=True)
            except ValueError as e:
                errors.append(
                    ValidationError(
                        line_number=line_idx,
                        command=None,
                        error_type="SYNTAX_ERROR",
                        severity="ERROR",
                        message=f"Lexical tokenization error: {str(e)}",
                        suggestion="Check for unmatched quotes or malformed characters on this line."
                    )
                )
                continue

            if not tokens:
                continue

            step, step_errors = self._parse_line_tokens(line_idx, raw_line.strip(), tokens)
            if step_errors:
                errors.extend(step_errors)
            if step is not None:
                parsed_steps.append(step)

        return parsed_steps, errors

    def _parse_line_tokens(
        self, line_num: int, raw_line: str, tokens: List[str]
    ) -> Tuple[Optional[ParsedStep], List[ValidationError]]:
        """Parse tokenized line into a ParsedStep."""
        errors: List[ValidationError] = []
        step_is_valid = True
        raw_tool = tokens[0].upper()

        # Check for unknown top-level tool
        if raw_tool not in TOOL_STAGES:
            suggestion = self._suggest_tool_alternative(raw_tool)
            errors.append(
                ValidationError(
                    line_number=line_num,
                    command=tokens[0],
                    error_type="UNKNOWN_COMMAND",
                    severity="ERROR",
                    message=f"Command '{tokens[0]}' is not a recognized genomic tool in this compiler.",
                    suggestion=suggestion
                )
            )
            # Unknown tool step is rejected entirely from AST
            return None, errors

        current_idx = 1
        subcommand: Optional[str] = None

        # Check if this tool uses subcommands
        if raw_tool in KNOWN_SUBCOMMANDS:
            if current_idx < len(tokens) and not tokens[current_idx].startswith("-"):
                sub_candidate = tokens[current_idx].upper()
                valid_subs = KNOWN_SUBCOMMANDS[raw_tool]
                if sub_candidate in valid_subs:
                    subcommand = sub_candidate
                    current_idx += 1
                else:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command=f"{raw_tool} {tokens[current_idx]}",
                            error_type="UNKNOWN_SUBCOMMAND",
                            severity="ERROR",
                            message=f"Tool '{raw_tool}' does not have subcommand '{tokens[current_idx]}'. Recognized subcommands: {sorted(list(valid_subs))}",
                            suggestion=f"Use one of: {', '.join([f'{raw_tool} {s}' for s in sorted(list(valid_subs))])}"
                        )
                    )
                    step_is_valid = False
                    current_idx += 1
            else:
                valid_subs = sorted(list(KNOWN_SUBCOMMANDS[raw_tool]))
                errors.append(
                    ValidationError(
                        line_number=line_num,
                        command=raw_tool,
                        error_type="MISSING_SUBCOMMAND",
                        severity="ERROR",
                        message=f"Tool '{raw_tool}' requires a subcommand (e.g. {valid_subs}).",
                        suggestion=f"Specify a subcommand, e.g. '{raw_tool} {valid_subs[0]}'."
                    )
                )
                step_is_valid = False

        # Parse parameter flags and arguments
        parameters: Dict[str, Any] = {}
        input_files: List[str] = []
        output_files: List[str] = []

        while current_idx < len(tokens):
            token = tokens[current_idx]

            # Case A: Numeric literal outside a flag (e.g. unexpected negative or positive number)
            if is_numeric_literal(token) and not token.startswith("--"):
                errors.append(
                    ValidationError(
                        line_number=line_num,
                        command=f"{raw_tool} {subcommand}" if subcommand else raw_tool,
                        error_type="UNEXPECTED_ARGUMENT",
                        severity="ERROR",
                        message=f"Unexpected numeric argument '{token}'. Numbers must be preceded by a parameter flag.",
                        suggestion=f"Specify a flag for this value (e.g. '--threads {token}' or '--minlen {token}')."
                    )
                )
                step_is_valid = False
                current_idx += 1
                continue

            # Case B: Flag syntax (--flag or -f or --flag=value)
            if token.startswith("-"):
                # Handle --flag=value syntax
                if "=" in token:
                    parts = token.split("=", 1)
                    flag_name = parts[0].lstrip("-").lower()
                    val = parts[1].strip("\"'")

                    if flag_name in parameters:
                        errors.append(
                            ValidationError(
                                line_number=line_num,
                                command=f"{raw_tool} {subcommand}" if subcommand else raw_tool,
                                error_type="DUPLICATE_PARAMETER",
                                severity="ERROR",
                                message=f"Parameter '--{flag_name}' is specified multiple times in command.",
                                suggestion=f"Remove duplicate '--{flag_name}' parameter."
                            )
                        )
                        step_is_valid = False

                    parameters[flag_name] = val
                    self._register_file_parameter(raw_tool, subcommand, flag_name, val, input_files, output_files)
                    current_idx += 1
                    continue

                flag_name = token.lstrip("-").lower()

                # Check duplicate flag
                if flag_name in parameters:
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command=f"{raw_tool} {subcommand}" if subcommand else raw_tool,
                            error_type="DUPLICATE_PARAMETER",
                            severity="ERROR",
                            message=f"Parameter '--{flag_name}' is specified multiple times in command.",
                            suggestion=f"Remove duplicate '--{flag_name}' parameter."
                        )
                    )
                    step_is_valid = False

                # Check if it is a known boolean switch
                if flag_name in KNOWN_BOOLEAN_FLAGS:
                    parameters[flag_name] = True
                    current_idx += 1
                    continue

                # It expects a value
                if current_idx + 1 < len(tokens):
                    next_tok = tokens[current_idx + 1]

                    # Next token is a value if it does not start with "-" OR if it is a negative number literal
                    if not next_tok.startswith("-") or is_numeric_literal(next_tok):
                        val = next_tok
                        parameters[flag_name] = val
                        self._register_file_parameter(raw_tool, subcommand, flag_name, val, input_files, output_files)
                        current_idx += 2
                    else:
                        # Next token is another flag, meaning this flag is missing its value!
                        errors.append(
                            ValidationError(
                                line_number=line_num,
                                command=f"{raw_tool} {subcommand}" if subcommand else raw_tool,
                                error_type="MISSING_PARAMETER_VALUE",
                                severity="ERROR",
                                message=f"Parameter '--{flag_name}' expects a value, but another flag was encountered.",
                                suggestion=f"Provide a value for '--{flag_name}' (e.g. '--{flag_name} <value>')."
                            )
                        )
                        step_is_valid = False
                        current_idx += 1
                else:
                    # Flag is at the end of line without a value
                    errors.append(
                        ValidationError(
                            line_number=line_num,
                            command=f"{raw_tool} {subcommand}" if subcommand else raw_tool,
                            error_type="MISSING_PARAMETER_VALUE",
                            severity="ERROR",
                            message=f"Parameter '--{flag_name}' is missing an argument value at the end of command.",
                            suggestion=f"Specify an argument value after '--{flag_name}'."
                        )
                    )
                    step_is_valid = False
                    current_idx += 1
            else:
                # Unexpected positional string argument
                errors.append(
                    ValidationError(
                        line_number=line_num,
                        command=f"{raw_tool} {subcommand}" if subcommand else raw_tool,
                        error_type="UNEXPECTED_ARGUMENT",
                        severity="ERROR",
                        message=f"Unexpected positional argument '{token}'. Arguments must be preceded by parameter flags (e.g., --in, --out).",
                        suggestion=f"Prefix '{token}' with an appropriate flag like --in or --out."
                    )
                )
                step_is_valid = False
                current_idx += 1

        stage = TOOL_STAGES.get(raw_tool, "UNKNOWN")

        step = ParsedStep(
            line_number=line_num,
            raw_command=raw_line,
            tool=raw_tool,
            subcommand=subcommand,
            parameters=parameters,
            input_files=input_files,
            output_files=output_files,
            stage=stage,
            is_valid=step_is_valid
        )

        return step, errors

    def _register_file_parameter(
        self,
        tool: str,
        subcommand: Optional[str],
        flag_name: str,
        val: str,
        input_files: List[str],
        output_files: List[str]
    ) -> None:
        """
        Register files with command-specific semantics:
        - For GATK MarkDuplicates: --metrics is an OUTPUT file artifact.
        - For other tools: --out / --output are outputs.
        - Inputs: --in, --input, --ref, --reference.
        """
        # Specific command semantics for GATK MarkDuplicates metrics
        if tool == "GATK" and subcommand == "MARKDUPLICATES" and flag_name == "metrics":
            output_files.append(val)
            return

        if flag_name in ("in", "input", "ref", "reference"):
            input_files.append(val)
        elif flag_name in ("out", "output"):
            output_files.append(val)

    def _suggest_tool_alternative(self, tool_name: str) -> str:
        """Provide helpful suggestion for invalid tool commands."""
        known = list(TOOL_STAGES.keys())
        aliases = {
            "FASTQ": "FASTQC",
            "FASTQ-QC": "FASTQC",
            "TRIM": "TRIMMOMATIC",
            "TRIMMER": "TRIMMOMATIC",
            "BWAMEM": "BWA MEM",
            "SAMTOOL": "SAMTOOLS",
            "SAM": "SAMTOOLS",
            "GATK4": "GATK",
            "PICARD": "GATK MarkDuplicates",
            "BOWTIE": "BWA",
            "BOWTIE2": "BWA MEM",
            "STAR": "BWA MEM",
        }
        if tool_name in aliases:
            return f"Did you mean '{aliases[tool_name]}'? Supported tools are: {', '.join(known)}."
        return f"Supported genomic tools are: {', '.join(known)}."
