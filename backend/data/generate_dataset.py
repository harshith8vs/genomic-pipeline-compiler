"""
Robust Group-Aware Dataset Generator for Genomic Pipeline GLM Training (Version 2.1.0).

Generates a verified, balanced, and group-traceable dataset of valid and faulty genomic pipelines.
Features:
- 12 explicitly defined baseline template architecture groups (lengths 2 to 7).
- Explicit `group_id` assignment for strict group-aware splitting (GroupKFold / StratifiedGroupKFold).
- Prevents mutation lineage and template structure leakage between training and evaluation splits.
- Mutation-based fault injection with guaranteed compiler diagnostic verification.
- Verified absence of errors in all STATIC_VALID samples (failed = 0).
- Verified presence of compiler errors in all STATIC_INVALID samples (failed = 1).
- Complete deduplication of pipeline scripts.
- Rich metadata capture (generator version, seed, group distributions, validation statistics).
"""

import os
import sys
import re
import json
import random
import hashlib
from typing import List, Dict, Any, Tuple, Set
import pandas as pd

# Add app parent to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.compiler import PipelineCompiler
from app.validator import PipelineValidator
from app.features import extract_features, features_to_dict, FEATURE_COLUMNS

GENERATOR_VERSION = "2.1.0"


def get_grouped_templates() -> List[Dict[str, Any]]:
    """
    12 distinct template architecture groups covering diverse stages and commands.
    Every group has a unique group_id for group-aware evaluation.
    """
    return [
        {
            "group_id": "GRP_01_PREPROC_SHORT",
            "desc": "Short Preprocessing: FASTQC -> TRIMMOMATIC",
            "template": [
                "FASTQC --in reads_{id}.fastq --out qc_{id}.html",
                "TRIMMOMATIC --in reads_{id}.fastq --out clean_{id}.fq --minlen {minlen} --slidingwindow 4:20",
            ],
        },
        {
            "group_id": "GRP_02_QC_TRIM_QC",
            "desc": "Dual Quality Control: FASTQC -> TRIMMOMATIC -> FASTQC",
            "template": [
                "FASTQC --in raw_{id}.fq --out qc_{id}.html",
                "TRIMMOMATIC --in raw_{id}.fq --out trim_{id}.fq --minlen {minlen} --slidingwindow 4:15",
                "FASTQC --in trim_{id}.fq --out clean_qc_{id}.html",
            ],
        },
        {
            "group_id": "GRP_03_ALIGN_INDEX_MIN",
            "desc": "Basic Alignment & Indexing: FASTQC -> BWA MEM -> SAMTOOLS SORT -> SAMTOOLS INDEX",
            "template": [
                "FASTQC --in reads_{id}.fq --out qc_{id}.html",
                "BWA MEM --ref reference.fa --in reads_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.sam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
            ],
        },
        {
            "group_id": "GRP_04_INDEX_REF_FIRST",
            "desc": "Reference Indexing First: BWA INDEX -> BWA MEM -> SAMTOOLS SORT -> SAMTOOLS INDEX",
            "template": [
                "BWA INDEX --ref reference.fa",
                "BWA MEM --ref reference.fa --in reads_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.sam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
            ],
        },
        {
            "group_id": "GRP_05_ALIGN_VARS_FASTQC",
            "desc": "Alignment to Variant Discovery with QC: BWA MEM -> SORT -> INDEX -> GATK HaplotypeCaller -> FASTQC",
            "template": [
                "BWA MEM --ref reference.fa --in reads_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.sam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
                "GATK HaplotypeCaller --ref reference.fa --in sorted_{id}.bam --out variants_{id}.vcf",
                "FASTQC --in reads_{id}.fq --out qc_{id}.html",
            ],
        },
        {
            "group_id": "GRP_06_ALIGN_DEDUP_VARS",
            "desc": "Deduplication Pipeline: BWA MEM -> SORT -> GATK MarkDuplicates -> INDEX -> GATK HaplotypeCaller",
            "template": [
                "BWA MEM --ref reference.fa --in reads_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.sam --out sorted_{id}.bam --threads {threads}",
                "GATK MarkDuplicates --in sorted_{id}.bam --out dedup_{id}.bam --metrics dedup_{id}.metrics",
                "SAMTOOLS INDEX --in dedup_{id}.bam",
                "GATK HaplotypeCaller --ref reference.fa --in dedup_{id}.bam --out variants_{id}.vcf",
            ],
        },
        {
            "group_id": "GRP_07_STANDARD_WGS",
            "desc": "Standard 6-Step WGS: FASTQC -> TRIMMOMATIC -> BWA MEM -> SORT -> INDEX -> GATK HaplotypeCaller",
            "template": [
                "FASTQC --in reads_{id}.fastq --out qc_{id}.html",
                "TRIMMOMATIC --in reads_{id}.fastq --out clean_{id}.fq --minlen {minlen} --slidingwindow 4:20",
                "BWA MEM --ref reference.fa --in clean_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.sam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
                "GATK HaplotypeCaller --ref reference.fa --in sorted_{id}.bam --out variants_{id}.vcf",
            ],
        },
        {
            "group_id": "GRP_08_FULL_GERMLINE",
            "desc": "Full 7-Step Germline: FASTQC -> TRIM -> BWA MEM -> SAMTOOLS VIEW -> SORT -> INDEX -> GATK HaplotypeCaller",
            "template": [
                "FASTQC --in reads_{id}.fastq --out qc_{id}.html",
                "TRIMMOMATIC --in reads_{id}.fastq --out clean_{id}.fq --minlen {minlen} --slidingwindow 4:20",
                "BWA MEM --ref reference.fa --in clean_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS VIEW --in align_{id}.sam --out align_{id}.bam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.bam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
                "GATK HaplotypeCaller --ref reference.fa --in sorted_{id}.bam --out variants_{id}.vcf",
            ],
        },
        {
            "group_id": "GRP_09_SOMATIC_MUTECT2",
            "desc": "Somatic Variant Calling: FASTQC -> TRIM -> BWA MEM -> SORT -> INDEX -> GATK Mutect2",
            "template": [
                "FASTQC --in reads_{id}.fastq --out qc_{id}.html",
                "TRIMMOMATIC --in reads_{id}.fastq --out clean_{id}.fq --minlen {minlen} --slidingwindow 4:20",
                "BWA MEM --ref reference.fa --in clean_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.sam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
                "GATK Mutect2 --ref reference.fa --in sorted_{id}.bam --out variants_{id}.vcf",
            ],
        },
        {
            "group_id": "GRP_10_DEDUP_SOMATIC",
            "desc": "Deduplicated Somatic Calling: BWA MEM -> VIEW -> SORT -> MarkDuplicates -> INDEX -> GATK Mutect2",
            "template": [
                "BWA MEM --ref reference.fa --in reads_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS VIEW --in align_{id}.sam --out align_{id}.bam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.bam --out sorted_{id}.bam --threads {threads}",
                "GATK MarkDuplicates --in sorted_{id}.bam --out dedup_{id}.bam --metrics dedup_{id}.metrics",
                "SAMTOOLS INDEX --in dedup_{id}.bam",
                "GATK Mutect2 --ref reference.fa --in dedup_{id}.bam --out somatic_{id}.vcf",
            ],
        },
        {
            "group_id": "GRP_11_VIEW_SORT_INDEX",
            "desc": "Format Conversion Pipeline: BWA MEM -> SAMTOOLS VIEW -> SAMTOOLS SORT -> SAMTOOLS INDEX",
            "template": [
                "BWA MEM --ref reference.fa --in reads_{id}.fq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS VIEW --in align_{id}.sam --out align_{id}.bam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.bam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
            ],
        },
        {
            "group_id": "GRP_12_INDEX_REF_FULL",
            "desc": "Full Indexed Reference WGS: BWA INDEX -> FASTQC -> BWA MEM -> VIEW -> SORT -> INDEX -> GATK HaplotypeCaller",
            "template": [
                "BWA INDEX --ref reference.fa",
                "FASTQC --in reads_{id}.fastq --out qc_{id}.html",
                "BWA MEM --ref reference.fa --in reads_{id}.fastq --out align_{id}.sam --threads {threads}",
                "SAMTOOLS VIEW --in align_{id}.sam --out align_{id}.bam --threads {threads}",
                "SAMTOOLS SORT --in align_{id}.bam --out sorted_{id}.bam --threads {threads}",
                "SAMTOOLS INDEX --in sorted_{id}.bam",
                "GATK HaplotypeCaller --ref reference.fa --in sorted_{id}.bam --out variants_{id}.vcf",
            ],
        },
    ]


def get_declared_inputs(sample_id: str) -> List[str]:
    """Return standard initial source files available for a given sample."""
    return [
        f"reads_{sample_id}.fastq",
        f"reads_{sample_id}.fq",
        f"raw_{sample_id}.fastq",
        f"raw_{sample_id}.fq",
        "reference.fa",
        "hg38.fa",
        f"ref_{sample_id}.fa",
    ]


def apply_mutation(lines: List[str], fault_type: str) -> Tuple[List[str], bool]:
    """
    Apply a deterministic, verifiable mutation to a list of command lines.
    Returns: (mutated_lines, mutation_applied_successfully)
    """
    mutated = list(lines)

    if fault_type == "SYNTAX_UNKNOWN_TOOL":
        idx = random.randint(0, len(mutated) - 1)
        for old, new in [("BWA MEM", "BOWTIE2"), ("FASTQC", "FASTQC_PRO"), ("SAMTOOLS", "SAMTOOL_CLI")]:
            if old in mutated[idx]:
                mutated[idx] = mutated[idx].replace(old, new, 1)
                return mutated, True
        tokens = mutated[idx].split()
        tokens[0] = "UNKNOWN_TOOL"
        mutated[idx] = " ".join(tokens)
        return mutated, True

    if fault_type == "SYNTAX_UNKNOWN_SUBCOMMAND":
        sub_indices = [i for i, l in enumerate(mutated) if any(l.startswith(t) for t in ("BWA ", "SAMTOOLS ", "GATK "))]
        if sub_indices:
            idx = random.choice(sub_indices)
            parts = mutated[idx].split(maxsplit=2)
            parts[1] = "INVALID_SUBCOMMAND"
            mutated[idx] = " ".join(parts)
            return mutated, True
        return mutated, False

    if fault_type == "MISSING_REQUIRED_PARAM":
        target_lines = [i for i, l in enumerate(mutated) if any(flag in l for flag in ("--ref", "--out", "--in"))]
        if target_lines:
            idx = random.choice(target_lines)
            tokens = mutated[idx].split()
            new_tokens = []
            skip = False
            removed = False
            for t in tokens:
                if skip:
                    skip = False
                    continue
                if not removed and t in ("--ref", "--out", "--in"):
                    skip = True
                    removed = True
                    continue
                new_tokens.append(t)
            if removed:
                mutated[idx] = " ".join(new_tokens)
                return mutated, True
        return mutated, False

    if fault_type == "MISSING_PARAM_VALUE":
        idx = random.randint(0, len(mutated) - 1)
        mutated[idx] = mutated[idx] + " --threads"
        return mutated, True

    if fault_type == "INVALID_PARAM_FORMAT":
        target_lines = [i for i, l in enumerate(mutated) if "--threads" in l or "--slidingwindow" in l]
        if target_lines:
            idx = random.choice(target_lines)
            if "--threads" in mutated[idx]:
                mutated[idx] = mutated[idx].replace("--threads 4", "--threads -4").replace("--threads 8", "--threads -8").replace("--threads 2", "--threads -2").replace("--threads 16", "--threads -16")
                return mutated, True
            if "--slidingwindow" in mutated[idx]:
                mutated[idx] = mutated[idx].replace("4:20", "invalid_window").replace("4:15", "invalid_window")
                return mutated, True
        return mutated, False

    if fault_type == "INVALID_EXTENSION":
        for i, l in enumerate(mutated):
            if ".bam" in l:
                mutated[i] = l.replace(".bam", ".txt", 1)
                return mutated, True
            if ".sam" in l:
                mutated[i] = l.replace(".sam", ".pdf", 1)
                return mutated, True
        return mutated, False

    if fault_type == "INVERT_COMMAND_SEQUENCE":
        if len(mutated) >= 2:
            mutated[0], mutated[-1] = mutated[-1], mutated[0]
            return mutated, True
        return mutated, False

    if fault_type == "UNINDEXED_BAM_GATK":
        has_gatk = any("GATK" in l for l in mutated)
        has_index = any("SAMTOOLS INDEX" in l for l in mutated)
        if has_gatk and has_index:
            mutated = [l for l in mutated if "SAMTOOLS INDEX" not in l]
            return mutated, True
        return mutated, False

    if fault_type == "UNRESOLVED_INTERMEDIATE_FILE":
        for i, l in enumerate(mutated):
            if i > 0 and "--in" in l:
                mutated[i] = re.sub(r"--in\s+\S+", "--in non_existent_intermediate_file.bam", l)
                return mutated, True
        return mutated, False

    if fault_type == "CONFLICT_OVERWRITE":
        for i, l in enumerate(mutated):
            if "SAMTOOLS SORT" in l:
                mutated[i] = re.sub(r"--out\s+\S+", "--out sample_conflict.bam", l)
                mutated[i] = re.sub(r"--in\s+\S+", "--in sample_conflict.bam", mutated[i])
                return mutated, True
        return mutated, False

    return mutated, False


def generate_pipeline_dataset(
    num_samples: int = 240,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate verified, group-traceable dataset balanced across 12 template groups.
    Each template group produces an equal share of STATIC_VALID and STATIC_INVALID examples.

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    random.seed(seed)
    compiler = PipelineCompiler()
    validator = PipelineValidator()

    group_defs = get_grouped_templates()
    num_groups = len(group_defs)
    samples_per_group = num_samples // num_groups
    half_per_group = samples_per_group // 2

    records: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()

    fault_types = [
        "SYNTAX_UNKNOWN_TOOL",
        "SYNTAX_UNKNOWN_SUBCOMMAND",
        "MISSING_REQUIRED_PARAM",
        "MISSING_PARAM_VALUE",
        "INVALID_PARAM_FORMAT",
        "INVALID_EXTENSION",
        "INVERT_COMMAND_SEQUENCE",
        "UNINDEXED_BAM_GATK",
        "UNRESOLVED_INTERMEDIATE_FILE",
        "CONFLICT_OVERWRITE",
    ]

    for g_idx, g_def in enumerate(group_defs):
        group_id = g_def["group_id"]
        template = g_def["template"]

        # A. Generate STATIC_VALID examples for this group
        valid_count = 0
        attempts = 0
        while valid_count < half_per_group and attempts < half_per_group * 20:
            attempts += 1
            sample_id = f"g{g_idx+1}_v{valid_count+1}"
            threads = random.choice([2, 4, 8, 16])
            minlen = random.choice([30, 36, 40, 50])
            lines = [line.format(id=sample_id, threads=threads, minlen=minlen) for line in template]
            script = "\n".join(lines)

            script_hash = hashlib.md5(script.strip().encode()).hexdigest()
            if script_hash in seen_hashes:
                continue

            declared = get_declared_inputs(sample_id)
            steps, syntax_errs = compiler.parse(script)
            report = validator.validate(steps, syntax_errs, declared_inputs=declared)

            # Strict check: 0 errors
            if not report.is_valid or report.total_errors > 0:
                continue

            feat = extract_features(report, steps)
            feat_dict = features_to_dict(feat)
            feat_dict["group_id"] = group_id
            feat_dict["pipeline_text"] = script
            feat_dict["failed"] = 0
            feat_dict["label"] = "STATIC_VALID"
            feat_dict["label_desc"] = f"Valid baseline ({g_def['desc']})"

            seen_hashes.add(script_hash)
            records.append(feat_dict)
            valid_count += 1

        # B. Generate STATIC_INVALID examples for this group
        invalid_count = 0
        attempts = 0
        while invalid_count < half_per_group and attempts < half_per_group * 30:
            attempts += 1
            sample_id = f"g{g_idx+1}_f{invalid_count+1}"
            threads = random.choice([2, 4, 8])
            minlen = random.choice([30, 36, 40])
            lines = [line.format(id=sample_id, threads=threads, minlen=minlen) for line in template]

            f_type = random.choice(fault_types)
            mutated_lines, applied = apply_mutation(lines, f_type)
            if not applied:
                continue

            script = "\n".join(mutated_lines)
            script_hash = hashlib.md5(script.strip().encode()).hexdigest()
            if script_hash in seen_hashes:
                continue

            declared = get_declared_inputs(sample_id)
            steps, syntax_errs = compiler.parse(script)
            report = validator.validate(steps, syntax_errs, declared_inputs=declared)

            # Strict check: >0 errors
            if report.is_valid or report.total_errors == 0:
                continue

            feat = extract_features(report, steps)
            feat_dict = features_to_dict(feat)
            feat_dict["group_id"] = group_id
            feat_dict["pipeline_text"] = script
            feat_dict["failed"] = 1
            feat_dict["label"] = "STATIC_INVALID"
            feat_dict["label_desc"] = f"Mutated failure ({f_type}) in {group_id}"

            seen_hashes.add(script_hash)
            records.append(feat_dict)
            invalid_count += 1

    df = pd.DataFrame(records)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "random_seed": seed,
        "total_samples": len(df),
        "total_groups": df["group_id"].nunique(),
        "groups": sorted(list(df["group_id"].unique())),
        "target_column": "failed",
        "label_column": "label",
        "label_mapping": {"STATIC_VALID": 0, "STATIC_INVALID": 1},
        "class_distribution": {
            "STATIC_VALID": int((df["failed"] == 0).sum()),
            "STATIC_INVALID": int((df["failed"] == 1).sum()),
        },
        "validation_statistics": {
            "valid_samples_with_zero_errors": int((df[df["failed"] == 0]["total_error_count"] == 0).sum()),
            "invalid_samples_with_errors": int((df[df["failed"] == 1]["total_error_count"] > 0).sum()),
            "duplicate_scripts_detected": 0,
        },
    }

    return df, metadata


def main():
    data_dir = os.path.dirname(__file__)
    json_path = os.path.join(data_dir, "train_dataset.json")
    csv_path = os.path.join(data_dir, "train_dataset.csv")

    print(f"Generating group-aware genomic pipeline dataset (v{GENERATOR_VERSION})...")
    df, metadata = generate_pipeline_dataset(num_samples=240, seed=42)

    payload = {
        "metadata": metadata,
        "features": FEATURE_COLUMNS,
        "records": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    df.to_csv(csv_path, index=False)

    print("Dataset generation complete:")
    print(f"  Total samples: {metadata['total_samples']} across {metadata['total_groups']} groups")
    print(f"  STATIC_VALID:   {metadata['class_distribution']['STATIC_VALID']}")
    print(f"  STATIC_INVALID: {metadata['class_distribution']['STATIC_INVALID']}")
    print(f"  Saved to: {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
