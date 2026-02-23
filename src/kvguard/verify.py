"""Dataset v1 verification: data paranoia checks before modeling.

Implements the mandatory DATA VERIFICATION CHECKLIST from RALPH:
- Row counts (expected vs actual)
- Label distributions
- Sample inspection
- Feature sanity
- Train/test integrity (signal completeness)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class Check:
    """Result of a single verification check."""

    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    expected: str
    actual: str
    detail: str = ""


@dataclass
class VerificationReport:
    """Full verification report for a sweep results directory."""

    results_dir: str
    num_prompts: int
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status == "PASS" for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    def to_dict(self) -> dict[str, Any]:
        return {
            "results_dir": self.results_dir,
            "num_prompts": self.num_prompts,
            "passed": self.passed,
            "n_fail": self.n_fail,
            "n_warn": self.n_warn,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "expected": c.expected,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


# Expected sweep grid
EXPECTED_RATIOS = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875]
EXPECTED_METHODS = ["streaming_llm", "snapkv", "observed_attention"]
REQUIRED_SIGNAL_FIELDS = [
    "entropy",
    "top1_prob",
    "top5_prob",
    "rank_of_chosen",
    "top20_logprobs",
    "h_alts",
    "avg_logp",
    "delta_h",
    "rep_count",
    "is_thinking_token",
]


def verify_sweep(
    results_dir: Path,
    *,
    num_prompts: int = 50,
) -> VerificationReport:
    """Run full verification on sweep results.

    Args:
        results_dir: Root results directory.
        num_prompts: Expected prompt count per config.

    Returns:
        VerificationReport with all checks.
    """
    report = VerificationReport(
        results_dir=str(results_dir),
        num_prompts=num_prompts,
    )
    pattern = f"*_{num_prompts}p.json"
    files = sorted(results_dir.rglob(pattern))

    # --- Check 1: File count ---
    # Expected: baseline (1) + 5 ratios * 3 methods = 16
    expected_configs = 1 + (len(EXPECTED_RATIOS) - 1) * len(EXPECTED_METHODS)
    report.checks.append(
        Check(
            name="file_count",
            status="PASS" if len(files) == expected_configs else "FAIL",
            expected=str(expected_configs),
            actual=str(len(files)),
            detail=f"Found files: {[str(f.relative_to(results_dir)) for f in files]}",
        )
    )

    if not files:
        report.checks.append(
            Check(
                name="no_files",
                status="FAIL",
                expected="At least 1 result file",
                actual="0 files",
                detail=f"No files matching {pattern} in {results_dir}",
            )
        )
        return report

    # --- Check 2-N: Per-file checks ---
    total_traces = 0
    total_catastrophes = 0
    catastrophe_counts: dict[str, int] = {}
    configs_seen: set[str] = set()
    all_have_rep_count = True
    all_have_onsets = True
    signal_field_issues: list[str] = []

    for fpath in files:
        data = json.loads(fpath.read_text())
        config = data.get("config", {})
        results_list = data.get("results", [])
        press = config.get("press_name", "unknown")
        ratio = config.get("compression_ratio", -1)
        config_key = f"{press}@{ratio}"
        configs_seen.add(config_key)

        # Check prompt count per config
        if len(results_list) != num_prompts:
            report.checks.append(
                Check(
                    name=f"prompt_count_{config_key}",
                    status="FAIL",
                    expected=str(num_prompts),
                    actual=str(len(results_list)),
                    detail=f"File: {fpath.name}",
                )
            )

        for result in results_list:
            total_traces += 1
            cats = result.get("catastrophes", [])
            if cats:
                total_catastrophes += 1
                for c in cats:
                    catastrophe_counts[c] = catastrophe_counts.get(c, 0) + 1

            # Check signal completeness
            signals = result.get("signals", [])
            n_tok = result.get("num_tokens_generated", 0)
            if len(signals) != n_tok:
                signal_field_issues.append(
                    f"{config_key}/{result.get('prompt_id', '?')}: "
                    f"signals={len(signals)} != n_tokens={n_tok}"
                )

            # Check required signal fields on first token
            if signals:
                first_sig = signals[0]
                for field_name in REQUIRED_SIGNAL_FIELDS:
                    if field_name not in first_sig:
                        signal_field_issues.append(
                            f"{config_key}/{result.get('prompt_id', '?')}: "
                            f"missing field '{field_name}'"
                        )
                        if field_name == "rep_count":
                            all_have_rep_count = False

            # Check catastrophe_onsets field exists
            if "catastrophe_onsets" not in result:
                all_have_onsets = False

    # --- Aggregate checks ---

    # Check 3: Total trace count
    expected_traces = expected_configs * num_prompts
    report.checks.append(
        Check(
            name="total_traces",
            status="PASS" if total_traces == expected_traces else "FAIL",
            expected=str(expected_traces),
            actual=str(total_traces),
        )
    )

    # Check 4: Catastrophe distribution (should not be all-zero or all-catastrophe)
    cat_rate = total_catastrophes / total_traces if total_traces > 0 else 0
    if total_catastrophes == 0:
        cat_status = "FAIL"
        cat_detail = "No catastrophes found — all traces benign. Compression may not be working."
    elif cat_rate > 0.95:
        cat_status = "WARN"
        cat_detail = "Almost all traces are catastrophic — check if baseline is correct."
    else:
        cat_status = "PASS"
        cat_detail = f"Types: {catastrophe_counts}"
    report.checks.append(
        Check(
            name="catastrophe_distribution",
            status=cat_status,
            expected="Mix of catastrophe and clean traces",
            actual=f"{total_catastrophes}/{total_traces} ({cat_rate:.1%}) catastrophic",
            detail=cat_detail,
        )
    )

    # Check 5: rep_count field present (T-001 feature)
    report.checks.append(
        Check(
            name="rep_count_present",
            status="PASS" if all_have_rep_count else "FAIL",
            expected="All signals have rep_count",
            actual="present" if all_have_rep_count else "MISSING in some traces",
            detail="rep_count was added in T-001. If missing, traces are from stale data.",
        )
    )

    # Check 6: catastrophe_onsets field present (T-001 feature)
    report.checks.append(
        Check(
            name="catastrophe_onsets_present",
            status="PASS" if all_have_onsets else "FAIL",
            expected="All results have catastrophe_onsets",
            actual="present" if all_have_onsets else "MISSING in some results",
            detail="catastrophe_onsets was added in T-001. If missing, results are stale.",
        )
    )

    # Check 7: Signal field completeness
    if signal_field_issues:
        report.checks.append(
            Check(
                name="signal_completeness",
                status="FAIL",
                expected="All tokens have all required signal fields",
                actual=f"{len(signal_field_issues)} issues found",
                detail="; ".join(signal_field_issues[:10]),
            )
        )
    else:
        report.checks.append(
            Check(
                name="signal_completeness",
                status="PASS",
                expected="All tokens have all required signal fields",
                actual="All fields present",
            )
        )

    # Check 8: Baseline has no catastrophes (sanity)
    baseline_key = "none@0.0"
    if baseline_key in configs_seen:
        baseline_cats = 0
        for fpath in files:
            data = json.loads(fpath.read_text())
            cfg = data.get("config", {})
            if cfg.get("press_name") == "none" and cfg.get("compression_ratio", -1) == 0.0:
                for r in data.get("results", []):
                    if r.get("catastrophes"):
                        baseline_cats += 1
        status = "PASS" if baseline_cats == 0 else "WARN"
        report.checks.append(
            Check(
                name="baseline_sanity",
                status=status,
                expected="Baseline (no compression) should have 0 catastrophes",
                actual=f"{baseline_cats} catastrophes in baseline",
                detail="If baseline has catastrophes, the model itself is unstable.",
            )
        )

    return report


def print_report(report: VerificationReport) -> None:
    """Print verification report in human-readable format."""
    logger.info("=" * 60)
    logger.info(f"DATASET VERIFICATION: {report.results_dir}")
    logger.info(f"Prompt count: {report.num_prompts}")
    logger.info("=" * 60)

    for check in report.checks:
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!"}[check.status]
        logger.info(f"  [{icon}] {check.name}: {check.actual} (expected: {check.expected})")
        if check.detail and check.status != "PASS":
            logger.info(f"      {check.detail}")

    logger.info("-" * 60)
    if report.passed:
        logger.info("RESULT: ALL CHECKS PASSED")
    else:
        logger.info(f"RESULT: {report.n_fail} FAIL, {report.n_warn} WARN")
    logger.info("=" * 60)
