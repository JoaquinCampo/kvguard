"""Phase 4 live validation: run static vs controller-driven generation.

Compares five conditions on the same prompts:
  1. baseline — no compression (ratio=0)
  2. static_0.75 — StreamingLLM at ratio=0.75 (no controller)
  3. controlled_0.75 — StreamingLLM at ratio=0.75, controller relaxes to 0
  4. static_0.875 — StreamingLLM at ratio=0.875 (no controller)
  5. controlled_0.875 — StreamingLLM at ratio=0.875, controller relaxes to 0

This measures the gap between offline simulation and actual live behaviour.
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
import xgboost as xgb
from loguru import logger

from kvguard.config import LiveResult
from kvguard.controller import ControllerConfig, RiskController
from kvguard.detectors import parse_gsm8k_answer
from kvguard.live import generate_live
from kvguard.prompts import format_prompt, load_gsm8k

# ---------------------------------------------------------------------------
# Single-prompt runner
# ---------------------------------------------------------------------------


def _run_prompt(
    model: object,
    tokenizer: object,
    device: str,
    prompt_data: dict[str, Any],
    *,
    compression_ratio: float,
    max_new_tokens: int,
    n_sink: int,
    num_fewshot: int,
    seed: int,
    controller: RiskController | None = None,
    predictor: xgb.XGBClassifier | None = None,
) -> LiveResult:
    """Run one prompt through live generation."""
    prompt_text = format_prompt(prompt_data["question"], num_fewshot)
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)  # type: ignore[operator]

    if controller is not None:
        controller.reset()

    result = generate_live(
        model,  # type: ignore[arg-type]
        tokenizer,  # type: ignore[arg-type]
        inputs["input_ids"],
        inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        compression_ratio=compression_ratio,
        n_sink=n_sink,
        controller=controller,
        predictor=predictor,
        device=device,
    )

    # Fill in prompt metadata
    gt = prompt_data["ground_truth"]
    result.prompt_id = prompt_data["prompt_id"]
    result.prompt_text = prompt_text
    result.model = prompt_data.get("model", "")
    result.seed = seed
    result.ground_truth = gt

    # Evaluate correctness
    predicted = parse_gsm8k_answer(result.generated_text)
    result.predicted_answer = predicted
    try:
        result.correct = float(predicted) == float(gt) if predicted else None
    except ValueError:
        result.correct = None

    return result


# ---------------------------------------------------------------------------
# Full validation experiment
# ---------------------------------------------------------------------------


def run_live_validation(
    model_name: str,
    predictor_path: Path,
    *,
    num_prompts: int = 50,
    compression_ratios: list[float] | None = None,
    max_new_tokens: int = 512,
    n_sink: int = 4,
    seed: int = 42,
    num_fewshot: int = 3,
    device: str = "auto",
    output_dir: Path = Path("results/live_validation"),
    tau_low: float = 0.3,
    tau_high: float = 0.7,
    k_escalate: int = 8,
    j_deescalate: int = 5,
) -> dict[str, list[LiveResult]]:
    """Run the Phase 4 live validation experiment.

    For each prompt, runs 1 + 2*len(ratios) conditions:
      - baseline (compression_ratio=0)
      - For each ratio: static + controlled

    Args:
        compression_ratios: List of ratios to test (default: [0.75, 0.875]).

    Returns:
        Dict mapping condition name to list of LiveResult.
    """
    from kvguard.config import ExperimentConfig
    from kvguard.experiment import load_model

    if compression_ratios is None:
        compression_ratios = [0.75, 0.875]

    config = ExperimentConfig(
        model_name=model_name,
        device=device,
    )

    # Load model
    logger.info(f"Loading model: {model_name}")
    model_obj, tokenizer, resolved_device = load_model(config)
    logger.info(f"Device: {resolved_device}")

    # Load predictor
    logger.info(f"Loading predictor: {predictor_path}")
    predictor = xgb.XGBClassifier()
    predictor.load_model(str(predictor_path))

    # Load prompts
    prompts = load_gsm8k(num_prompts=num_prompts, seed=seed)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Build condition names
    conditions: dict[str, list[LiveResult]] = {"baseline": []}
    for ratio in compression_ratios:
        r_str = str(ratio)
        conditions[f"static_{r_str}"] = []
        conditions[f"controlled_{r_str}"] = []

    for i, prompt_data in enumerate(prompts):
        pid = prompt_data["prompt_id"]
        logger.info(f"[{i + 1}/{len(prompts)}] {pid}")

        common: dict[str, Any] = dict(
            model=model_obj,
            tokenizer=tokenizer,
            device=resolved_device,
            prompt_data=prompt_data,
            max_new_tokens=max_new_tokens,
            n_sink=n_sink,
            num_fewshot=num_fewshot,
            seed=seed,
        )

        # 1. Baseline (no compression) — only once per prompt
        t0 = time.time()
        r_base = _run_prompt(compression_ratio=0.0, **common)
        r_base.model = model_name
        logger.info(
            f"  baseline: {'CORRECT' if r_base.correct else 'WRONG'} "
            f"| {r_base.num_tokens_generated} tokens | {time.time() - t0:.1f}s"
        )
        conditions["baseline"].append(r_base)

        # For each compression ratio: static + controlled
        for ratio in compression_ratios:
            r_str = str(ratio)

            # Controller config for this ratio
            ctrl_config = ControllerConfig(
                tau_low=tau_low,
                tau_high=tau_high,
                k_escalate=k_escalate,
                j_deescalate=j_deescalate,
                base_compression_ratio=ratio,
                safe_compression_ratio=0.0,
            )

            # Static (no controller)
            t0 = time.time()
            r_static = _run_prompt(compression_ratio=ratio, **common)
            r_static.model = model_name
            cats = [c for c in r_static.catastrophes if c != "none"]
            logger.info(
                f"  static_{r_str}: {'CORRECT' if r_static.correct else 'WRONG'} "
                f"| {r_static.num_tokens_generated} tokens "
                f"| cats={cats} | {time.time() - t0:.1f}s"
            )
            conditions[f"static_{r_str}"].append(r_static)

            # Controlled (controller active)
            t0 = time.time()
            controller = RiskController(ctrl_config)
            r_ctrl = _run_prompt(
                compression_ratio=ratio,
                controller=controller,
                predictor=predictor,
                **common,
            )
            r_ctrl.model = model_name
            cats = [c for c in r_ctrl.catastrophes if c != "none"]
            safe_at = r_ctrl.safe_trigger_token
            logger.info(
                f"  controlled_{r_str}: {'CORRECT' if r_ctrl.correct else 'WRONG'} "
                f"| {r_ctrl.num_tokens_generated} tokens "
                f"| cats={cats} | safe@{safe_at} | {time.time() - t0:.1f}s"
            )
            conditions[f"controlled_{r_str}"].append(r_ctrl)

        # Clear GPU cache between prompts
        if resolved_device == "cuda":
            torch.cuda.empty_cache()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    for cond_name, results in conditions.items():
        fpath = output_dir / f"{cond_name}_{num_prompts}p.json"
        data: dict[str, Any] = {
            "condition": cond_name,
            "model": model_name,
            "num_prompts": num_prompts,
            "results": [r.model_dump() for r in results],
        }
        fpath.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Saved {cond_name} → {fpath}")

    # Print summary table
    _print_summary(conditions)

    return conditions


def _print_summary(conditions: dict[str, list[LiveResult]]) -> None:
    """Print a comparison table with confidence intervals."""
    print("\n" + "=" * 85)
    print("LIVE VALIDATION RESULTS")
    print("=" * 85)
    header = f"{'Condition':<20} {'Accuracy':>12} {'CBR':>12} {'AvgTokens':>10} {'AvgCache':>10}"
    print(header)
    print("-" * 85)

    for cond_name, results in conditions.items():
        n = len(results)
        if n == 0:
            continue
        correct = sum(1 for r in results if r.correct)
        cbr_cats = ("looping", "non_termination")
        cbr = sum(1 for r in results if any(c in cbr_cats for c in r.catastrophes))
        avg_tok = sum(r.num_tokens_generated for r in results) / n
        avg_cache = 0.0
        for r in results:
            if r.cache_sizes:
                avg_cache += r.cache_sizes[-1]
            else:
                avg_cache += r.num_tokens_generated
        avg_cache /= n

        acc_ci = _wilson_ci(correct, n)
        cbr_ci = _wilson_ci(cbr, n)

        print(
            f"{cond_name:<20} "
            f"{correct}/{n} ({acc_ci})"
            f"  {cbr}/{n} ({cbr_ci})"
            f"  {avg_tok:>8.0f}  {avg_cache:>8.0f}"
        )

    # CBR reduction per ratio
    print()
    for cond_name in conditions:
        if not cond_name.startswith("static_"):
            continue
        ratio_str = cond_name.removeprefix("static_")
        ctrl_name = f"controlled_{ratio_str}"

        static_results = conditions.get(cond_name, [])
        ctrl_results = conditions.get(ctrl_name, [])
        if not static_results or not ctrl_results:
            continue

        cbr_cats = ("looping", "non_termination")
        static_cbr = sum(1 for r in static_results if any(c in cbr_cats for c in r.catastrophes))
        ctrl_cbr = sum(1 for r in ctrl_results if any(c in cbr_cats for c in r.catastrophes))
        n = len(static_results)
        if static_cbr > 0:
            reduction = (static_cbr - ctrl_cbr) / static_cbr * 100
            print(
                f"  ratio={ratio_str}: CBR {static_cbr}/{n} → {ctrl_cbr}/{n} "
                f"({reduction:.1f}% reduction)"
            )
        else:
            print(f"  ratio={ratio_str}: no CBR catastrophes in static condition")

    print("=" * 85)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> str:
    """Wilson score confidence interval, formatted as string."""
    if n == 0:
        return "N/A"
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    lo = max(0.0, centre - spread)
    hi = min(1.0, centre + spread)
    return f"{lo:.0%}-{hi:.0%}"
