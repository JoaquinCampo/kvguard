"""kvguard — Catastrophe-aware KV-cache compression controller."""

from pathlib import Path

import typer
from loguru import logger

from kvguard.config import ExperimentConfig

app = typer.Typer(help="KV-cache compression experiment runner.")


@app.command()
def run(
    model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model name"),
    press: str = typer.Option(
        "streaming_llm", help="Press: streaming_llm | observed_attention | snapkv | none"
    ),
    compression_ratio: float = typer.Option(
        0.875, help="Fraction of KV cache to REMOVE (0.875 = keep 12.5%)"
    ),
    num_prompts: int = typer.Option(10, help="Number of GSM8K prompts to evaluate"),
    max_new_tokens: int = typer.Option(512, help="Max tokens to generate per prompt"),
    seed: int = typer.Option(42, help="Random seed for prompt selection"),
    device: str = typer.Option("auto", help="Device: auto | cpu | cuda | mps"),
    output_dir: Path = typer.Option(Path("results"), help="Directory for output JSON"),
    num_fewshot: int = typer.Option(3, help="Number of few-shot examples"),
) -> None:
    """Run a KV-cache compression experiment on GSM8K."""
    from kvguard.experiment import run_experiment, save_results

    config = ExperimentConfig(
        model_name=model,
        press_name=press,
        compression_ratio=compression_ratio,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
        seed=seed,
        device=device,
        output_dir=output_dir,
        num_fewshot=num_fewshot,
    )

    logger.info(f"Config: {config.model_dump()}")
    results = run_experiment(config)
    path = save_results(results, config)
    logger.info(f"Done. Results at {path}")


@app.command()
def sweep(
    num_prompts: int = typer.Option(50, help="Prompts per configuration"),
    seed: int = typer.Option(42, help="Random seed for prompt selection"),
    output_dir: Path = typer.Option(Path("results"), help="Directory for output JSON"),
    max_new_tokens: int = typer.Option(512, help="Max tokens per prompt"),
    model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model"),
    skip_existing: bool = typer.Option(True, help="Skip configs with existing results"),
    prompt_timeout: float = typer.Option(
        300.0, help="Per-prompt timeout in seconds (kills MPS hangs)"
    ),
) -> None:
    """Run the full compression sweep: baseline + 5 ratios x 3 methods."""
    from kvguard.experiment import run_sweep as _run_sweep

    logger.info(f"Starting sweep: {num_prompts} prompts, output={output_dir}")
    _run_sweep(
        num_prompts=num_prompts,
        seed=seed,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
        model_name=model,
        skip_existing=skip_existing,
        prompt_timeout_seconds=prompt_timeout,
    )
    logger.info("Sweep complete.")


@app.command()
def analyze(
    output_dir: Path = typer.Option(Path("results"), help="Results directory"),
    num_prompts: int = typer.Option(50, help="Filter to results with this prompt count"),
) -> None:
    """Run full analysis on sweep results."""
    from kvguard.analyze import full_analysis

    full_analysis(output_dir, num_prompts)


@app.command()
def verify(
    output_dir: Path = typer.Option(Path("results"), help="Results directory"),
    num_prompts: int = typer.Option(50, help="Expected prompts per configuration"),
) -> None:
    """Verify dataset integrity before training (mandatory data paranoia)."""
    from kvguard.verify import print_report, verify_sweep

    report = verify_sweep(output_dir, num_prompts=num_prompts)
    print_report(report)
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def train(
    results_dir: Path = typer.Option(Path("results"), help="Sweep results directory"),
    num_prompts: int = typer.Option(50, help="Filter to results with this prompt count"),
    horizon: int = typer.Option(32, help="Hazard prediction horizon H (tokens)"),
    nt_onset_frac: float = typer.Option(0.75, help="Non-termination proxy onset fraction"),
    val_fraction: float = typer.Option(0.2, help="Fraction of traces for validation"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option(Path("models"), help="Directory for model + metrics"),
    run_cv: bool = typer.Option(True, help="Run leave-one-compressor-out CV"),
    model_filter: str = typer.Option("", help="Only use traces from this model (empty=all)"),
    press_exclude: str = typer.Option(
        "", help="Comma-separated compressor names to exclude (e.g. observed_attention)"
    ),
    exclude_features: str = typer.Option(
        "", help="Comma-separated feature names to drop before training"
    ),
) -> None:
    """Train hazard predictor on sweep results."""
    from kvguard.train import run_training

    mf = model_filter or None
    pe = [p.strip() for p in press_exclude.split(",") if p.strip()] or None
    ef = [f.strip() for f in exclude_features.split(",") if f.strip()] or None
    logger.info(f"Training: results={results_dir}, H={horizon}, val_frac={val_fraction}")
    if mf:
        logger.info(f"Model filter: {mf}")
    if pe:
        logger.info(f"Excluding compressors: {pe}")
    if ef:
        logger.info(f"Excluding features: {ef}")
    result = run_training(
        results_dir,
        num_prompts=num_prompts,
        horizon=horizon,
        nt_onset_frac=nt_onset_frac,
        val_fraction=val_fraction,
        random_state=seed,
        run_cv=run_cv,
        output_dir=output_dir,
        model_filter=mf,
        press_exclude=pe,
        exclude_features=ef,
    )
    logger.info(f"Train metrics: {result.train_metrics.to_dict()}")
    logger.info(f"Val metrics:   {result.val_metrics.to_dict()}")
    if result.cv_result:
        logger.info(f"CV mean AUROC: {result.cv_result.mean_auroc:.4f}")
        logger.info(f"CV mean F1:    {result.cv_result.mean_f1:.4f}")
    logger.info(f"Model saved to {output_dir}")


@app.command("eval-controller")
def eval_controller(
    results_dir: Path = typer.Option(Path("results"), help="Sweep results directory"),
    model_path: Path = typer.Option(
        Path("models/hazard_predictor.json"), help="Trained hazard predictor"
    ),
    num_prompts: int = typer.Option(50, help="Filter to results with this prompt count"),
    tau_low: float = typer.Option(0.3, help="NORMAL → ALERT threshold"),
    tau_high: float = typer.Option(0.7, help="ALERT → SAFE threshold"),
    safe_ratio: float = typer.Option(
        0.0, help="Compression ratio in SAFE mode (0.0 = no compression)"
    ),
    k_escalate: int = typer.Option(8, help="Consecutive high-risk tokens to escalate"),
    j_deescalate: int = typer.Option(5, help="Consecutive low-risk tokens to de-escalate"),
    output_path: Path = typer.Option(Path("results/controller_eval.json"), help="Output JSON path"),
    model_filter: str = typer.Option("", help="Only use traces from this model (empty=all)"),
) -> None:
    """Evaluate controller via offline simulation on existing sweep traces."""
    import json

    import xgboost as xgb

    from kvguard.controller import ControllerConfig
    from kvguard.evaluate_controller import (
        eval_result_to_dict,
        format_eval_table,
    )
    from kvguard.evaluate_controller import (
        evaluate_controller as _eval_ctrl,
    )

    mf = model_filter or None
    logger.info(f"Loading predictor from {model_path}")
    predictor = xgb.XGBClassifier()
    predictor.load_model(str(model_path))

    config = ControllerConfig(
        tau_low=tau_low,
        tau_high=tau_high,
        base_compression_ratio=0.875,
        safe_compression_ratio=safe_ratio,
        k_escalate=k_escalate,
        j_deescalate=j_deescalate,
    )

    # Load holdout split info if available
    holdout_prompt_ids: set[str] | None = None
    split_info_path = model_path.parent / "split_info.json"
    if split_info_path.exists():
        split_info = json.loads(split_info_path.read_text())
        holdout_prompt_ids = set(split_info["val_prompt_ids"])
        logger.info(f"Evaluating on {len(holdout_prompt_ids)} held-out prompts")

    logger.info(f"Running controller evaluation on {results_dir}")
    if mf:
        logger.info(f"Model filter: {mf}")
    result = _eval_ctrl(
        results_dir,
        predictor,
        num_prompts=num_prompts,
        controller_config=config,
        holdout_prompt_ids=holdout_prompt_ids,
        model_filter=mf,
    )

    print(format_eval_table(result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_result_to_dict(result), indent=2))
    logger.info(f"Results saved to {output_path}")


@app.command("live-validate")
def live_validate(
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model name"),
    predictor_path: Path = typer.Option(
        Path("models/hazard_predictor.json"), help="Trained hazard predictor"
    ),
    num_prompts: int = typer.Option(50, help="Number of GSM8K prompts"),
    ratios: str = typer.Option("0.75,0.875", help="Comma-separated compression ratios to test"),
    max_new_tokens: int = typer.Option(512, help="Max tokens per prompt"),
    n_sink: int = typer.Option(4, help="StreamingLLM sink tokens"),
    seed: int = typer.Option(42, help="Random seed"),
    device: str = typer.Option("auto", help="Device: auto | cpu | cuda | mps"),
    output_dir: Path = typer.Option(Path("results/live_validation"), help="Output directory"),
    tau_low: float = typer.Option(0.3, help="NORMAL → ALERT threshold"),
    tau_high: float = typer.Option(0.7, help="ALERT → SAFE threshold"),
    k_escalate: int = typer.Option(8, help="Consecutive tokens to escalate"),
    j_deescalate: int = typer.Option(5, help="Consecutive tokens to de-escalate"),
) -> None:
    """Run Phase 4 live validation: static vs controller-driven StreamingLLM."""
    from kvguard.live_experiment import run_live_validation

    ratio_list = [float(r.strip()) for r in ratios.split(",")]
    logger.info(f"Live validation: {num_prompts} prompts, ratios={ratio_list}")
    run_live_validation(
        model_name=model,
        predictor_path=predictor_path,
        num_prompts=num_prompts,
        compression_ratios=ratio_list,
        max_new_tokens=max_new_tokens,
        n_sink=n_sink,
        seed=seed,
        device=device,
        output_dir=output_dir,
        tau_low=tau_low,
        tau_high=tau_high,
        k_escalate=k_escalate,
        j_deescalate=j_deescalate,
    )


@app.command("analyze-live")
def analyze_live(
    output_dir: Path = typer.Option(
        Path("results/live_validation"), help="Live validation results directory"
    ),
    offline_eval: Path = typer.Option(
        Path("results/controller_eval.json"), help="Offline controller eval JSON (for comparison)"
    ),
    save_report: Path = typer.Option(
        Path("results/live_validation/analysis_report.json"), help="Save analysis JSON"
    ),
) -> None:
    """Analyze Phase 4 live validation results."""
    import json

    from kvguard.analyze_live import analyze_live_validation

    offline_path = offline_eval if offline_eval.exists() else None
    if offline_path:
        logger.info(f"Using offline eval: {offline_path}")
    else:
        logger.info("No offline eval found — skipping simulation gap comparison")

    report = analyze_live_validation(output_dir, offline_eval_path=offline_path)

    save_report.parent.mkdir(parents=True, exist_ok=True)
    save_report.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report saved to {save_report}")


def main() -> None:
    app()
