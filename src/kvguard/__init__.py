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


def main() -> None:
    app()
