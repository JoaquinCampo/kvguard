"""Experiment configuration."""

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ExperimentConfig(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    press_name: str = "streaming_llm"  # streaming_llm | observed_attention | snapkv | none
    compression_ratio: float = 0.875  # fraction to REMOVE (0.875 = keep 12.5%)
    max_new_tokens: int = 512
    num_prompts: int = 10
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | mps
    output_dir: Path = Path("results")
    num_fewshot: int = 3

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class TokenSignals(BaseModel):
    entropy: float
    top1_prob: float
    top5_prob: float
    top1_token: str
    rank_of_chosen: int


class RunResult(BaseModel):
    prompt_id: str
    prompt_text: str
    model: str
    press: str
    compression_ratio: float
    max_new_tokens: int
    seed: int
    generated_text: str
    ground_truth: str
    predicted_answer: str | None
    correct: bool | None
    stop_reason: str  # eos | max_tokens
    catastrophes: list[str]
    num_tokens_generated: int
    cache_size_after_prefill: int | None
    signals: list[TokenSignals]
