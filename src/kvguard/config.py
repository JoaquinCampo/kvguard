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
    prompt_timeout_seconds: float = 300.0  # kill generation if a single prompt exceeds this

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
    # Core logit features (from experiment 001)
    entropy: float  # H_overall: total entropy of output distribution
    top1_prob: float  # probability of most likely token
    top5_prob: float  # cumulative probability of top-5 tokens
    top1_token: str  # decoded top-1 token
    rank_of_chosen: int  # rank of the actually chosen token

    # HALT features (new — backward-compatible defaults)
    top20_logprobs: list[float] = []  # log-probs of top-20 tokens
    h_alts: float = 0.0  # entropy excluding top-1 (competitor disagreement)
    avg_logp: float = 0.0  # mean log-prob (distribution sharpness)

    # Temporal feature (computed from consecutive tokens)
    delta_h: float | None = None  # H(t) - H(t-1), None for first token

    # Repetition feature (computed from token sequence, not logits)
    rep_count: int = 0  # times the window ending here appeared before in the sequence

    # Token classification
    is_thinking_token: bool = False


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
    catastrophe_onsets: dict[str, int] = {}  # catastrophe_type -> token position of onset
    signals: list[TokenSignals]
