"""Live token-by-token generation with controller-driven KV-cache eviction.

Implements a manual autoregressive loop where the RiskController actively
modifies cache eviction behavior during generation. kvpress only compresses
at prefill (hooks are no-ops during generation), so we implement StreamingLLM
eviction ourselves by directly manipulating DynamicCache tensors.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import xgboost as xgb
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvguard.config import LiveResult, TokenSignals
from kvguard.controller import Mode, RiskController
from kvguard.detectors import (
    detect_all,
    detect_catastrophe_onsets,
    parse_gsm8k_answer,
)
from kvguard.features import (
    BASE_FEATURE_NAMES,
    ROLLING_COL_INDICES,
    ROLLING_WINDOW,
    flatten_token,
)
from kvguard.signals import compute_repetition_counts, extract_signals

# ---------------------------------------------------------------------------
# StreamingLLM cache eviction
# ---------------------------------------------------------------------------


def evict_streaming_llm_bulk(
    cache: DynamicCache,
    n_sink: int,
    window_size: int,
) -> None:
    """Bulk-evict tokens from the middle of the cache (post-prefill).

    Keeps the first ``n_sink`` tokens and the last ``window_size`` tokens,
    removing everything in between. Operates in-place on all layers.

    Each layer stores keys/values with shape ``[batch, heads, seq_len, head_dim]``.
    """
    for layer_idx in range(len(cache)):
        layer = cache.layers[layer_idx]
        assert layer.keys is not None and layer.values is not None
        keys = layer.keys
        values = layer.values
        seq_len = keys.shape[2]

        if seq_len <= n_sink + window_size:
            return  # nothing to evict

        keep_indices = torch.cat(
            [
                torch.arange(n_sink, device=keys.device),
                torch.arange(seq_len - window_size, seq_len, device=keys.device),
            ]
        )

        layer.keys = keys[:, :, keep_indices, :]
        layer.values = values[:, :, keep_indices, :]


def evict_streaming_llm_step(cache: DynamicCache, n_sink: int) -> None:
    """Remove the oldest non-sink token from the cache (one token at position n_sink).

    Called after each generation step to maintain a fixed cache size.
    After the model appends a new KV entry, the cache has one more token
    than the target size. This removes exactly one token at position ``n_sink``.
    """
    for layer_idx in range(len(cache)):
        layer = cache.layers[layer_idx]
        assert layer.keys is not None and layer.values is not None
        keys = layer.keys
        values = layer.values
        seq_len = keys.shape[2]

        if seq_len <= n_sink + 1:
            return  # nothing to evict

        layer.keys = torch.cat(
            [keys[:, :, :n_sink, :], keys[:, :, n_sink + 1 :, :]],
            dim=2,
        )
        layer.values = torch.cat(
            [values[:, :, :n_sink, :], values[:, :, n_sink + 1 :, :]],
            dim=2,
        )


def _cache_seq_len(cache: DynamicCache) -> int:
    """Return sequence length from the first layer of the cache."""
    if len(cache) == 0:
        return 0
    keys = cache.layers[0].keys
    assert keys is not None
    return int(keys.shape[2])


# ---------------------------------------------------------------------------
# Online feature builder
# ---------------------------------------------------------------------------


@dataclass
class OnlineFeatureBuilder:
    """Streaming computation of the 40-dim feature vector.

    Maintains a rolling window of base features and produces a complete
    feature vector one token at a time, matching the offline
    :func:`~kvguard.features.add_rolling_features` output.
    """

    window: int = ROLLING_WINDOW
    max_new_tokens: int = 512
    compression_ratio: float = 0.875

    # Internal state
    _step: int = field(default=0, init=False)
    _base_history: list[np.ndarray] = field(default_factory=list, init=False)
    _token_ids: list[int] = field(default_factory=list, init=False)
    _window_seen: dict[tuple[int, ...], int] = field(default_factory=dict, init=False)

    def push(self, signals: TokenSignals, token_id: int) -> np.ndarray:
        """Add one token's signals and return the full 40-dim feature vector."""
        sig_dict = signals.model_dump()

        # Online rep_count
        self._token_ids.append(token_id)
        sig_dict["rep_count"] = self._compute_rep_count()

        base = flatten_token(sig_dict)
        self._base_history.append(base)

        rolling = self._compute_rolling()

        denom = max(self.max_new_tokens, 1)
        token_position = np.float32(self._step / denom)

        self._step += 1

        return np.concatenate(
            [
                base,
                rolling,
                np.array([token_position, np.float32(self.compression_ratio)]),
            ]
        )

    def update_compression_ratio(self, ratio: float) -> None:
        """Update compression ratio when controller changes mode."""
        self.compression_ratio = ratio

    def reset(self) -> None:
        """Reset for a new generation."""
        self._step = 0
        self._base_history.clear()
        self._token_ids.clear()
        self._window_seen.clear()

    def _compute_rep_count(self) -> int:
        """Online repetition count matching compute_repetition_counts."""
        rep_window = 20
        ids = self._token_ids
        n = len(ids)
        if n < rep_window:
            return 0
        window = tuple(ids[n - rep_window : n])
        prev = self._window_seen.get(window, 0)
        self._window_seen[window] = prev + 1
        return prev

    def _compute_rolling(self) -> np.ndarray:
        """Compute rolling mean/std for ROLLING_COLS + rep_count rolling sum."""
        history = self._base_history
        w = min(len(history), self.window)
        recent = np.stack(history[-w:])  # (w, N_BASE)

        result: list[float] = []
        for col_idx in ROLLING_COL_INDICES:
            col = recent[:, col_idx]
            result.append(float(col.mean()))
            if w > 1:
                mean_val = col.mean()
                var = float(np.mean(col**2) - mean_val**2)
                result.append(float(np.sqrt(max(var, 0.0))))
            else:
                result.append(0.0)

        # rep_count rolling sum
        rep_idx = BASE_FEATURE_NAMES.index("rep_count")
        rep_col = recent[:, rep_idx]
        result.append(float(rep_col.sum()))

        return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Live generation loop
# ---------------------------------------------------------------------------


def generate_live(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    compression_ratio: float = 0.875,
    n_sink: int = 4,
    controller: RiskController | None = None,
    predictor: xgb.XGBClassifier | None = None,
    device: str = "cuda",
) -> LiveResult:
    """Run live generation with optional controller-driven cache eviction.

    Args:
        model: The language model.
        tokenizer: Tokenizer for the model.
        input_ids: Tokenized prompt, shape ``(1, prompt_len)``.
        attention_mask: Attention mask, shape ``(1, prompt_len)``.
        max_new_tokens: Maximum tokens to generate.
        compression_ratio: Fraction of cache to remove (0.875 = keep 12.5%).
        n_sink: Number of sink tokens to always keep.
        controller: If provided, controller drives eviction decisions.
        predictor: XGBoost predictor for hazard probability.
        device: Device string.

    Returns:
        LiveResult with generation output and controller trace.
    """
    prompt_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id  # type: ignore[attr-defined]
    controlled = controller is not None and predictor is not None

    # Compute window size from compression ratio
    tokens_to_keep = max(int(prompt_len * (1 - compression_ratio)), n_sink + 1)
    window_size = tokens_to_keep - n_sink

    # Feature builder for controller
    builder: OnlineFeatureBuilder | None = None
    if controlled:
        builder = OnlineFeatureBuilder(
            max_new_tokens=max_new_tokens,
            compression_ratio=compression_ratio,
        )

    # Tracking
    generated_ids: list[int] = []
    all_signals: list[TokenSignals] = []
    mode_history: list[int] = []
    hazard_probs: list[float] = []
    eviction_history: list[bool] = []
    cache_sizes: list[int] = []
    safe_trigger_token: int | None = None
    prev_entropy: float | None = None

    t_start = time.time()

    with torch.no_grad():
        # --- Prefill ---
        cache = DynamicCache()
        out = model(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        logits = out.logits  # (1, prompt_len, vocab_size)
        cache = out.past_key_values

        # Apply bulk eviction (StreamingLLM: keep sinks + window)
        if compression_ratio > 0:
            evict_streaming_llm_bulk(cache, n_sink, window_size)

        # Track the logical position (for RoPE)
        logical_position = prompt_len

        # --- Generation loop ---
        for step in range(max_new_tokens):
            # Greedy sample from last logits
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            next_token_id = int(next_token_logits.argmax().item())
            generated_ids.append(next_token_id)

            # Extract signals
            sig = extract_signals(next_token_logits, next_token_id, tokenizer, prev_entropy)
            prev_entropy = sig.entropy
            all_signals.append(sig)

            # Controller decision
            should_evict = compression_ratio > 0
            if (
                controlled
                and controller is not None
                and predictor is not None
                and builder is not None
            ):
                features = builder.push(sig, next_token_id)
                h_prob = float(predictor.predict_proba(features.reshape(1, -1))[0, 1])
                hazard_probs.append(h_prob)

                action = controller.step_with_risk(h_prob)
                mode_history.append(int(action.mode))

                if action.mode >= Mode.SAFE:
                    should_evict = False
                    if safe_trigger_token is None:
                        safe_trigger_token = step
                        logger.info(f"  SAFE mode at token {step} (hazard_prob={h_prob:.3f})")
                    # Update feature builder with new compression ratio
                    builder.update_compression_ratio(action.compression_ratio)

            eviction_history.append(should_evict)

            # Check EOS
            if next_token_id == eos_id:
                break

            # Prepare next forward pass
            next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            cache_position = torch.tensor([logical_position], dtype=torch.long, device=device)
            logical_position += 1

            out = model(  # type: ignore[operator]
                input_ids=next_input,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            logits = out.logits
            cache = out.past_key_values

            # Apply eviction if in NORMAL/ALERT mode
            if should_evict:
                evict_streaming_llm_step(cache, n_sink)

            cache_sizes.append(_cache_seq_len(cache))

    generation_time = time.time() - t_start

    # Post-generation: compute rep_count over full sequence
    rep_counts = compute_repetition_counts(generated_ids)
    for sig_obj, rc in zip(all_signals, rep_counts):
        sig_obj.rep_count = rc

    # Decode and detect
    generated_text = tokenizer.decode(  # type: ignore[attr-defined]
        generated_ids, skip_special_tokens=True
    )
    hit_max = len(generated_ids) >= max_new_tokens
    hit_eos = len(generated_ids) > 0 and generated_ids[-1] == eos_id
    stop_reason = "max_tokens" if (hit_max and not hit_eos) else "eos"

    catastrophes = detect_all(generated_text, generated_ids, stop_reason, "")
    catastrophe_onsets = detect_catastrophe_onsets(generated_ids, stop_reason, catastrophes)

    predicted = parse_gsm8k_answer(generated_text)

    return LiveResult(
        prompt_id="",  # filled by caller
        prompt_text="",  # filled by caller
        model="",  # filled by caller
        compression_ratio=compression_ratio,
        max_new_tokens=max_new_tokens,
        seed=0,  # filled by caller
        generated_text=generated_text,
        ground_truth="",  # filled by caller
        predicted_answer=predicted,
        correct=None,  # filled by caller
        stop_reason=stop_reason,
        catastrophes=catastrophes,
        catastrophe_onsets=catastrophe_onsets,
        num_tokens_generated=len(generated_ids),
        signals=all_signals,
        controlled=controlled,
        mode_history=mode_history,
        hazard_probs=hazard_probs,
        eviction_history=eviction_history,
        cache_sizes=cache_sizes,
        safe_trigger_token=safe_trigger_token,
        generation_time_seconds=generation_time,
    )
