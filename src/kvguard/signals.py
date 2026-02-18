"""Per-token signal extraction from logits — HALT 25-dim feature vector."""

import torch
import torch.nn.functional as F

from kvguard.config import TokenSignals

# Thinking tokens: transitional words that carry disproportionate reasoning
# content. From "Demystifying Reasoning Dynamics" (2506.02867) — suppressing
# these 1-5% of tokens degrades reasoning; suppressing random tokens doesn't.
THINKING_TOKENS = frozenset({
    "so", "wait", "therefore", "hmm", "let", "first", "but",
    "thus", "hence", "now", "next", "then", "however", "since",
    "because", "given", "note", "recall", "consider", "if",
    "ok", "okay", "well", "right", "actually", "alternatively",
    "step", "finally", "suppose", "assume", "check", "verify",
})


def extract_signals(
    logits: torch.Tensor,
    chosen_token_id: int,
    tokenizer: object,
    prev_entropy: float | None = None,
) -> TokenSignals:
    """Extract the HALT-inspired feature vector from logits at a single step.

    Features extracted (25 HALT + extensions):
    - top20_logprobs: top-20 log-probabilities (20 features)
    - entropy (H_overall): total entropy
    - h_alts: entropy excluding top-1 (competitor disagreement)
    - avg_logp: mean log-probability (distribution sharpness)
    - rank_of_chosen: rank of chosen token (RankProxy equivalent)
    - delta_h: H(t) - H(t-1) (DeltaH_dec — confidence shocks)
    - is_thinking_token: vocabulary-based thinking-token flag

    Args:
        logits: shape (vocab_size,) — raw logits for one position
        chosen_token_id: the token actually selected
        tokenizer: for decoding tokens to strings
        prev_entropy: entropy at t-1, for computing delta_h
    """
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # H_overall: total entropy
    entropy = -(probs * log_probs).sum().item()

    # Top-20 log-probs and indices
    k = min(20, log_probs.shape[-1])
    top_lp, top_idx = torch.topk(log_probs, k=k)
    top20_logprobs = top_lp.tolist()

    # Top-k probabilities (reuse top-20 indices)
    top_probs = probs[top_idx]
    top1_prob = top_probs[0].item()
    top5_prob = top_probs[:5].sum().item()
    top1_token = tokenizer.decode([top_idx[0].item()])  # type: ignore[arg-type]

    # H_alts: entropy of the distribution excluding top-1
    top1_p = probs[top_idx[0]].item()
    remaining_mass = 1.0 - top1_p
    if remaining_mass > 1e-10:
        alt_probs = probs.clone()
        alt_probs[top_idx[0]] = 0
        alt_probs = alt_probs / remaining_mass
        alt_log = torch.log(alt_probs.clamp(min=1e-10))
        h_alts = -(alt_probs * alt_log).sum().item()
    else:
        h_alts = 0.0

    # AvgLogP: mean log-probability (distribution sharpness)
    avg_logp = log_probs.mean().item()

    # Rank of chosen token
    sorted_indices = torch.argsort(probs, descending=True)
    rank_mask = sorted_indices == chosen_token_id
    rank_of_chosen = rank_mask.nonzero(as_tuple=True)[0].item()

    # DeltaH_dec: temporal entropy change
    delta_h = round(entropy - prev_entropy, 4) if prev_entropy is not None else None

    # Thinking-token detection
    token_text = tokenizer.decode([chosen_token_id]).strip().lower()  # type: ignore[arg-type]
    is_thinking = token_text in THINKING_TOKENS

    return TokenSignals(
        entropy=round(entropy, 4),
        top1_prob=round(top1_prob, 4),
        top5_prob=round(top5_prob, 4),
        top1_token=top1_token,
        rank_of_chosen=int(rank_of_chosen),
        top20_logprobs=[round(x, 3) for x in top20_logprobs],
        h_alts=round(h_alts, 4),
        avg_logp=round(avg_logp, 4),
        delta_h=delta_h,
        is_thinking_token=is_thinking,
    )
