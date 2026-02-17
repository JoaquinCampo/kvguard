"""Per-token signal extraction from logits."""

import torch
import torch.nn.functional as F

from kvguard.config import TokenSignals


def extract_signals(
    logits: torch.Tensor,
    chosen_token_id: int,
    tokenizer: object,
) -> TokenSignals:
    """Extract quality signals from the logit distribution at a single step.

    Args:
        logits: shape (vocab_size,) — raw logits for one position
        chosen_token_id: the token actually selected (greedy argmax)
        tokenizer: for decoding the top-1 token to a string
    """
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum().item()

    # Top-k probabilities
    top5_probs, top5_indices = torch.topk(probs, k=5)
    top1_prob = top5_probs[0].item()
    top5_prob = top5_probs.sum().item()
    top1_token = tokenizer.decode([top5_indices[0].item()])  # type: ignore[arg-type]

    # Rank of chosen token
    sorted_indices = torch.argsort(probs, descending=True)
    rank_of_chosen = (sorted_indices == chosen_token_id).nonzero(as_tuple=True)[0].item()

    return TokenSignals(
        entropy=round(entropy, 4),
        top1_prob=round(top1_prob, 4),
        top5_prob=round(top5_prob, 4),
        top1_token=top1_token,
        rank_of_chosen=int(rank_of_chosen),
    )
