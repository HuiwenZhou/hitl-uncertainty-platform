import math


def compute_internal_confidence(token_logprobs, token_top_logprobs):
    if not token_logprobs:
        return {
            "avg_logprob": None,
            "min_logprob": None,
            "avg_margin": None,
        }

    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    min_logprob = min(token_logprobs)

    margins = []

    for top_items in token_top_logprobs:
        if top_items and len(top_items) >= 2:
            top1 = top_items[0].logprob
            top2 = top_items[1].logprob
            margins.append(top1 - top2)

    avg_margin = sum(margins) / len(margins) if margins else None

    return {
        "avg_logprob": avg_logprob,
        "min_logprob": min_logprob,
        "avg_margin": avg_margin,
    }