def should_trigger_hitl(
    avg_logprob,
    min_logprob,
    avg_margin,
    agreement,
    thresholds
):
    reasons = []

    if avg_logprob is not None and avg_logprob < thresholds["avg_logprob"]:
        reasons.append("low_avg_logprob")

    if min_logprob is not None and min_logprob < thresholds["min_logprob"]:
        reasons.append("low_min_logprob")

    if avg_margin is not None and avg_margin < thresholds["avg_margin"]:
        reasons.append("low_margin")

    if agreement is not None and agreement < thresholds["agreement"]:
        reasons.append("low_consistency")

    return {
        "triggered": len(reasons) > 0,
        "reasons": reasons,
    }