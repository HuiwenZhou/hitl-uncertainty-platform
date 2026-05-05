import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from agents.one_step_agent import ask_llm
from uncertainty.internal_confidence import compute_internal_confidence
from uncertainty.consistency import compute_consistency
from hitl.trigger import should_trigger_hitl

from datetime import datetime
from evaluation.metrics import compute_correctness

load_dotenv()

THRESHOLDS = {
    "avg_logprob": -1.0,  # avg_logprob < -1.0 → Trigger HITL
    "min_logprob": -3.0,
    "avg_margin": 0.5,
    "agreement": 0.7,
}

MODEL = "gpt-4o-mini"
N_SAMPLES = 5


def run_one_question(row):
    question_id = row["question_id"]
    question = row["question"]
    reference_answer = row["reference_answer"]

    start_time = time.time()

    first_run = ask_llm(
        question=question,
        model=MODEL,
        temperature=0.0,
        logprobs=True,
    )

    confidence = compute_internal_confidence(
        first_run["token_logprobs"],
        first_run["token_top_logprobs"],
    )

    sampled_answers = []

    for _ in range(N_SAMPLES):
        sample = ask_llm(
            question=question,
            model=MODEL,
            temperature=0.7,
            logprobs=False,
        )
        sampled_answers.append(sample["answer"])

    consistency = compute_consistency(
    answers=sampled_answers,
    question=question,
    use_semantic=True,
    )

    hitl_decision = should_trigger_hitl(
        avg_logprob=confidence["avg_logprob"],
        min_logprob=confidence["min_logprob"],
        avg_margin=confidence["avg_margin"],
        agreement=consistency["agreement"],
        thresholds=THRESHOLDS,
    )

    correctness = compute_correctness(
    answer=first_run["answer"],
    reference_answer=reference_answer,
    )

    latency = time.time() - start_time

    return {
    "question_id": question_id,
    "model": MODEL,

    # ===== question =====
    "question": question,

    # ===== core output =====
    "answer_summary": first_run["answer"],  # Truncate to the first 100 characters.

    # ===== uncertainty =====
    "avg_logprob": confidence["avg_logprob"],
    "min_logprob": confidence["min_logprob"],
    "avg_margin": confidence["avg_margin"],
    "agreement": consistency["agreement"],

    # ===== consistency =====
    "num_unique_answers": consistency["num_unique_answers"],
    "semantic_agreement": consistency["semantic_agreement"],
    "surface_agreement": consistency["surface_agreement"],
    "num_semantic_unique_answers": consistency["num_semantic_unique_answers"],
    "uncertainty_score": -confidence["avg_logprob"],

    # ===== HITL =====
    "triggered_hitl": hitl_decision["triggered"],
    "trigger_reasons": hitl_decision["reasons"],

    # ===== cost =====
    "total_tokens": first_run["total_tokens"],
    "latency_seconds": latency,
    }


def main():
    os.makedirs("results", exist_ok=True)

    questions = pd.read_csv("data/questions_complex.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/results_{timestamp}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(questions.iterrows(), total=len(questions)):
            result = run_one_question(row)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()