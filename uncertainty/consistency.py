from collections import Counter
import os
import re
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def normalize_answer(text):
    text = str(text).strip().lower()
    text = text.replace(".", "")
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_core_answer(text):
    """
    Rule-based core answer extractor.
    Useful for simple factual questions.
    """
    text_lower = str(text).lower()

    if "stands for" in text_lower:
        core = text_lower.split("stands for")[-1]
        core = core.split(".")[0]
        return normalize_answer(core)

    if "capital of france is" in text_lower:
        core = text_lower.split("capital of france is")[-1]
        core = core.split(".")[0]
        return normalize_answer(core)

    return normalize_answer(text)


def compute_surface_consistency(answers):
    if not answers:
        return {
            "surface_agreement": None,
            "majority_answer": None,
            "answer_distribution": {},
            "core_answers": [],
            "num_unique_answers": 0,
        }
    """
    Surface-level consistency based on normalized/core string matching.
    """
    core_answers = [extract_core_answer(a) for a in answers]
    counts = Counter(core_answers)

    most_common_answer, count = counts.most_common(1)[0]
    agreement = count / len(answers)

    return {
        "surface_agreement": agreement,
        "majority_answer": most_common_answer,
        "answer_distribution": dict(counts),
        "core_answers": core_answers,
        "num_unique_answers": len(counts),
    }


def judge_semantic_equivalence(question, answer_a, answer_b, model="gpt-4o-mini"):
    """
    Use an evaluator LLM to judge whether two answers are semantically consistent.
    """

    prompt = f"""
You are evaluating answer consistency for an uncertainty estimation experiment.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Decide whether Answer A and Answer B provide substantially the same final recommendation,
decision, or core meaning.

They do NOT need to use the same wording.
They can include different details.
They are semantically consistent if their main conclusion is the same.

Return only valid JSON in this exact format:
{{"equivalent": true}}
or
{{"equivalent": false}}
"""

    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a strict but fair semantic consistency evaluator."},
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content.strip()

    try:
        result = json.loads(text)
        return bool(result.get("equivalent", False))
    except Exception:
        return False


def compute_semantic_consistency(
    question,
    answers,
    evaluator_model="gpt-4o-mini",
):
    """
    Semantic consistency:
    compare each sampled answer with the first sampled answer.
    agreement = proportion of answers semantically equivalent to the first answer.
    """

    if not answers:
        return {
            "semantic_agreement": None,
            "num_semantic_unique_answers": None,
            "semantic_equivalence_flags": [],
        }

    base_answer = answers[0]
    flags = []

    for ans in answers:
        equivalent = judge_semantic_equivalence(
            question=question,
            answer_a=base_answer,
            answer_b=ans,
            model=evaluator_model,
        )
        flags.append(equivalent)

    equivalent_count = sum(flags)
    semantic_agreement = equivalent_count / len(answers)

    return {
        "semantic_agreement": semantic_agreement,
        "num_semantic_unique_answers": len(answers) - equivalent_count + 1,
        "semantic_equivalence_flags": flags,
    }


def compute_consistency(
    answers,
    question=None,
    use_semantic=True,
    evaluator_model="gpt-4o-mini",
):
    """
    Unified consistency function.

    It returns both:
    1. surface_agreement: string/core-answer matching
    2. semantic_agreement: LLM-judged semantic consistency

    In run_experiment.py, you can use semantic_agreement as the main agreement score.
    """

    surface_result = compute_surface_consistency(answers)

    if use_semantic and question is not None:
        semantic_result = compute_semantic_consistency(
            question=question,
            answers=answers,
            evaluator_model=evaluator_model,
        )
    else:
        semantic_result = {
            "semantic_agreement": None,
            "num_semantic_unique_answers": None,
            "semantic_equivalence_flags": [],
        }

    final_agreement = (
        semantic_result["semantic_agreement"]
        if semantic_result["semantic_agreement"] is not None
        else surface_result["surface_agreement"]
    )

    return {
        "agreement": final_agreement,
        "surface_agreement": surface_result["surface_agreement"],
        "semantic_agreement": semantic_result["semantic_agreement"],
        "majority_answer": surface_result["majority_answer"],
        "num_unique_answers": surface_result["num_unique_answers"],
        "num_semantic_unique_answers": semantic_result["num_semantic_unique_answers"],
        "semantic_equivalence_flags": semantic_result["semantic_equivalence_flags"],
    }