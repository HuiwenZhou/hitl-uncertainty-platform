import pandas as pd


def compute_correctness(answer, reference_answer):
    """
    If there is no reference answer, return None.
    Otherwise, check whether the reference answer appears in the model answer.
    """
    if pd.isna(reference_answer) or str(reference_answer).strip() == "":
        return None

    answer_norm = str(answer).lower().strip()
    ref_norm = str(reference_answer).lower().strip()

    return ref_norm in answer_norm