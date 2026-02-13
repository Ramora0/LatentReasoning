from math_verify import parse, verify


def check_answer(pred_text: str, gold_text: str) -> bool:
    """Check if a predicted answer matches the gold answer using math-verify.

    Handles LaTeX, plain expressions, sets, intervals, and numeric comparisons
    via SymPy-based symbolic verification.
    """
    try:
        gold_parsed = parse(gold_text)
        pred_parsed = parse(pred_text)
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False
