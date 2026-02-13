import re
import math


def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output.

    Handles formats:
    - \\boxed{42}
    - "The answer is 42"
    - "#### 42"
    - Plain number at end of string
    """
    text = text.strip()

    # Try \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return _clean_number(boxed[-1])

    # Try "#### <number>"  (GSM8K format)
    hash_match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if hash_match:
        return _clean_number(hash_match.group(1))

    # Try "The answer is <number>"
    answer_match = re.search(r"[Tt]he\s+answer\s+is\s*:?\s*(.+?)[\.\s]*$", text)
    if answer_match:
        return _clean_number(answer_match.group(1))

    # Try last number in string
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return _clean_number(numbers[-1])

    return None


def _clean_number(s: str) -> str:
    """Clean a number string: remove commas, whitespace, dollar signs, percent signs."""
    s = s.strip()
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    return s


def numerical_compare(pred: str | None, gold: str | None, tol: float = 1e-4) -> bool:
    """Compare two number strings with float tolerance."""
    if pred is None or gold is None:
        return False
    try:
        pred_f = float(pred)
        gold_f = float(gold)
        if gold_f == 0:
            return abs(pred_f - gold_f) < tol
        return abs(pred_f - gold_f) / max(abs(gold_f), 1e-10) < tol
    except (ValueError, OverflowError):
        # Fall back to string comparison
        return pred.strip() == gold.strip()
