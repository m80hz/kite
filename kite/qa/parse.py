import re
import ast
from typing import Tuple, Optional, List

def parse_mcq_label(text: str) -> Tuple[Optional[str], float]:
    """Parse MCQ choice like 'Answer: (C) mis-grasp' or 'C' from model output; returns (label_letter, confidence).
    Very lightweight heuristic.    """
    if text is None:
        return None, 0.0
    t = text.strip()
    # Look for (A)/(B)/... pattern
    m = re.search(r"\(([A-E])\)", t)
    if m:
        return m.group(1), 0.6
    # Look for 'Answer: A' or just 'A'
    m = re.search(r"\b([A-E])\b", t)
    if m:
        return m.group(1), 0.5
    return None, 0.0


def extract_options_from_question(question_text: str) -> List[str]:
    """Extract a list of options from a question string like:
    "Your answer should choose one of the following options: ['A', 'B', 'C']".
    Returns [] if none found.
    """
    if not question_text:
        return []
    # Try to find a Python-like list literal inside brackets
    m = re.search(r"\[(.*?)\]", question_text, re.DOTALL)
    if not m:
        return []
    lit = '[' + m.group(1) + ']'
    try:
        val = ast.literal_eval(lit)
        if isinstance(val, list):
            return [str(v).strip() for v in val]
    except Exception:
        pass
    # Fallback: split by quotes/comma
    parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", lit)
    opts = []
    for a,b in parts:
        s = a or b
        if s:
            opts.append(s.strip())
    return opts


def parse_option_from_answer(answer_text: str, options: List[str]) -> Tuple[Optional[str], float]:
    """Return the matched option (exact or substring) from model answer.
    If answer contains a letter (A-E) and options provided, map A->options[0], etc.
    """
    if not answer_text:
        return None, 0.0
    t = answer_text.strip()
    # Letter mapping first
    letter, conf = parse_mcq_label(t)
    if letter and options:
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return options[idx], max(conf, 0.6)
    # Try exact match ignoring case and trailing punct
    t_clean = re.sub(r"[\s\.]$", "", t.lower())
    for opt in options:
        o_clean = re.sub(r"[\s\.]$", "", str(opt).lower())
        if t_clean == o_clean:
            return opt, 0.7
    # Substring containment (model may wrap with explanation)
    for opt in options:
        if str(opt).lower() in t.lower():
            return opt, 0.6
    # Nothing matched
    return None, 0.0

