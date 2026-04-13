"""Prompt augmentation utilities for evaluation.

We append light instruction suffixes for generic (non-RoboFAC fine-tuned) VLMs
so answers are formatted consistently across descriptive question types.

Design decisions:
- Keep augmentation separate from adapter transport logic.
- Avoid modifying original annotation text; augmentation only applied right before querying.
- Multi-choice style questions (identification/locating) are left untouched.
"""
from __future__ import annotations
from typing import Dict

# Mapping of question type -> suffix to append (preceded by a space)
QUESTION_AUGMENT_RULES: Dict[str, str] = {
    "High-level correction": " (Answer in 2-3 detailed sentences.)",
    "Low-level correction": " (Specify direction relative to the robot arm and approximate movement magnitude.)",
    "Failure explanation": " (Answer in 2-3 explanatory sentences.)",
    "Task planning": " (Respond as numbered steps: '1. ... 2. ... 3. ...')",
    "Task identification": " (Answer with a brief task phrase.)",
    "Failure detection": " (Your answer should choose one of the following options: ['Yes.', 'No.'])",
}

CHOICE_TYPES = {"Failure identification", "Failure locating"}


def augment_question(model_name: str, question_type: str, base_question: str) -> str:
    """Return augmented question text if model is not RoboFAC-specific.

    Parameters
    ----------
    model_name : str
        Name or path; if contains 'robofac' (case-insensitive) we skip augmentation.
    question_type : str
        Annotation key describing the question category.
    base_question : str
        Original question text from annotations.
    """
    if not base_question:
        return base_question

    lower_name = model_name.lower() if model_name else ""
    if "robofac" in lower_name:
        return base_question  # fine-tuned model already aligned
    if question_type in CHOICE_TYPES:
        return base_question  # keep raw text for MCQ / matching
    suffix = QUESTION_AUGMENT_RULES.get(question_type)
    if not suffix:
        return base_question  # unknown type -> conservative
    return base_question + suffix

__all__ = ["augment_question", "QUESTION_AUGMENT_RULES", "CHOICE_TYPES"]
