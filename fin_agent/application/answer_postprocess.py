from __future__ import annotations

from fin_agent.application.guardrails import extract_json_payload, normalize_answer_letters
from fin_agent.domain.models import AnswerFormat


def normalize_answer(fmt: AnswerFormat, text: str) -> str:
    payload = extract_json_payload(text)
    if payload:
        final_answer = payload.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            return normalize_answer_letters(fmt=fmt, text=final_answer)

        option_evaluation = payload.get("option_evaluation")
        if isinstance(option_evaluation, dict):
            letters = []
            for key in sorted(option_evaluation.keys()):
                item = option_evaluation.get(key)
                if isinstance(item, dict) and bool(item.get("verdict")):
                    letters.append(key)
            if letters:
                return normalize_answer_letters(fmt=fmt, text="".join(letters))

    return normalize_answer_letters(fmt=fmt, text=text)

