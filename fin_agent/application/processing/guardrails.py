from __future__ import annotations

import json
import re

from fin_agent.domain.models import AnswerFormat


def extract_json_payload(text: str) -> dict | None:
    content = (text or "").strip()
    if not content:
        return None
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def normalize_answer_letters(fmt: AnswerFormat, text: str) -> str:
    letters = re.findall(r"[A-D]", (text or "").upper())
    if fmt in {AnswerFormat.MCQ, AnswerFormat.TF}:
        return letters[0] if letters else "A"
    unique = sorted(set(letters))
    return "".join(unique) if unique else "A"

