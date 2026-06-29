from __future__ import annotations

from fin_agent.application.retrieval import DOMAIN_PROMPT_HINTS, build_domain_reasoning_instruction
from fin_agent.domain.models import AnswerFormat, Question
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage


def build_messages(q: Question, context: str) -> list[ChatMessage]:
    system = (
        "你是一名严谨的金融审计师。"
        "你必须严格基于证据作答，不得使用常识或外部知识。"
        "如果上下文提供的证据不足以推导某选项，则该选项视为错误。"
        "请逐项审阅 A/B/C/D，但最终只能输出 JSON。"
    )
    user = format_user_prompt(q=q, context=context)
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def format_user_prompt(q: Question, context: str) -> str:
    options_text = "\n".join(f"{key}. {value}" for key, value in sorted(q.options.items()))
    format_hint = {
        AnswerFormat.MCQ: "最终答案必须是一个大写字母。",
        AnswerFormat.TF: "最终答案必须是一个大写字母（A 或 B）。",
        AnswerFormat.MULTI: "最终答案必须是按字母排序且去重后的多个大写字母。",
    }[q.answer_format]
    domain_instruction = build_domain_reasoning_instruction(q.domain)
    return (
        f"领域：{q.domain}\n"
        f"题目：{q.question}\n"
        f"选项：\n{options_text}\n\n"
        f"证据链：\n{context}\n\n"
        f"领域提示：{DOMAIN_PROMPT_HINTS.get(q.domain, '')}\n"
        f"领域专项要求：{domain_instruction}\n"
        f"{format_hint}\n"
        "请输出如下 JSON：\n"
        '{\n'
        '  "option_evaluation": {\n'
        '    "A": {"verdict": true, "evidence": "DocID + Title", "reason": "一句话"},\n'
        '    "B": {"verdict": false, "evidence": "...", "reason": "一句话"}\n'
        "  },\n"
        '  "final_answer": "AC"\n'
        "}\n"
        "不要输出 JSON 以外的内容。"
    )

