from __future__ import annotations

import json
import logging
import re

from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage, OpenAiCompatibleChatClient

logger = logging.getLogger(__name__)


class LlmHeadingDetector:
    """使用 LLM 对疑似标题行做批量补判。"""

    def __init__(self, client: OpenAiCompatibleChatClient, batch_size: int = 40) -> None:
        self._client = client
        self._batch_size = max(1, batch_size)

    def classify_candidates(self, doc_id: str, lines: list[str]) -> set[str]:
        candidates = [line for line in _deduplicate_preserve_order(lines) if is_potential_heading_candidate(line)]
        if not candidates:
            return set()

        matched: set[str] = set()
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            try:
                matched.update(self._classify_batch(doc_id=doc_id, lines=batch))
            except Exception as exc:
                logger.warning("LLM heading 判定失败：doc_id=%s error=%s", doc_id, repr(exc))
        return matched

    def _classify_batch(self, doc_id: str, lines: list[str]) -> set[str]:
        numbered_lines = "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(lines))
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "你是金融文档结构识别器。"
                    "请判断给定行是否应该被视为文档标题、章节名、小节名、条款名、表格标题或结构性小标题。"
                    "必须保留原文，不得改写。只输出 JSON。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"文档ID：{doc_id}\n\n"
                    "请从下列文本行中挑出应视为 heading 的行，输出 JSON：\n"
                    '{\n  "headings": ["原文行1", "原文行2"]\n}\n\n'
                    "判断标准：\n"
                    "- heading 通常是标题、章节名、条款起始、目录项、表格名、附录名、定义项。\n"
                    "- 简短概括性短语更可能是 heading。\n"
                    "- 完整叙述句、长说明句、普通正文一般不是 heading。\n"
                    "- 必须返回完全一致的原文行，不能返回编号。\n\n"
                    f"待判断文本行：\n{numbered_lines}"
                ),
            ),
        ]
        response = self._client.chat(messages)
        payload = _extract_json_payload(response.content)
        headings = payload.get("headings") if isinstance(payload, dict) else None
        if not isinstance(headings, list):
            return set()
        source = set(lines)
        return {str(item).strip() for item in headings if str(item).strip() in source}


def is_potential_heading_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) < 2 or len(stripped) > 60:
        return False
    if stripped.endswith(("。", "；", ";", "！", "？", ":", "：")):
        return False
    if stripped.count("，") >= 2 or stripped.count(",") >= 2:
        return False
    if re.search(r"[。！？!?；;]", stripped):
        return False
    if re.fullmatch(r"[\W_]+", stripped):
        return False
    return True


def _deduplicate_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return ordered


def _extract_json_payload(text: str) -> dict | None:
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

