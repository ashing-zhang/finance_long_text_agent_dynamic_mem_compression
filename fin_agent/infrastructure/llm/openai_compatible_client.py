from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from fin_agent.domain.models import LlmConfig, TokenUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """OpenAI-compatible chat message."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """OpenAI-compatible chat response."""

    content: str
    usage: TokenUsage
    raw: dict


class OpenAiCompatibleChatClient:
    """OpenAI-compatible Chat Completions 客户端（stdlib 实现）。"""

    def __init__(self, config: LlmConfig) -> None:
        """初始化客户端。"""
        self._config = config

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """调用 /chat/completions 并返回文本与 token usage。"""
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        api_key = os.getenv(self._config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"未设置 API Key：请设置环境变量 {self._config.api_key_env}，或在配置中修改 api_key_env。"
            )

        payload = {
            "model": self._config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return self._parse_chat_response(raw)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "LLM 调用失败（attempt=%s/%s）：%s",
                    attempt + 1,
                    self._config.max_retries + 1,
                    repr(exc),
                )
                if attempt >= self._config.max_retries:
                    break
                time.sleep(min(2**attempt, 4))

        raise RuntimeError(f"LLM 调用失败：{last_error!r}") from last_error

    def _parse_chat_response(self, raw: dict) -> ChatResponse:
        """解析 OpenAI-compatible 响应。"""
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage_raw = raw.get("usage") or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
        completion_tokens = int(usage_raw.get("completion_tokens") or 0)
        total_tokens = int(usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens))
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return ChatResponse(content=content or "", usage=usage, raw=raw)
