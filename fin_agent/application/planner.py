from __future__ import annotations

import json
import logging
import re
from fin_agent.compat import dataclass

from fin_agent.application.guardrails import extract_json_payload
from fin_agent.application.retrieval import QueryFeatures, expand_query_by_domain, extract_query_features
from fin_agent.application.tracing import zero_token_usage
from fin_agent.domain.models import Question, TokenUsage
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage, OpenAiCompatibleChatClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    global_query: str
    option_queries: dict[str, str]
    features: QueryFeatures
    option_features: dict[str, QueryFeatures]


def build_retrieval_plan(
    llm: OpenAiCompatibleChatClient,
    query_feature_cache: dict[str, tuple[QueryFeatures, dict[str, QueryFeatures]]],
    q: Question,
) -> tuple[RetrievalPlan, TokenUsage]:
    global_query = build_global_query(q)
    option_queries = build_option_queries(q)
    global_features, option_features, usage = extract_plan_features_with_llm(
        llm=llm,
        query_feature_cache=query_feature_cache,
        domain=q.domain,
        global_query=global_query,
        option_queries=option_queries,
    )
    plan = RetrievalPlan(
        global_query=global_query,
        option_queries=option_queries,
        features=global_features,
        option_features=option_features,
    )
    return plan, usage


def build_global_query(q: Question) -> str:
    parts = [q.question.strip(), "选项："]
    for key in sorted(q.options.keys()):
        parts.append(f"{key}. {q.options[key].strip()}")
    parts.append(expand_query_by_domain(domain=q.domain, text=q.question))
    return "\n".join(parts).strip()


def build_option_queries(q: Question) -> dict[str, str]:
    option_queries: dict[str, str] = {}
    for key in sorted(q.options.keys()):
        option_text = q.options[key]
        option_query = "\n".join(
            [
                q.question.strip(),
                f"选项 {key}: {option_text.strip()}",
                expand_query_by_domain(domain=q.domain, text=option_text),
            ]
        ).strip()
        option_queries[key] = option_query
    return option_queries


def extract_plan_features_with_llm(
    llm: OpenAiCompatibleChatClient,
    query_feature_cache: dict[str, tuple[QueryFeatures, dict[str, QueryFeatures]]],
    domain: str,
    global_query: str,
    option_queries: dict[str, str],
) -> tuple[QueryFeatures, dict[str, QueryFeatures], TokenUsage]:
    cache_key = json.dumps(
        {
            "domain": domain,
            "global_query": global_query,
            "option_queries": {k: option_queries[k] for k in sorted(option_queries.keys())},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = query_feature_cache.get(cache_key)
    if cached is not None:
        return cached[0], dict(cached[1]), zero_token_usage()

    messages = [
        ChatMessage(
            role="system",
            content=(
                "你是信息抽取器，负责从查询文本中提取检索特征。"
                "你必须严格从输入文本中抽取“原文子串”，不得改写、不得引入外部知识。"
                "只输出 JSON，不要输出任何解释。"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"领域：{domain}\n\n"
                "请对以下查询文本抽取特征，输出 JSON，格式如下：\n"
                "{\n"
                '  "global": {"years": [], "numbers": [], "clauses": [], "keywords": []},\n'
                '  "options": {\n'
                '    "A": {"years": [], "numbers": [], "clauses": [], "keywords": []}\n'
                "  }\n"
                "}\n\n"
                "抽取要求：\n"
                "- years：仅保留 4 位年份（如 2024、2025），必须在原文出现。\n"
                "- numbers：保留带单位的数值原文子串（如 6.5亿元、75%、30天、1个月），必须在原文出现。\n"
                "- clauses：保留条款编号原文子串（如 第四十七条、第3章），必须在原文出现。\n"
                "- keywords：保留 6~18 个能区分检索的关键词/术语，必须在原文出现；优先保留金融专有名词、主体、评级、产品名、条款名；避免整句。\n"
                "- 对英文/缩写请保持原样（例如 AAA）。\n"
                "- 去重后输出。\n\n"
                f"global_query:\n{global_query}\n\n"
                "option_queries:\n"
                + "\n".join(f"{k}:\n{option_queries[k]}" for k in sorted(option_queries.keys()))
            ),
        ),
    ]
    try:
        resp = llm.chat(messages)
        payload = extract_json_payload(resp.content)
        global_features = parse_features_payload(payload.get("global") if isinstance(payload, dict) else None)
        option_features: dict[str, QueryFeatures] = {}
        options_payload = payload.get("options") if isinstance(payload, dict) else None
        for key in sorted(option_queries.keys()):
            item = options_payload.get(key) if isinstance(options_payload, dict) else None
            option_features[key] = parse_features_payload(item)
        query_feature_cache[cache_key] = (global_features, dict(option_features))
        return global_features, option_features, resp.usage
    except Exception as exc:
        logger.warning("LLM 特征抽取失败，回退正则：%s", repr(exc))
        global_features = extract_query_features(global_query)
        option_features = {k: extract_query_features(v) for k, v in option_queries.items()}
        query_feature_cache[cache_key] = (global_features, dict(option_features))
        return global_features, option_features, zero_token_usage()


def parse_features_payload(payload: object) -> QueryFeatures:
    if not isinstance(payload, dict):
        return QueryFeatures(years=(), numbers=(), clauses=(), keywords=())

    years_raw = payload.get("years")
    numbers_raw = payload.get("numbers")
    clauses_raw = payload.get("clauses")
    keywords_raw = payload.get("keywords")

    years = tuple(sorted({str(x).strip() for x in (years_raw if isinstance(years_raw, list) else []) if str(x).strip()}))
    years = tuple(y for y in years if re.fullmatch(r"(?:19|20)\d{2}", y))

    numbers = tuple(sorted({str(x).strip() for x in (numbers_raw if isinstance(numbers_raw, list) else []) if str(x).strip()}))
    clauses = tuple(sorted({str(x).strip() for x in (clauses_raw if isinstance(clauses_raw, list) else []) if str(x).strip()}))
    keywords = tuple(sorted({str(x).strip() for x in (keywords_raw if isinstance(keywords_raw, list) else []) if str(x).strip()}))
    return QueryFeatures(years=years, numbers=numbers, clauses=clauses, keywords=keywords)

