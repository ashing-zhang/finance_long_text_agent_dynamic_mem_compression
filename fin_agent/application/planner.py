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

FEATURE_PROMPT_VERSION = "v6_key_value_pairs"


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
            "prompt_version": FEATURE_PROMPT_VERSION,
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
                "你是金融检索特征生成器，负责把问题与每个选项转成“更容易命中原文证据”的检索特征。"
                "你可以利用推理与常识做适度改写与扩展（例如同义表达、字段别名、常见表述模板），以提升召回率。"
                "不要引入与问题无关的实体或事实，不要猜测文档里不存在的具体数值。"
                "重要：options 中每个选项都必须产出非空 features。"
                "只输出 JSON，不要输出任何解释。"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"领域：{domain}\n\n"
                "请对以下查询文本抽取特征，输出 JSON，格式如下：\n"
                "{\n"
                '  "global": {"years": [], "numbers": [], "clauses": [], "features": [], "feature_pairs": []},\n'
                '  "options": {\n'
                '    "A": {"years": [], "numbers": [], "clauses": [], "features": [], "feature_pairs": []}\n'
                "  }\n"
                "}\n\n"
                "抽取要求：\n"
                "- years：仅保留 4 位年份（如 2024、2025）；如果文本中没有年份可为空。\n"
                "- numbers：优先保留带单位的数值短串（如 6.5亿元、75%、30天、1个月）；如果没有可为空；不要编造新数值。\n"
                "- clauses：条款编号或章节定位（如 第四十七条、第3章、第一条定义及解释）；可以生成常见定位短语来辅助检索。\n"
                "- features：每个对象输出 2~5 个“检索键（key）”，必须是字段名/术语（名词），不要输出整句。\n"
                "- feature_pairs：每个对象额外输出 0~3 个 (key, value) 对；当选项里存在明确的比较对象、实体名、机构名、评级结果、金额/比例/期限等 value 时，优先抽取。\n"
                "  - 例如“发行人名称为广东省广晟控股集团有限公司”应输出：key=发行人名称，value=广东省广晟控股集团有限公司。\n"
                "  - 例如“主体信用评级均为 AAA”应输出：key=主体信用评级，value=AAA。\n"
                "  - 例如“明确指定国信证券股份有限公司为受托管理人”应输出：key=受托管理人，value=国信证券股份有限公司。\n"
                "  - 若只有 key 没有明确 value，则 feature_pairs 可为空。\n"
                "  - feature_pairs 中的 key 和 value 都尽量贴近选项原文；value 可以保留实体全称或精确短串。\n"
                "  - feature_pairs 使用 JSON 对象数组，格式为 {\"key\": \"...\", \"value\": \"...\"}。\n"
                "  - 不要把多个词用“/”合并在同一条里；一条 features 只能是一个 key。\n"
                "  - 每条建议不超过 12 字。\n"
                "  - 金融合同常见 key 示例：\n"
                "    - 发行人 / 发行主体 / 发行人名称\n"
                "    - 发行规模 / 发行金额 / 发行总额 / 上限\n"
                "    - 主体信用评级 / 主体信用等级 / 信用评级结果\n"
                "    - 受托管理人 / 债券受托管理人 / 受托管理机构\n"
                "- global.features：放“全局能区分检索的领域要素与主题词”。\n"
                "- options.<key>.features：放“只对该选项成立/不成立最关键的字段线索”。\n"
                "- options.<key>.feature_pairs：放“该选项最关键的 key-value 精确约束”。\n"
                "- 你必须为每个选项 A/B/C/D 输出非空 features（至少 3 条）。\n"
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
        global_features = parse_features_payload(payload.get("global") if isinstance(payload, dict) else None, max_features=12, max_pairs=4)
        option_features: dict[str, QueryFeatures] = {}
        options_payload = payload.get("options") if isinstance(payload, dict) else None
        for key in sorted(option_queries.keys()):
            item = options_payload.get(key) if isinstance(options_payload, dict) else None
            option_features[key] = parse_features_payload(item, max_features=5, max_pairs=3)
        query_feature_cache[cache_key] = (global_features, dict(option_features))
        return global_features, option_features, resp.usage
    except Exception as exc:
        logger.warning("LLM 特征抽取失败，回退正则：%s", repr(exc))
        global_features = extract_query_features(global_query)
        option_features = {k: extract_query_features(v) for k, v in option_queries.items()}
        query_feature_cache[cache_key] = (global_features, dict(option_features))
        return global_features, option_features, zero_token_usage()


def _normalize_feature_key(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    for sep in ("：", ":", "为"):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
    if "/" in raw:
        raw = raw.split("/", 1)[0].strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > 24:
        raw = raw[:24].strip()
    return raw


def _normalize_feature_value(text: str) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > 64:
        raw = raw[:64].strip()
    return raw


def parse_features_payload(payload: object, max_features: int, max_pairs: int) -> QueryFeatures:
    if not isinstance(payload, dict):
        return QueryFeatures(years=(), numbers=(), clauses=(), features=(), feature_pairs=())

    years_raw = payload.get("years")
    numbers_raw = payload.get("numbers")
    clauses_raw = payload.get("clauses")
    features_raw = payload.get("features")
    feature_pairs_raw = payload.get("feature_pairs")
    if features_raw is None:
        features_raw = payload.get("keywords")

    years = tuple(sorted({str(x).strip() for x in (years_raw if isinstance(years_raw, list) else []) if str(x).strip()}))
    years = tuple(y for y in years if re.fullmatch(r"(?:19|20)\d{2}", y))

    numbers = tuple(sorted({str(x).strip() for x in (numbers_raw if isinstance(numbers_raw, list) else []) if str(x).strip()}))
    clauses = tuple(sorted({str(x).strip() for x in (clauses_raw if isinstance(clauses_raw, list) else []) if str(x).strip()}))
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in (features_raw if isinstance(features_raw, list) else []):
        key = _normalize_feature_key(str(item))
        if not key or key in seen:
            continue
        cleaned.append(key)
        seen.add(key)
        if len(cleaned) >= max(1, int(max_features)):
            break
    cleaned_pairs: list[tuple[str, str]] = []
    pair_seen: set[tuple[str, str]] = set()
    for item in (feature_pairs_raw if isinstance(feature_pairs_raw, list) else []):
        if not isinstance(item, dict):
            continue
        key = _normalize_feature_key(item.get("key", ""))
        value = _normalize_feature_value(item.get("value", ""))
        if not key or not value:
            continue
        pair = (key, value)
        if pair in pair_seen:
            continue
        cleaned_pairs.append(pair)
        pair_seen.add(pair)
        if len(cleaned_pairs) >= max(1, int(max_pairs)):
            break
    return QueryFeatures(
        years=years,
        numbers=numbers,
        clauses=clauses,
        features=tuple(cleaned),
        feature_pairs=tuple(cleaned_pairs),
    )
