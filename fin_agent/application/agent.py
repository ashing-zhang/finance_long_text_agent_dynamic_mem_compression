from __future__ import annotations

import json
import logging
import math
import re
from fin_agent.compat import dataclass
from pathlib import Path

from fin_agent.application.domain_specialists import build_domain_supplement
from fin_agent.domain.models import (
    AnswerFormat,
    AnswerRecord,
    EvidenceSnippet,
    LlmConfig,
    Question,
    RetrievalConfig,
    RunConfig,
    TokenUsage,
)
from fin_agent.infrastructure.data_access import (
    DocumentRepository,
    QuestionRepository,
)
from fin_agent.infrastructure.llm.openai_compatible_client import (
    ChatMessage,
    OpenAiCompatibleChatClient,
)

logger = logging.getLogger(__name__)

DOMAIN_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "insurance": {
        "退保": ["解除保险合同", "现金价值", "退保金额", "保单账户价值"],
        "身故": ["身故保险金", "基本保险金额", "已交保费", "现金价值"],
        "领取": ["养老年金", "给付", "年金领取"],
    },
    "regulatory": {
        "受益所有人": ["受益人识别", "受益所有人识别", "备案信息"],
        "客户尽职调查": ["身份资料", "交易记录保存", "尽职调查"],
        "担保": ["股东大会", "特别决议", "普通决议"],
    },
    "financial_contracts": {
        "发行": ["发行规模", "募集说明书", "发行人"],
        "评级": ["主体信用评级", "债项评级", "AAA"],
        "赎回": ["回售", "违约责任", "受托管理人"],
    },
    "financial_reports": {
        "营业收入": ["收入", "营收", "营业总收入"],
        "净利润": ["归属于上市公司股东的净利润", "利润总额"],
        "现金流": ["经营活动产生的现金流量净额", "现金流量净额"],
        "研发投入": ["研发费用", "研发投入占营业收入比例"],
    },
    "research": {
        "行业趋势": ["景气度", "市场空间", "需求变化"],
        "公司比较": ["对比", "竞争格局", "盈利预测"],
        "观点": ["核心结论", "投资建议", "催化因素"],
    },
}

DOMAIN_PROMPT_HINTS: dict[str, str] = {
    "insurance": "保险题优先核对公式、给付条件、已交保费、现金价值与账户价值。",
    "regulatory": "监管题优先核对法条编号、施行日期、时限、比例与决议类型。",
    "financial_contracts": "金融合同题优先核对发行主体、评级、发行规模、受托管理人与条款引用。",
    "financial_reports": "财报题优先进行跨年度口径核对，必要时对比 2025 年报中的上年同期与 2024 年报本期。",
    "research": "研报题允许适度保留上下文，重点核对结论、对比关系与图表附近说明。",
}


def configure_logging(level: str = "INFO") -> None:
    """配置 logging。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """整批评测结果。"""

    answers: list[AnswerRecord]
    total_usage: TokenUsage
    traces: list["QuestionTrace"]


@dataclass(frozen=True, slots=True)
class RetrievalRoundTrace:
    """单轮检索的结构化 trace。"""

    round_index: int
    query_mode: str
    option_queries: dict[str, str]
    hit_count: int
    top_hits: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """检索阶段 trace。"""

    candidate_doc_ids: list[str]
    rounds: list[RetrievalRoundTrace]
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class QuestionTrace:
    """单题 trace，用于写入 logs.csv。"""

    qid: str
    domain: str
    question: str
    options: dict[str, str]
    thought_trace: dict[str, object]
    search_trace: dict[str, object]
    answer_trace: dict[str, object]


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """题目与选项特征。"""

    years: tuple[str, ...]
    numbers: tuple[str, ...]
    clauses: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """检索计划。"""

    global_query: str
    option_queries: dict[str, str]
    features: QueryFeatures
    option_features: dict[str, QueryFeatures]


class FinanceLongTextAgent:
    """金融长文本问答 Agent（文档级粗筛 + 选项级召回 + 严格推理）。"""

    def __init__(
        self,
        llm: OpenAiCompatibleChatClient,
        docs: DocumentRepository,
        retrieval: RetrievalConfig,
    ) -> None:
        """初始化 Agent。"""
        self._llm = llm
        self._docs = docs
        self._retrieval = retrieval
        self._query_feature_cache: dict[str, tuple[QueryFeatures, dict[str, QueryFeatures]]] = {}

    def answer(self, q: Question) -> tuple[AnswerRecord, list[EvidenceSnippet], QuestionTrace]:
        """回答单题，返回 answer.csv 行、证据片段与结构化 trace。"""
        plan, feature_usage = self._build_retrieval_plan(q)
        doc_ids = self._select_candidate_docs(q, plan)
        evidence, retrieval_trace = self._retrieve_evidence(q, doc_ids, plan)
        context, refine_usage = self._build_context(q, plan, doc_ids, evidence)

        messages = self._build_messages(q=q, context=context)
        resp = self._llm.chat(messages)
        answer = self._normalize_answer(q.answer_format, resp.content)
        usage = sum_token_usage(feature_usage, refine_usage, resp.usage)
        record = AnswerRecord(qid=q.qid, answer=answer, usage=usage)
        trace = self._build_question_trace(
            q=q,
            plan=plan,
            doc_ids=doc_ids,
            retrieval_trace=retrieval_trace,
            evidence=evidence,
            context=context,
            refine_usage=refine_usage,
            messages=messages,
            model_output=resp.content,
            answer=answer,
            answer_usage=resp.usage,
            total_usage=usage,
            feature_usage=feature_usage,
        )
        return record, evidence, trace

    def _build_retrieval_plan(self, q: Question) -> tuple[RetrievalPlan, TokenUsage]:
        """构造全局 query 与分选项 query。"""
        global_query = self._build_query(q)
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
        global_features, option_features, usage = self._extract_plan_features_with_llm(
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

    def _extract_plan_features_with_llm(
        self,
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
        cached = self._query_feature_cache.get(cache_key)
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
            resp = self._llm.chat(messages)
            payload = extract_json_payload(resp.content)
            global_features = self._parse_features_payload(payload.get("global") if isinstance(payload, dict) else None)
            option_features: dict[str, QueryFeatures] = {}
            options_payload = payload.get("options") if isinstance(payload, dict) else None
            for key in sorted(option_queries.keys()):
                item = options_payload.get(key) if isinstance(options_payload, dict) else None
                option_features[key] = self._parse_features_payload(item)
            self._query_feature_cache[cache_key] = (global_features, dict(option_features))
            return global_features, option_features, resp.usage
        except Exception as exc:
            logger.warning("LLM 特征抽取失败，回退正则：%s", repr(exc))
            global_features = self._extract_query_features_regex(global_query)
            option_features = {k: self._extract_query_features_regex(v) for k, v in option_queries.items()}
            self._query_feature_cache[cache_key] = (global_features, dict(option_features))
            return global_features, option_features, zero_token_usage()

    def _parse_features_payload(self, payload: object) -> QueryFeatures:
        if not isinstance(payload, dict):
            return QueryFeatures(years=(), numbers=(), clauses=(), keywords=())

        years_raw = payload.get("years")
        numbers_raw = payload.get("numbers")
        clauses_raw = payload.get("clauses")
        keywords_raw = payload.get("keywords")

        years = tuple(sorted({str(x).strip() for x in (years_raw if isinstance(years_raw, list) else []) if str(x).strip()}))
        years = tuple(y for y in years if re.fullmatch(r"(?:19|20)\d{2}", y))

        numbers = tuple(
            sorted({str(x).strip() for x in (numbers_raw if isinstance(numbers_raw, list) else []) if str(x).strip()})
        )
        clauses = tuple(
            sorted({str(x).strip() for x in (clauses_raw if isinstance(clauses_raw, list) else []) if str(x).strip()})
        )
        keywords = tuple(
            sorted({str(x).strip() for x in (keywords_raw if isinstance(keywords_raw, list) else []) if str(x).strip()})
        )
        return QueryFeatures(years=years, numbers=numbers, clauses=clauses, keywords=keywords)

    def _extract_query_features_regex(self, text: str) -> QueryFeatures:
        return extract_query_features(text)

    def _build_query(self, q: Question) -> str:
        """构造检索 query。"""
        parts = [q.question.strip(), "选项："]
        for key in sorted(q.options.keys()):
            parts.append(f"{key}. {q.options[key].strip()}")
        parts.append(expand_query_by_domain(domain=q.domain, text=q.question))
        return "\n".join(parts).strip()

    def _select_candidate_docs(self, q: Question, plan: RetrievalPlan) -> list[str]:
        """先做文档级粗筛，再返回候选文档。"""
        if q.doc_ids:
            return q.doc_ids

        all_doc_ids = self._docs.list_doc_ids(q.domain)
        if not all_doc_ids:
            logger.warning("领域下没有可选文档：domain=%s qid=%s", q.domain, q.qid)
            return []

        profiles: list[str] = []
        valid_doc_ids: list[str] = []
        for doc_id in all_doc_ids:
            try:
                profiles.append(
                    self._docs.load_doc_profile(
                        domain=q.domain,
                        doc_id=doc_id,
                        window_chars=self._retrieval.coarse_doc_window_chars,
                    )
                )
                valid_doc_ids.append(doc_id)
            except Exception as exc:
                logger.warning("文档 profile 构建失败：%s/%s %s", q.domain, doc_id, repr(exc))

        if not profiles:
            return []

        rankings = bm25_rank(query=plan.global_query, chunks=profiles)
        scored_doc_ids: list[tuple[str, float]] = []
        for index, score in rankings:
            doc_id = valid_doc_ids[index]
            profile_text = profiles[index]
            boosted = score + compute_symbolic_boost(
                text=profile_text,
                title=doc_id,
                features=plan.features,
                domain=q.domain,
            )
            scored_doc_ids.append((doc_id, boosted))
        scored_doc_ids.sort(key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in scored_doc_ids[: self._retrieval.doc_top_k]]

    def _retrieve_evidence(
        self,
        q: Question,
        doc_ids: list[str],
        plan: RetrievalPlan,
    ) -> tuple[list[EvidenceSnippet], RetrievalTrace]:
        """执行两阶段检索，返回按 score 排序的证据片段与 trace。"""
        merged: dict[str, EvidenceSnippet] = {}
        relaxed_queries = build_relaxed_queries(q=q, plan=plan)
        round_traces: list[RetrievalRoundTrace] = []

        for round_index in range(self._retrieval.max_routing_rounds):
            option_queries = plan.option_queries if round_index == 0 else relaxed_queries
            current_hits = self._retrieve_round(
                q=q,
                doc_ids=doc_ids,
                option_queries=option_queries,
                option_features=plan.option_features,
            )
            round_traces.append(
                RetrievalRoundTrace(
                    round_index=round_index + 1,
                    query_mode=("planned" if round_index == 0 else "relaxed"),
                    option_queries=dict(option_queries),
                    hit_count=len(current_hits),
                    top_hits=summarize_evidence_hits(current_hits, limit=8),
                )
            )
            for hit in current_hits:
                existing = merged.get(hit.chunk_id)
                if existing is None or hit.score > existing.score:
                    merged[hit.chunk_id] = hit
            if merged:
                break

        used_fallback = False
        if not merged:
            fallback = self._build_fallback_evidence(q=q, doc_ids=doc_ids)
            for hit in fallback:
                merged[hit.chunk_id] = hit
            used_fallback = True

        evidence = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        limited = evidence[: self._retrieval.top_k_chunks]
        retrieval_trace = RetrievalTrace(
            candidate_doc_ids=list(doc_ids),
            rounds=round_traces,
            used_fallback=used_fallback,
        )
        return limited, retrieval_trace

    def _retrieve_round(
        self,
        q: Question,
        doc_ids: list[str],
        option_queries: dict[str, str],
        option_features: dict[str, QueryFeatures],
    ) -> list[EvidenceSnippet]:
        """执行一轮选项级段落召回。"""
        results: list[EvidenceSnippet] = []
        for option_key, option_query in option_queries.items():
            option_feature = option_features.get(option_key) or self._extract_query_features_regex(option_query)
            for doc_id in doc_ids:
                try:
                    chunks = self._docs.load_chunks(
                        domain=q.domain,
                        doc_id=doc_id,
                        max_chars=adjust_chunk_size(domain=q.domain, base_size=self._retrieval.chunk_max_chars),
                    )
                except Exception as exc:
                    logger.warning("chunk 加载失败：%s/%s %s", q.domain, doc_id, repr(exc))
                    continue

                ranking_texts = [chunk.to_index_text() for chunk in chunks]
                rankings = bm25_rank(query=option_query, chunks=ranking_texts)
                per_doc_hits = 0
                for index, base_score in rankings:
                    chunk = chunks[index]
                    boosted = base_score + compute_symbolic_boost(
                        text=chunk.content,
                        title=chunk.title,
                        features=option_feature,
                        domain=q.domain,
                    )
                    boosted += compute_domain_specific_boost(
                        domain=q.domain,
                        option_text=q.options[option_key],
                        doc_id=chunk.doc_id,
                        title=chunk.title,
                        text=chunk.content,
                        features=option_feature,
                    )
                    focused_content = focus_chunk_content(
                        domain=q.domain,
                        option_text=q.options[option_key],
                        text=chunk.content,
                    )
                    results.append(
                        EvidenceSnippet(
                            doc_id=chunk.doc_id,
                            title=chunk.title,
                            content=focused_content,
                            score=float(boosted),
                            chunk_id=chunk.chunk_id,
                            option_key=option_key,
                        )
                    )
                    per_doc_hits += 1
                    if per_doc_hits >= self._retrieval.per_doc_top_k:
                        break

        results.sort(key=lambda item: item.score, reverse=True)
        limited: list[EvidenceSnippet] = []
        per_option_counter: dict[str, int] = {}
        for item in results:
            option_key = item.option_key or "_"
            used = per_option_counter.get(option_key, 0)
            if used >= self._retrieval.per_option_top_k:
                continue
            per_option_counter[option_key] = used + 1
            limited.append(item)
        return limited

    def _build_fallback_evidence(self, q: Question, doc_ids: list[str]) -> list[EvidenceSnippet]:
        """在检索失败时使用目录与首个 chunk 兜底。"""
        fallback: list[EvidenceSnippet] = []
        for doc_id in doc_ids[: self._retrieval.doc_top_k]:
            try:
                outline = self._docs.build_outline(domain=q.domain, doc_id=doc_id, max_items=6)
            except Exception as exc:
                logger.warning("fallback 大纲加载失败：%s/%s %s", q.domain, doc_id, repr(exc))
                outline = ""
            if outline:
                fallback.append(
                    EvidenceSnippet(
                        doc_id=doc_id,
                        title="文档大纲",
                        content=outline,
                        score=0.1,
                        chunk_id=f"{doc_id}::outline",
                        option_key=None,
                    )
                )
            try:
                chunks = self._docs.load_chunks(
                    domain=q.domain,
                    doc_id=doc_id,
                    max_chars=adjust_chunk_size(domain=q.domain, base_size=self._retrieval.chunk_max_chars),
                )
            except Exception:
                continue
            if chunks:
                chunk = chunks[0]
                fallback.append(
                    EvidenceSnippet(
                        doc_id=chunk.doc_id,
                        title=chunk.title,
                        content=chunk.content,
                        score=0.05,
                        chunk_id=chunk.chunk_id,
                        option_key=None,
                    )
                )
        return fallback

    def _build_context(
        self,
        q: Question,
        plan: RetrievalPlan,
        doc_ids: list[str],
        evidence: list[EvidenceSnippet],
    ) -> tuple[str, TokenUsage]:
        """按选项拼装证据链，并在必要时调用 Qwen 做精筛压缩。"""
        context = build_grouped_context(
            q=q,
            doc_ids=doc_ids,
            evidence=evidence,
            docs=self._docs,
        )
        supplement = build_domain_supplement(q=q, doc_ids=doc_ids, docs=self._docs, evidence=evidence)
        if supplement is not None and supplement.content:
            context = f"{context}\n\n## {supplement.title}\n{supplement.content}".strip()
        if len(context) <= self._retrieval.max_context_chars:
            return context, zero_token_usage()

        if len(context) > self._retrieval.refine_context_chars:
            refined, usage = self._refine_context_with_llm(q=q, plan=plan, context=context)
            if refined:
                refined = trim_grouped_context(
                    query=plan.global_query,
                    text=refined,
                    max_chars=self._retrieval.max_context_chars,
                )
                return refined, usage

        compressed = trim_grouped_context(
            query=plan.global_query,
            text=context,
            max_chars=self._retrieval.max_context_chars,
        )
        return compressed, zero_token_usage()

    def _refine_context_with_llm(
        self,
        q: Question,
        plan: RetrievalPlan,
        context: str,
    ) -> tuple[str, TokenUsage]:
        """使用 Qwen 做证据精筛与上下文压缩。"""
        options_text = "\n".join(f"{key}. {value}" for key, value in sorted(q.options.items()))
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "你是金融证据压缩器。"
                    "只保留能够直接支撑或反驳选项的证据。"
                    "不允许引入外部知识。"
                    "若某条证据无关，直接丢弃。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"领域：{q.domain}\n"
                    f"题目：{q.question}\n"
                    f"选项：\n{options_text}\n\n"
                    f"请按 A/B/C/D 分组输出最关键证据，每条证据保留 DocID 与 Title。\n"
                    f"如果某个选项没有直接证据，请输出 None。\n\n"
                    f"候选证据：\n{context}\n\n"
                    f"补充提示：{DOMAIN_PROMPT_HINTS.get(q.domain, '')}"
                ),
            ),
        ]
        try:
            response = self._llm.chat(messages)
            return response.content.strip(), response.usage
        except Exception as exc:
            logger.warning("上下文精筛失败：qid=%s error=%s", q.qid, repr(exc))
            return "", zero_token_usage()

    def _build_messages(self, q: Question, context: str) -> list[ChatMessage]:
        """构造最终推理 messages。"""
        system = (
            "你是一名严谨的金融审计师。"
            "你必须严格基于证据作答，不得使用常识或外部知识。"
            "如果上下文提供的证据不足以推导某选项，则该选项视为错误。"
            "请逐项审阅 A/B/C/D，但最终只能输出 JSON。"
        )
        user = self._format_user_prompt(q=q, context=context)
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _format_user_prompt(self, q: Question, context: str) -> str:
        """构造最终推理 prompt。"""
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

    def _normalize_answer(self, fmt: AnswerFormat, text: str) -> str:
        """从模型输出中抽取并规范化答案。"""
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

    def _build_question_trace(
        self,
        q: Question,
        plan: RetrievalPlan,
        doc_ids: list[str],
        retrieval_trace: RetrievalTrace,
        evidence: list[EvidenceSnippet],
        context: str,
        refine_usage: TokenUsage,
        messages: list[ChatMessage],
        model_output: str,
        answer: str,
        answer_usage: TokenUsage,
        total_usage: TokenUsage,
        feature_usage: TokenUsage,
    ) -> QuestionTrace:
        """构建单题的结构化 trace。"""
        thought_trace = {
            "global_query": plan.global_query,
            "option_queries": dict(plan.option_queries),
            "query_features": {
                "years": list(plan.features.years),
                "numbers": list(plan.features.numbers),
                "clauses": list(plan.features.clauses),
                "keywords": list(plan.features.keywords),
            },
            "domain_hint": DOMAIN_PROMPT_HINTS.get(q.domain, ""),
            "candidate_doc_ids": list(doc_ids),
        }
        search_trace = {
            "candidate_doc_ids": list(retrieval_trace.candidate_doc_ids),
            "used_fallback": retrieval_trace.used_fallback,
            "retrieval_rounds": [
                {
                    "round_index": item.round_index,
                    "query_mode": item.query_mode,
                    "option_queries": item.option_queries,
                    "hit_count": item.hit_count,
                    "top_hits": item.top_hits,
                }
                for item in retrieval_trace.rounds
            ],
            "evidence": summarize_evidence_hits(evidence, limit=len(evidence)),
        }
        answer_trace = {
            "context_chars": len(context),
            "context_preview": truncate_text(context, max_chars=1500),
            "messages": [{"role": m.role, "content": truncate_text(m.content, max_chars=4000)} for m in messages],
            "feature_usage": token_usage_to_dict(feature_usage),
            "refine_usage": token_usage_to_dict(refine_usage),
            "answer_usage": token_usage_to_dict(answer_usage),
            "total_usage": token_usage_to_dict(total_usage),
            "model_output": model_output,
            "normalized_answer": answer,
        }
        return QuestionTrace(
            qid=q.qid,
            domain=q.domain,
            question=q.question,
            options=dict(q.options),
            thought_trace=thought_trace,
            search_trace=search_trace,
            answer_trace=answer_trace,
        )


def extract_query_features(text: str) -> QueryFeatures:
    """抽取时间、数值、法条与关键词特征。"""
    normalized = normalize_text(text)
    years = tuple(sorted(set(re.findall(r"(?:19|20)\d{2}年?|(?:19|20)\d{2}", normalized))))
    numbers = tuple(
        sorted(
            set(
                re.findall(
                    r"\d+(?:\.\d+)?(?:%|％|亿元|万元|万|亿|元|股|倍|个工作日|工作日|日|天|个月|月|年)?",
                    normalized,
                )
            )
        )
    )
    clauses = tuple(sorted(set(re.findall(r"第[一二三四五六七八九十百千万0-9]+[条章节款]", normalized))))
    keywords = tuple(sorted(set(extract_keywords(normalized))))
    return QueryFeatures(years=years, numbers=numbers, clauses=clauses, keywords=keywords)


def extract_keywords(text: str) -> list[str]:
    """提取较有区分度的中文/英文关键词。"""
    stopwords = {
        "根据",
        "关于",
        "下列",
        "哪些",
        "是否",
        "可以",
        "相关",
        "公司",
        "规定",
        "以下",
        "正确",
        "错误",
        "准确",
        "结合",
        "判断",
    }
    keywords: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]{2,10}|[a-z0-9_]{3,}", text.lower()):
        if part in stopwords:
            continue
        keywords.append(part)
    return keywords


def expand_query_by_domain(domain: str, text: str) -> str:
    """使用领域词表扩展 query。"""
    expansions: list[str] = []
    synonyms = DOMAIN_SYNONYMS.get(domain, {})
    for key, terms in synonyms.items():
        if key in text:
            expansions.extend(terms)
    return " ".join(sorted(set(expansions)))


def build_relaxed_queries(q: Question, plan: RetrievalPlan) -> dict[str, str]:
    """构建第二轮放宽的 option queries。"""
    relaxed: dict[str, str] = {}
    feature_terms = list(plan.features.years) + list(plan.features.numbers) + list(plan.features.clauses)
    feature_terms.extend(plan.features.keywords[:12])
    feature_text = " ".join(feature_terms)
    for key in sorted(q.options.keys()):
        relaxed[key] = f"{q.options[key]} {feature_text} {expand_query_by_domain(q.domain, q.options[key])}".strip()
    return relaxed


def adjust_chunk_size(domain: str, base_size: int) -> int:
    """按领域调整 chunk 大小。"""
    if domain == "research":
        return int(base_size * 1.5)
    if domain == "regulatory":
        return max(1200, int(base_size * 0.9))
    return base_size


def build_domain_reasoning_instruction(domain: str) -> str:
    """为推理阶段生成更强的领域化要求。"""
    if domain == "financial_reports":
        return "先按年份列出关键指标，再核对本期、上年同期、同比、现金分红和研发投入口径，不一致时以证据不足处理。"
    if domain == "insurance":
        return "先提取公式、触发条件与代入值；若涉及身故保险金、退保金额、免赔额或赔付金额，先计算再判断选项。"
    return "先定位直接证据，再逐项判断，不要跳步。"


def build_grouped_context(
    q: Question,
    doc_ids: list[str],
    evidence: list[EvidenceSnippet],
    docs: DocumentRepository,
) -> str:
    """将证据按选项聚合为法庭证据链。"""
    sections: list[str] = []
    for option_key in sorted(q.options.keys()):
        sections.append(f"## 选项 {option_key}")
        sections.append(f"候选陈述：{q.options[option_key]}")
        option_hits = [item for item in evidence if item.option_key == option_key]
        if not option_hits:
            sections.append("None")
            sections.append("")
            continue
        for item in option_hits:
            sections.append(
                f"- [DocID: {item.doc_id} | Title: {item.title} | Score: {item.score:.3f}] {item.content}"
            )
        sections.append("")

    if evidence:
        sections.append("## 汇总证据")
        for item in evidence:
            sections.append(f"- [{item.doc_id} | {item.title}] {item.content}")
    else:
        sections.append("## 汇总证据")
        for doc_id in doc_ids:
            try:
                outline = docs.build_outline(domain=q.domain, doc_id=doc_id, max_items=6)
            except Exception:
                continue
            if outline:
                sections.append(f"- [{doc_id}] {outline}")
    return "\n".join(sections).strip()

 
def normalize_text(text: str) -> str:
    """规范化文本空白。"""
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    """对中英混合文本做轻量 tokenization（非 embedding）。"""
    normalized = normalize_text(text).lower()
    tokens: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(list(part))
        else:
            tokens.append(part)
    return tokens


def bm25_rank(query: str, chunks: list[str]) -> list[tuple[int, float]]:
    """对文本列表使用 BM25 打分并排序。"""
    q_terms = tokenize(query)
    if not q_terms:
        return [(index, 0.0) for index in range(len(chunks))]

    docs_terms = [tokenize(item) for item in chunks]
    doc_lens = [len(terms) for terms in docs_terms]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0

    df: dict[str, int] = {}
    for terms in docs_terms:
        for token in set(terms):
            df[token] = df.get(token, 0) + 1

    n = len(chunks)
    k1 = 1.5
    b = 0.75

    def idf(token: str) -> float:
        dft = df.get(token, 0)
        return math.log(1 + (n - dft + 0.5) / (dft + 0.5))

    q_tf: dict[str, int] = {}
    for token in q_terms:
        q_tf[token] = q_tf.get(token, 0) + 1

    scores: list[tuple[int, float]] = []
    for index, terms in enumerate(docs_terms):
        tf: dict[str, int] = {}
        for token in terms:
            tf[token] = tf.get(token, 0) + 1
        doc_len = doc_lens[index] or 1
        denom_base = k1 * (1 - b + b * (doc_len / (avgdl or 1.0)))
        score = 0.0
        for token in q_tf:
            frequency = tf.get(token, 0)
            if frequency <= 0:
                continue
            score += idf(token) * (frequency * (k1 + 1)) / (frequency + denom_base)
        scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def compute_symbolic_boost(text: str, title: str, features: QueryFeatures, domain: str) -> float:
    """对数值、法条、标题与领域关键字命中做额外加权。"""
    normalized_text = normalize_text(text)
    normalized_title = normalize_text(title)
    score = 0.0

    for year in features.years:
        if year and year in normalized_text:
            score += 0.4
    for number in features.numbers:
        if number and number in normalized_text:
            score += 0.25
    for clause in features.clauses:
        if clause and (clause in normalized_text or clause in normalized_title):
            score += 0.8
    for keyword in features.keywords[:12]:
        if keyword and (keyword in normalized_title or keyword in normalized_text):
            score += 0.1
    for synonyms in DOMAIN_SYNONYMS.get(domain, {}).values():
        for term in synonyms:
            if term in normalized_text:
                score += 0.05
    return score


def compute_domain_specific_boost(
    domain: str,
    option_text: str,
    doc_id: str,
    title: str,
    text: str,
    features: QueryFeatures,
) -> float:
    """对高价值领域施加更细的排序加权。"""
    normalized_text = normalize_text(text)
    normalized_title = normalize_text(title)
    normalized_option = normalize_text(option_text)
    score = 0.0

    if domain == "financial_reports":
        doc_years = set(extract_years_from_text(f"{doc_id} {title}"))
        option_years = set(extract_years_from_text(normalized_option)) | set(extract_years_from_text(" ".join(features.years)))
        if doc_years and option_years and doc_years & option_years:
            score += 0.6
        report_terms = (
            "营业收入",
            "营业总收入",
            "净利润",
            "归属于上市公司股东的净利润",
            "经营活动产生的现金流量净额",
            "研发投入",
            "研发费用",
            "现金分红",
            "分红",
            "上年同期",
            "本期",
            "同比",
        )
        score += 0.15 * count_term_hits(normalized_text, report_terms)
        if any(term in normalized_option for term in ("增长", "下降", "优于", "减少", "提升")) and any(
            term in normalized_text for term in ("同比", "较上年", "增减", "上年同期")
        ):
            score += 0.5

    if domain == "insurance":
        insurance_terms = (
            "身故保险金",
            "退保",
            "现金价值",
            "保单账户价值",
            "个人账户价值",
            "基本保险金额",
            "已交保费",
            "免赔额",
            "给付",
            "赔付",
            "年金",
            "较大者",
            "max",
            "乘以",
        )
        score += 0.18 * count_term_hits(normalized_text, insurance_terms)
        if any(symbol in text for symbol in ("max", "MAX", "*", "×", "÷", "+", "-")):
            score += 0.45
        if re.search(r"\d+(?:\.\d+)?(?:万元|万|亿元|亿|元)", normalized_text):
            score += 0.25
        if any(term in normalized_option for term in ("排序", "计算", "赔付", "退保", "身故")) and any(
            term in normalized_text for term in ("较大者", "已交保费", "现金价值", "账户价值", "免赔额")
        ):
            score += 0.55

    return score


def focus_chunk_content(domain: str, option_text: str, text: str) -> str:
    """针对高价值领域抽取更聚焦的证据内容。"""
    if domain == "financial_reports":
        focused = extract_financial_report_focus(option_text=option_text, text=text)
        return focused or text
    if domain == "insurance":
        focused = extract_insurance_focus(option_text=option_text, text=text)
        return focused or text
    return text


def extract_years_from_text(text: str) -> list[str]:
    """从文本中抽取年份。"""
    return re.findall(r"(?:19|20)\d{2}", text or "")


def count_term_hits(text: str, terms: tuple[str, ...] | list[str]) -> int:
    """统计术语命中数。"""
    return sum(1 for term in terms if term in text)


def trim_grouped_context(query: str, text: str, max_chars: int) -> str:
    """保留结构标题并对内容做抽取式压缩。"""
    lines = text.splitlines()
    headings = [line for line in lines if line.startswith("## ")]
    compressed_body = compress_text_by_overlap(query=query, text=text, max_chars=max_chars)
    prefix = "\n".join(headings[:8]).strip()
    if prefix:
        candidate = f"{prefix}\n{compressed_body}".strip()
        return candidate[:max_chars]
    return compressed_body[:max_chars]


def compress_text_by_overlap(query: str, text: str, max_chars: int) -> str:
    """按 query 的 token overlap 做抽取式压缩。"""
    q_terms = set(tokenize(query))
    if not q_terms:
        return (text or "")[:max_chars]

    sentences = re.split(r"(?<=[。！？!?\n])\s*", text)
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = tokenize(sentence)
        if not tokens:
            continue
        overlap = sum(1 for token in tokens if token in q_terms)
        scored.append((float(overlap) / (len(tokens) or 1), sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    buffer: list[str] = []
    total = 0
    for _, sentence in scored:
        if total + len(sentence) + 1 > max_chars:
            continue
        buffer.append(sentence)
        total += len(sentence) + 1
        if total >= max_chars:
            break
    return "\n".join(buffer).strip() or (text or "")[:max_chars]


def extract_financial_report_focus(option_text: str, text: str) -> str:
    """抽取财报题更聚焦的指标证据。"""
    terms = [
        "营业收入",
        "营业总收入",
        "净利润",
        "归属于上市公司股东的净利润",
        "经营活动产生的现金流量净额",
        "研发投入",
        "研发费用",
        "现金分红",
        "上年同期",
        "本期",
        "同比",
        "增长",
        "下降",
    ]
    if "分红" in option_text:
        terms.extend(["利润分配", "现金股利", "每10股"])
    if "研发" in option_text:
        terms.extend(["研发投入占营业收入比例", "研发人员"])
    if "现金流" in option_text:
        terms.extend(["现金流量", "经营活动"])

    return extract_focus_sentences(text=text, terms=terms, max_sentences=6, max_chars=900)


def extract_insurance_focus(option_text: str, text: str) -> str:
    """抽取保险题更聚焦的公式/金额证据。"""
    terms = [
        "身故保险金",
        "退保",
        "现金价值",
        "保单账户价值",
        "个人账户价值",
        "已交保费",
        "基本保险金额",
        "免赔额",
        "赔付",
        "给付",
        "年金",
        "账户价值",
        "较大者",
        "乘以",
        "max",
    ]
    if "白血病" in option_text:
        terms.extend(["白血病", "医保", "复发", "住院", "保险责任"])
    if "退保" in option_text:
        terms.extend(["解除保险合同", "退保费用"])
    return extract_focus_sentences(text=text, terms=terms, max_sentences=6, max_chars=900)


def extract_focus_sentences(text: str, terms: list[str], max_sentences: int, max_chars: int) -> str:
    """从文本中抽取命中术语较多的句子。"""
    sentences = re.split(r"(?<=[。；;！？!?\n])\s*", text)
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        hits = count_term_hits(sentence, terms)
        if hits <= 0 and not re.search(r"\d+(?:\.\d+)?(?:万元|万|亿元|亿|元|%|％)", sentence):
            continue
        if re.search(r"(max|MAX|较大者|乘以|免赔额|上年同期|同比|本期)", sentence):
            hits += 2
        scored.append((hits, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    total = 0
    for _, sentence in scored:
        if len(selected) >= max_sentences:
            break
        if total + len(sentence) + 1 > max_chars:
            continue
        selected.append(sentence)
        total += len(sentence) + 1
    return "\n".join(selected).strip()


def extract_json_payload(text: str) -> dict | None:
    """从模型输出中提取 JSON 对象。"""
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
    """规范化答案字母。"""
    letters = re.findall(r"[A-D]", (text or "").upper())
    if fmt in {AnswerFormat.MCQ, AnswerFormat.TF}:
        return letters[0] if letters else "A"
    unique = sorted(set(letters))
    return "".join(unique) if unique else "A"


def zero_token_usage() -> TokenUsage:
    """返回零 token usage。"""
    return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def sum_token_usage(*items: TokenUsage) -> TokenUsage:
    """汇总多个 token usage。"""
    prompt_tokens = sum(item.prompt_tokens for item in items)
    completion_tokens = sum(item.completion_tokens for item in items)
    total_tokens = sum(item.total_tokens for item in items)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def token_usage_to_dict(usage: TokenUsage) -> dict[str, int]:
    """将 TokenUsage 转为可序列化 dict。"""
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def truncate_text(text: str, max_chars: int) -> str:
    """截断文本，避免日志列过长。"""
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 15].rstrip() + "\n...(truncated)"


def summarize_evidence_hits(items: list[EvidenceSnippet], limit: int) -> list[dict[str, object]]:
    """将证据片段压缩为更适合日志记录的摘要。"""
    return [
        {
            "doc_id": item.doc_id,
            "title": item.title,
            "chunk_id": item.chunk_id,
            "score": round(float(item.score), 4),
            "option_key": item.option_key,
            "content_preview": truncate_text(item.content, max_chars=240),
        }
        for item in items[:limit]
    ]


def write_answer_csv(path: Path, records: list[AnswerRecord]) -> None:
    """写入符合提交格式的 answer.csv。"""
    import csv

    total_prompt = sum(r.usage.prompt_tokens for r in records)
    total_completion = sum(r.usage.completion_tokens for r in records)
    total_total = sum(r.usage.total_tokens for r in records)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", total_prompt, total_completion, total_total])
        for r in records:
            writer.writerow([r.qid, r.answer, r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens])


def write_logs_csv(path: Path, traces: list[QuestionTrace]) -> None:
    """写入包含单题 thought/search/answer trace 的 logs.csv。"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "qid",
                "domain",
                "question",
                "options_json",
                "thought_trace_json",
                "search_trace_json",
                "answer_trace_json",
            ]
        )
        for trace in traces:
            writer.writerow(
                [
                    trace.qid,
                    trace.domain,
                    trace.question,
                    json.dumps(trace.options, ensure_ascii=False),
                    json.dumps(trace.thought_trace, ensure_ascii=False),
                    json.dumps(trace.search_trace, ensure_ascii=False),
                    json.dumps(trace.answer_trace, ensure_ascii=False),
                ]
            )


def write_evidence_jsonl(path: Path, q: Question, evidence: list[EvidenceSnippet]) -> None:
    """追加写入单题证据到 jsonl（可选调试）。"""
    payload = {
        "qid": q.qid,
        "domain": q.domain,
        "answer_format": q.answer_format.value,
        "evidence_retrieval": [
            {
                "doc_id": e.doc_id,
                "title": e.title,
                "chunk_id": e.chunk_id,
                "score": e.score,
                "option_key": e.option_key,
                "quoted_clause": e.content,
            }
            for e in evidence
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_evaluation(run: RunConfig, llm: LlmConfig, retrieval: RetrievalConfig) -> EvaluationResult:
    """按配置执行评测并产出 answer.csv。"""
    questions_dir = run.dataset_root / run.questions_subdir
    raw_docs_root = run.dataset_root / run.raw_docs_subdir

    q_repo = QuestionRepository(questions_dir=questions_dir)
    d_repo = DocumentRepository(raw_root=raw_docs_root)

    client = OpenAiCompatibleChatClient(llm)
    agent = FinanceLongTextAgent(llm=client, docs=d_repo, retrieval=retrieval)

    questions = q_repo.load_questions(split=run.split)
    logger.info("加载题目：%s（split=%s）", len(questions), run.split)

    records: list[AnswerRecord] = []
    traces: list[QuestionTrace] = []
    total_prompt = 0
    total_completion = 0

    for idx, q in enumerate(questions, start=1):
        logger.info("作答中：%s (%s/%s)", q.qid, idx, len(questions))
        record, evidence, trace = agent.answer(q)
        records.append(record)
        traces.append(trace)
        total_prompt += record.usage.prompt_tokens
        total_completion += record.usage.completion_tokens

        if run.output_evidence_jsonl is not None:
            write_evidence_jsonl(run.output_evidence_jsonl, q=q, evidence=evidence)

    write_answer_csv(run.output_csv, records=records)
    write_logs_csv(run.output_csv.parent / "logs.csv", traces=traces)
    total_usage = TokenUsage(
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
    )
    return EvaluationResult(answers=records, total_usage=total_usage, traces=traces)
