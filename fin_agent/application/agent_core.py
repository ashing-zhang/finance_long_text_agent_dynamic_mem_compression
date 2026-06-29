from __future__ import annotations

import json
import logging
import re
from fin_agent.compat import dataclass
from pathlib import Path

from fin_agent.application.domain_specialists import build_domain_supplement
from fin_agent.application.guardrails import extract_json_payload, normalize_answer_letters
from fin_agent.application.io_utils import write_answer_csv, write_evidence_jsonl, write_logs_csv
from fin_agent.application.retrieval import (
    DOMAIN_PROMPT_HINTS,
    QueryFeatures,
    adjust_chunk_size,
    bm25_rank,
    build_domain_reasoning_instruction,
    build_grouped_context,
    compute_domain_specific_boost,
    compute_symbolic_boost,
    expand_query_by_domain,
    extract_query_features,
    focus_chunk_content,
    trim_grouped_context,
)
from fin_agent.application.tracing import (
    EvaluationResult,
    QuestionTrace,
    RetrievalRoundTrace,
    RetrievalTrace,
    summarize_evidence_hits,
    sum_token_usage,
    token_usage_to_dict,
    truncate_text,
    zero_token_usage,
)
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


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    global_query: str
    option_queries: dict[str, str]
    features: QueryFeatures
    option_features: dict[str, QueryFeatures]


class FinanceLongTextAgent:
    def __init__(
        self,
        llm: OpenAiCompatibleChatClient,
        docs: DocumentRepository,
        retrieval: RetrievalConfig,
    ) -> None:
        self._llm = llm
        self._docs = docs
        self._retrieval = retrieval
        self._query_feature_cache: dict[str, tuple[QueryFeatures, dict[str, QueryFeatures]]] = {}

    def answer(self, q: Question) -> tuple[AnswerRecord, list[EvidenceSnippet], QuestionTrace]:
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
        parts = [q.question.strip(), "选项："]
        for key in sorted(q.options.keys()):
            parts.append(f"{key}. {q.options[key].strip()}")
        parts.append(expand_query_by_domain(domain=q.domain, text=q.question))
        return "\n".join(parts).strip()

    def _select_candidate_docs(self, q: Question, plan: RetrievalPlan) -> list[str]:
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
        system = (
            "你是一名严谨的金融审计师。"
            "你必须严格基于证据作答，不得使用常识或外部知识。"
            "如果上下文提供的证据不足以推导某选项，则该选项视为错误。"
            "请逐项审阅 A/B/C/D，但最终只能输出 JSON。"
        )
        user = self._format_user_prompt(q=q, context=context)
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _format_user_prompt(self, q: Question, context: str) -> str:
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


def build_relaxed_queries(q: Question, plan: RetrievalPlan) -> dict[str, str]:
    relaxed: dict[str, str] = {}
    feature_terms = list(plan.features.years) + list(plan.features.numbers) + list(plan.features.clauses)
    feature_terms.extend(plan.features.keywords[:12])
    feature_text = " ".join(feature_terms)
    for key in sorted(q.options.keys()):
        relaxed[key] = f"{q.options[key]} {feature_text} {expand_query_by_domain(q.domain, q.options[key])}".strip()
    return relaxed


def run_evaluation(run: RunConfig, llm: LlmConfig, retrieval: RetrievalConfig) -> EvaluationResult:
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

