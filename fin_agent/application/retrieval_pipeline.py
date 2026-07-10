from __future__ import annotations

import logging

from fin_agent.application.planner import RetrievalPlan
from fin_agent.application.retrieval import (
    QueryFeatures,
    adjust_chunk_size,
    bm25_rank,
    collect_grep_terms,
    compute_domain_specific_boost,
    compute_grep_style_boost,
    compute_symbolic_boost,
    expand_query_by_domain,
    extract_query_features,
    focus_chunk_content,
)
from fin_agent.application.tracing import (
    RetrievalRoundTrace,
    RetrievalTrace,
    summarize_evidence_hits,
    summarize_hits_by_option_doc,
    summarize_hits_by_option_doc_flat,
)
from fin_agent.domain.models import EvidenceSnippet, Question, RetrievalConfig
from fin_agent.infrastructure.data_access import DocumentRepository

logger = logging.getLogger(__name__)


def build_bm25_query(option_query: str, features: QueryFeatures) -> str:
    terms: list[str] = []
    terms.extend([item for item in features.features[:12] if item])
    terms.extend([item for item in features.expanded_terms[:48] if item])
    for key, value in features.feature_pairs[:8]:
        if key:
            terms.append(key)
        if value:
            terms.append(value)
        if key and value:
            terms.append(f"{key} {value}")
    terms.extend([item for item in features.years if item])
    terms.extend([item for item in features.numbers if item])
    terms.extend([item for item in features.clauses if item])
    compact = " ".join(terms).strip()
    return compact or option_query


def build_relaxed_queries(q: Question, plan: RetrievalPlan) -> dict[str, str]:
    relaxed: dict[str, str] = {}
    feature_terms = list(plan.features.years) + list(plan.features.numbers) + list(plan.features.clauses)
    feature_terms.extend(plan.features.features[:12])
    feature_terms.extend(plan.features.expanded_terms[:48])
    for key, value in plan.features.feature_pairs[:8]:
        if key:
            feature_terms.append(key)
        if value:
            feature_terms.append(value)
    feature_text = " ".join(feature_terms)
    for key in sorted(q.options.keys()):
        relaxed[key] = f"{q.options[key]} {feature_text} {expand_query_by_domain(q.domain, q.options[key])}".strip()
    return relaxed


def select_candidate_docs(docs: DocumentRepository, retrieval: RetrievalConfig, q: Question, plan: RetrievalPlan) -> list[str]:
    if q.doc_ids:
        return q.doc_ids

    all_doc_ids = docs.list_doc_ids(q.domain)
    if not all_doc_ids:
        logger.warning("领域下没有可选文档：domain=%s qid=%s", q.domain, q.qid)
        return []

    profiles: list[str] = []
    valid_doc_ids: list[str] = []
    for doc_id in all_doc_ids:
        try:
            profiles.append(
                docs.load_doc_profile(
                    domain=q.domain,
                    doc_id=doc_id,
                    window_chars=retrieval.coarse_doc_window_chars,
                )
            )
            valid_doc_ids.append(doc_id)
        except Exception as exc:
            logger.warning("文档 profile 构建失败：%s/%s %s", q.domain, doc_id, repr(exc))

    if not profiles:
        return []

    rankings = bm25_rank(query=plan.global_query, chunks=profiles)
    grep_terms = (
        collect_grep_terms(
            query=plan.global_query,
            features=plan.features,
            limit=retrieval.grep_terms_per_query,
        )
        if retrieval.enable_grep_retrieval
        else ()
    )
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
        if grep_terms:
            boosted += compute_grep_style_boost(
                text=profile_text,
                title=doc_id,
                terms=grep_terms,
            )
        scored_doc_ids.append((doc_id, boosted))
    scored_doc_ids.sort(key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in scored_doc_ids[: retrieval.doc_top_k]]


def retrieve_evidence(
    docs: DocumentRepository,
    retrieval: RetrievalConfig,
    q: Question,
    doc_ids: list[str],
    plan: RetrievalPlan,
) -> tuple[list[EvidenceSnippet], RetrievalTrace]:
    merged: dict[tuple[str | None, str], EvidenceSnippet] = {}
    relaxed_queries = build_relaxed_queries(q=q, plan=plan)
    round_traces: list[RetrievalRoundTrace] = []

    for round_index in range(retrieval.max_routing_rounds):
        option_queries = plan.option_queries if round_index == 0 else relaxed_queries
        literal_terms = {
            option_key: list(
                collect_grep_terms(
                    query=option_query,
                    features=(plan.option_features.get(option_key) or extract_query_features(option_query)),
                    limit=retrieval.grep_terms_per_query,
                )
            )
            if retrieval.enable_grep_retrieval
            else []
            for option_key, option_query in option_queries.items()
        }
        bm25_queries = {
            option_key: build_bm25_query(
                option_query=option_query,
                features=(plan.option_features.get(option_key) or extract_query_features(option_query)),
            )
            for option_key, option_query in option_queries.items()
        }
        current_hits = retrieve_round(
            docs=docs,
            retrieval=retrieval,
            q=q,
            doc_ids=doc_ids,
            option_queries=option_queries,
            bm25_queries=bm25_queries,
            option_features=plan.option_features,
            literal_terms=literal_terms,
        )
        round_traces.append(
            RetrievalRoundTrace(
                round_index=round_index + 1,
                query_mode=("planned" if round_index == 0 else "relaxed"),
                option_queries=dict(bm25_queries),
                literal_terms={key: list(value) for key, value in literal_terms.items()},
                hit_count=len(current_hits),
                top_hits=summarize_hits_by_option_doc_flat(
                    items=current_hits,
                    option_keys=sorted(option_queries.keys()),
                    doc_ids=doc_ids,
                    per_doc_limit=retrieval.per_doc_top_k,
                ),
                option_doc_hits=summarize_hits_by_option_doc(
                    items=current_hits,
                    option_keys=sorted(option_queries.keys()),
                    doc_ids=doc_ids,
                ),
            )
        )
        for hit in current_hits:
            merge_key = (hit.option_key, hit.chunk_id)
            existing = merged.get(merge_key)
            if existing is None or hit.score > existing.score:
                merged[merge_key] = hit
        if merged:
            break

    used_fallback = False
    if not merged:
        fallback = build_fallback_evidence(docs=docs, retrieval=retrieval, q=q, doc_ids=doc_ids)
        for hit in fallback:
            merged[(hit.option_key, hit.chunk_id)] = hit
        used_fallback = True

    evidence = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    limited = select_coverage_preserving_evidence(
        evidence=evidence,
        doc_ids=doc_ids,
        option_keys=sorted(plan.option_queries.keys()),
        per_doc_top_k=retrieval.per_doc_top_k,
        soft_limit=retrieval.top_k_chunks,
    )
    retrieval_trace = RetrievalTrace(
        candidate_doc_ids=list(doc_ids),
        rounds=round_traces,
        used_fallback=used_fallback,
    )
    return limited, retrieval_trace


def retrieve_round(
    docs: DocumentRepository,
    retrieval: RetrievalConfig,
    q: Question,
    doc_ids: list[str],
    option_queries: dict[str, str],
    bm25_queries: dict[str, str],
    option_features: dict[str, QueryFeatures],
    literal_terms: dict[str, list[str]],
) -> list[EvidenceSnippet]:
    results: list[EvidenceSnippet] = []
    cached_chunks: dict[str, list] = {}
    for option_key, option_query in option_queries.items():
        option_feature = option_features.get(option_key) or extract_query_features(option_query)
        bm25_query = bm25_queries.get(option_key) or build_bm25_query(option_query=option_query, features=option_feature)
        grep_terms = tuple(literal_terms.get(option_key) or ())

        for doc_id in doc_ids:
            chunks = cached_chunks.get(doc_id)
            if chunks is None:
                try:
                    chunks = docs.load_chunks(
                        domain=q.domain,
                        doc_id=doc_id,
                        max_chars=adjust_chunk_size(domain=q.domain, base_size=retrieval.chunk_max_chars),
                    )
                except Exception as exc:
                    logger.warning("chunk 加载失败：%s/%s %s", q.domain, doc_id, repr(exc))
                    cached_chunks[doc_id] = []
                    continue
                cached_chunks[doc_id] = chunks
            if not chunks:
                continue

            per_tuple: list[EvidenceSnippet] = []
            ranking_texts = [chunk.to_index_text() for chunk in chunks]
            rankings = bm25_rank(query=bm25_query, chunks=ranking_texts)
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
                if grep_terms:
                    boosted += compute_grep_style_boost(
                        text=chunk.content,
                        title=chunk.title,
                        terms=grep_terms,
                    )
                focused_content = focus_chunk_content(
                    domain=q.domain,
                    option_text=q.options[option_key],
                    text=chunk.content,
                    grep_terms=grep_terms,
                    grep_context_window=retrieval.grep_context_window,
                )
                per_tuple.append(
                    EvidenceSnippet(
                        doc_id=chunk.doc_id,
                        title=chunk.title,
                        content=focused_content,
                        score=float(boosted),
                        chunk_id=chunk.chunk_id,
                        option_key=option_key,
                    )
                )
            per_tuple.sort(key=lambda item: item.score, reverse=True)
            results.extend(per_tuple[: retrieval.per_doc_top_k])

    return results


def build_fallback_evidence(
    docs: DocumentRepository,
    retrieval: RetrievalConfig,
    q: Question,
    doc_ids: list[str],
) -> list[EvidenceSnippet]:
    fallback: list[EvidenceSnippet] = []
    for doc_id in doc_ids[: retrieval.doc_top_k]:
        try:
            outline = docs.build_outline(domain=q.domain, doc_id=doc_id, max_items=6)
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
            chunks = docs.load_chunks(
                domain=q.domain,
                doc_id=doc_id,
                max_chars=adjust_chunk_size(domain=q.domain, base_size=retrieval.chunk_max_chars),
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


def select_coverage_preserving_evidence(
    evidence: list[EvidenceSnippet],
    doc_ids: list[str],
    option_keys: list[str],
    per_doc_top_k: int,
    soft_limit: int,
) -> list[EvidenceSnippet]:
    """优先保留每个选项在每个候选文档中的 top n 证据，再补充高分证据。"""
    if not evidence:
        return []

    doc_id_set = set(doc_ids)
    option_key_set = set(option_keys)
    selected: list[EvidenceSnippet] = []
    selected_keys: set[tuple[str | None, str]] = set()

    grouped: dict[tuple[str, str], list[EvidenceSnippet]] = {}
    leftovers: list[EvidenceSnippet] = []
    for item in evidence:
        if item.option_key is None:
            leftovers.append(item)
            continue
        if item.option_key not in option_key_set or item.doc_id not in doc_id_set:
            leftovers.append(item)
            continue
        grouped.setdefault((item.option_key, item.doc_id), []).append(item)

    for option_key in option_keys:
        for doc_id in doc_ids:
            items = grouped.get((option_key, doc_id))
            if not items:
                continue
            items.sort(key=lambda item: item.score, reverse=True)
            for item in items[:per_doc_top_k]:
                item_key = (item.option_key, item.chunk_id)
                if item_key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(item_key)

    target_size = max(soft_limit, len(selected))
    remaining = [item for item in evidence if (item.option_key, item.chunk_id) not in selected_keys]
    remaining.sort(key=lambda item: item.score, reverse=True)
    for item in remaining:
        item_key = (item.option_key, item.chunk_id)
        if item_key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(item_key)
        if len(selected) >= target_size:
            break
    return selected
