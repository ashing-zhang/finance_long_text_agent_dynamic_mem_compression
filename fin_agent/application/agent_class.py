from __future__ import annotations

import logging

from fin_agent.application.answer_postprocess import normalize_answer
from fin_agent.application.context_builder import build_context
from fin_agent.application.planner import build_retrieval_plan
from fin_agent.application.prompt_builder import build_messages
from fin_agent.application.retrieval_pipeline import retrieve_evidence, select_candidate_docs
from fin_agent.application.trace_builder import build_question_trace
from fin_agent.application.retrieval import QueryFeatures
from fin_agent.application.tracing import sum_token_usage
from fin_agent.application.tracing import QuestionTrace
from fin_agent.domain.models import AnswerRecord, EvidenceSnippet, Question, RetrievalConfig
from fin_agent.infrastructure.data_access import DocumentRepository
from fin_agent.infrastructure.llm.openai_compatible_client import OpenAiCompatibleChatClient


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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
        plan, feature_usage = build_retrieval_plan(
            llm=self._llm,
            query_feature_cache=self._query_feature_cache,
            q=q,
        )
        doc_ids = select_candidate_docs(docs=self._docs, retrieval=self._retrieval, q=q, plan=plan)
        evidence, retrieval_trace = retrieve_evidence(docs=self._docs, retrieval=self._retrieval, q=q, doc_ids=doc_ids, plan=plan)
        context, refine_usage = build_context(
            llm=self._llm,
            docs=self._docs,
            retrieval=self._retrieval,
            q=q,
            plan=plan,
            doc_ids=doc_ids,
            evidence=evidence,
        )

        messages = build_messages(q=q, context=context)
        resp = self._llm.chat(messages)
        answer = normalize_answer(fmt=q.answer_format, text=resp.content)
        usage = sum_token_usage(feature_usage, refine_usage, resp.usage)
        record = AnswerRecord(qid=q.qid, answer=answer, usage=usage)
        trace = build_question_trace(
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
