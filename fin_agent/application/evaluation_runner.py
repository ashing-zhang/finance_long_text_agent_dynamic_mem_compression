from __future__ import annotations

import logging

from fin_agent.application.agent_class import FinanceLongTextAgent
from fin_agent.application.io_utils import write_answer_csv, write_evidence_jsonl, write_logs_csv
from fin_agent.application.tracing import EvaluationResult
from fin_agent.domain.models import LlmConfig, RetrievalConfig, RunConfig, TokenUsage
from fin_agent.infrastructure.data_access import DocumentRepository, QuestionRepository
from fin_agent.infrastructure.llm.openai_compatible_client import OpenAiCompatibleChatClient

logger = logging.getLogger(__name__)


def run_evaluation(run: RunConfig, llm: LlmConfig, retrieval: RetrievalConfig) -> EvaluationResult:
    questions_dir = run.dataset_root / run.questions_subdir
    raw_docs_root = run.dataset_root / run.raw_docs_subdir

    client = OpenAiCompatibleChatClient(llm)
    q_repo = QuestionRepository(questions_dir=questions_dir)
    d_repo = DocumentRepository(raw_root=raw_docs_root, markdown_converter=None)
    agent = FinanceLongTextAgent(llm=client, docs=d_repo, retrieval=retrieval)

    questions = q_repo.load_questions(split=run.split)
    logger.info("加载题目：%s（split=%s）", len(questions), run.split)

    records = []
    traces = []
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
