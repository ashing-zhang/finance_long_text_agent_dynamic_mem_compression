from __future__ import annotations

import logging
import math
import os
import re
from functools import lru_cache
from fin_agent.compat import dataclass

from fin_agent.domain.models import EvidenceSnippet, Question
from fin_agent.infrastructure.data_access import DocumentRepository

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


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    years: tuple[str, ...]
    numbers: tuple[str, ...]
    clauses: tuple[str, ...]
    features: tuple[str, ...]
    feature_pairs: tuple[tuple[str, str], ...]
    expanded_terms: tuple[str, ...] = ()


def extract_query_features(text: str) -> QueryFeatures:
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
    features = tuple(sorted(set(extract_keywords(normalized))))
    return QueryFeatures(years=years, numbers=numbers, clauses=clauses, features=features, feature_pairs=(), expanded_terms=())


def extract_keywords(text: str) -> list[str]:
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
    expansions: list[str] = []
    synonyms = DOMAIN_SYNONYMS.get(domain, {})
    for key, terms in synonyms.items():
        if key in text:
            expansions.extend(terms)
    return " ".join(sorted(set(expansions)))


def adjust_chunk_size(domain: str, base_size: int) -> int:
    if domain == "research":
        return int(base_size * 1.5)
    if domain == "regulatory":
        return max(1200, int(base_size * 0.9))
    return base_size


def build_domain_reasoning_instruction(domain: str) -> str:
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
            sections.append(f"- [DocID: {item.doc_id} | Title: {item.title} | Score: {item.score:.3f}] {item.content}")
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
    return re.sub(r"\s+", " ", text or "").strip()


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _is_high_signal_grep_term(term: str) -> bool:
    normalized = normalize_text(term)
    if not normalized:
        return False
    if len(normalized) > 64:
        return False
    if normalized in {"根据", "下列", "哪些", "是否", "有关", "要求", "规定", "说法", "描述"}:
        return False
    if any(normalized.startswith(prefix) for prefix in ("是否", "根据", "下列", "哪个", "哪些", "何种", "关于")):
        return False
    if re.fullmatch(r"(?:19|20)\d{2}年?|第[一二三四五六七八九十百千万0-9]+[条章节款]", normalized):
        return True
    if re.search(r"\d", normalized):
        return len(normalized) >= 2
    if _contains_cjk(normalized):
        compact = normalized.replace(" ", "")
        return 2 <= len(compact) <= 10
    return len(normalized) >= 3


def _grep_term_weight(term: str) -> float:
    normalized = normalize_text(term)
    if re.fullmatch(r"第[一二三四五六七八九十百千万0-9]+[条章节款]", normalized):
        return 2.4
    if re.fullmatch(r"(?:19|20)\d{2}年?", normalized):
        return 1.8
    if re.search(r"\d", normalized):
        return 1.4
    if len(normalized) >= 6:
        return 1.2
    return 1.0


def collect_grep_terms(query: str, features: QueryFeatures, *, limit: int) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        normalized = normalize_text(str(value or ""))
        if not normalized:
            return
        if normalized in seen:
            return
        if not _is_high_signal_grep_term(normalized):
            return
        terms.append(normalized)
        seen.add(normalized)

    for item in features.clauses:
        push(item)
    for item in features.years:
        push(item)
    for item in features.numbers:
        push(item)
    for key, value in features.feature_pairs[:8]:
        push(key)
        push(value)
        if key and value:
            push(f"{key} {value}")
            if _contains_cjk(key) and _contains_cjk(value):
                push(f"{key}{value}")
    for item in features.features[:12]:
        push(item)
    for item in features.expanded_terms[:48]:
        push(item)
    for item in re.findall(r"[\u4e00-\u9fff]{2,16}|[a-zA-Z0-9_]{3,}", normalize_text(query)):
        push(item)

    return tuple(terms[: max(1, int(limit))])


def compute_grep_style_boost(text: str, title: str, terms: tuple[str, ...] | list[str]) -> float:
    normalized_text = normalize_text(text)
    normalized_title = normalize_text(title)
    if not normalized_text and not normalized_title:
        return 0.0

    score = 0.0
    distinct_hits = 0
    for term in terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        title_hits = normalized_title.count(normalized)
        text_hits = normalized_text.count(normalized)
        if title_hits <= 0 and text_hits <= 0:
            continue
        distinct_hits += 1
        weight = _grep_term_weight(normalized)
        score += min(title_hits, 2) * (0.45 * weight)
        score += min(text_hits, 3) * (0.3 * weight)
        if title_hits > 0 and text_hits > 0:
            score += 0.18 * weight

    if distinct_hits >= 2:
        score += min(0.9, distinct_hits * 0.15)
    return score


def extract_grep_focus(text: str, terms: tuple[str, ...] | list[str], *, context_window: int, max_chars: int) -> str:
    source = text or ""
    if not source:
        return ""

    spans: list[tuple[int, int]] = []
    for term in terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        start = 0
        captured = 0
        while captured < 2:
            index = source.find(normalized, start)
            if index < 0:
                break
            spans.append(
                (
                    max(0, index - max(20, int(context_window))),
                    min(len(source), index + len(normalized) + max(20, int(context_window))),
                )
            )
            start = index + len(normalized)
            captured += 1
        if len(spans) >= 6:
            break

    if not spans:
        return ""

    spans.sort(key=lambda item: item[0])
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1] + 20:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    pieces: list[str] = []
    total = 0
    for start, end in merged:
        snippet = source[start:end].strip()
        if not snippet:
            continue
        if start > 0:
            snippet = f"...{snippet}"
        if end < len(source):
            snippet = f"{snippet}..."
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
        pieces.append(snippet)
        total += len(snippet) + 1
        if total >= max_chars:
            break
    return "\n".join(pieces).strip()


def _merge_focus_content(*parts: str, max_chars: int) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    total = 0
    for part in parts:
        normalized = (part or "").strip()
        if not normalized or normalized in seen:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(normalized) > remaining:
            normalized = normalized[:remaining].rstrip()
        merged.append(normalized)
        seen.add(normalized)
        total += len(normalized) + 1
    return "\n".join(merged).strip()


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _load_qwen_tokenizer():
    model_name = os.getenv("FIN_AGENT_TOKENIZER_MODEL", "Qwen/Qwen3-0.6B")
    if os.getenv("FIN_AGENT_TOKENIZER_BACKEND", "qwen").strip().lower() != "qwen":
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        logger.warning("未安装 transformers，回退到正则分词：error=%s", repr(exc))
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=_read_bool_env("FIN_AGENT_TOKENIZER_TRUST_REMOTE_CODE", True),
            local_files_only=_read_bool_env("FIN_AGENT_TOKENIZER_LOCAL_ONLY", False),
        )
        logger.info("已启用 Qwen tokenizer：model=%s", model_name)
        return tokenizer
    except Exception as exc:
        logger.warning("加载 Qwen tokenizer 失败，回退到正则分词：model=%s error=%s", model_name, repr(exc))
        return None


@lru_cache(maxsize=8192)
def _tokenize_with_qwen(normalized_text: str) -> tuple[str, ...]:
    if not normalized_text:
        return ()
    tokenizer = _load_qwen_tokenizer()
    if tokenizer is None:
        return ()
    try:
        input_ids = tokenizer(normalized_text, add_special_tokens=False).input_ids
    except Exception as exc:
        logger.warning("Qwen tokenizer 分词失败，回退到正则分词：error=%s", repr(exc))
        return ()

    tokens: list[str] = []
    for token_id in input_ids:
        cleaned = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip().lower()
        if not cleaned:
            continue
        if cleaned.startswith("<|") and cleaned.endswith("|>"):
            continue
        if re.fullmatch(r"[\W_]+", cleaned, flags=re.UNICODE):
            continue
        tokens.append(cleaned)
    return tuple(tokens)


def _tokenize_by_regex(normalized_text: str) -> list[str]:
    lowered = normalized_text.lower()
    tokens: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", lowered):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(list(part))
        else:
            tokens.append(part)
    return tokens


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    qwen_tokens = list(_tokenize_with_qwen(normalized))
    if qwen_tokens:
        return qwen_tokens
    return _tokenize_by_regex(normalized)


def bm25_rank(query: str, chunks: list[str]) -> list[tuple[int, float]]:
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
    for feature in features.features[:12]:
        if feature and (feature in normalized_title or feature in normalized_text):
            score += 0.1
    for key, value in features.feature_pairs[:8]:
        key_hit = bool(key) and (key in normalized_title or key in normalized_text)
        value_hit = bool(value) and (value in normalized_title or value in normalized_text)
        if key_hit:
            score += 0.12
        if value_hit:
            score += 0.35
        if key_hit and value_hit:
            score += 0.25
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
    normalized_text = normalize_text(text)
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


def focus_chunk_content(
    domain: str,
    option_text: str,
    text: str,
    *,
    grep_terms: tuple[str, ...] | list[str] = (),
    grep_context_window: int = 120,
) -> str:
    grep_focus = extract_grep_focus(
        text=text,
        terms=grep_terms,
        context_window=grep_context_window,
        max_chars=900,
    )
    if domain == "financial_reports":
        focused = extract_financial_report_focus(option_text=option_text, text=text)
        return _merge_focus_content(grep_focus, focused, max_chars=900) or text
    if domain == "insurance":
        focused = extract_insurance_focus(option_text=option_text, text=text)
        return _merge_focus_content(grep_focus, focused, max_chars=900) or text
    return grep_focus or text


def extract_years_from_text(text: str) -> list[str]:
    return re.findall(r"(?:19|20)\d{2}", text or "")


def count_term_hits(text: str, terms: tuple[str, ...] | list[str]) -> int:
    return sum(1 for term in terms if term in text)


def trim_grouped_context(query: str, text: str, max_chars: int) -> str:
    lines = text.splitlines()
    headings = [line for line in lines if line.startswith("## ")]
    compressed_body = compress_text_by_overlap(query=query, text=text, max_chars=max_chars)
    prefix = "\n".join(headings[:8]).strip()
    if prefix:
        candidate = f"{prefix}\n{compressed_body}".strip()
        return candidate[:max_chars]
    return compressed_body[:max_chars]


def compress_text_by_overlap(query: str, text: str, max_chars: int) -> str:
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
