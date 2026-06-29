from __future__ import annotations

import math
import re
from fin_agent.compat import dataclass

from fin_agent.domain.models import EvidenceSnippet, Question
from fin_agent.infrastructure.data_access import DocumentRepository

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
    keywords: tuple[str, ...]


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
    keywords = tuple(sorted(set(extract_keywords(normalized))))
    return QueryFeatures(years=years, numbers=numbers, clauses=clauses, keywords=keywords)


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


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    tokens: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(list(part))
        else:
            tokens.append(part)
    return tokens


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


def focus_chunk_content(domain: str, option_text: str, text: str) -> str:
    if domain == "financial_reports":
        focused = extract_financial_report_focus(option_text=option_text, text=text)
        return focused or text
    if domain == "insurance":
        focused = extract_insurance_focus(option_text=option_text, text=text)
        return focused or text
    return text


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

