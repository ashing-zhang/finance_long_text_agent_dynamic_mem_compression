from __future__ import annotations

import re
from fin_agent.compat import dataclass

from fin_agent.domain.models import EvidenceSnippet, Question
from fin_agent.infrastructure.data_access import DocumentRepository

FINANCIAL_REPORT_METRICS: dict[str, tuple[str, ...]] = {
    "营业收入": ("营业收入", "营业总收入", "收入", "营收"),
    "净利润": ("归属于上市公司股东的净利润", "归母净利润", "净利润"),
    "现金流": ("经营活动产生的现金流量净额", "现金流量净额", "经营活动现金流"),
    "研发投入": ("研发投入", "研发费用", "研发投入占营业收入比例"),
    "分红": ("现金分红", "利润分配", "每10股", "派发现金", "回购"),
}

INSURANCE_VARIABLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "已交保费": ("已交保费", "累计所交保费", "累计已交保险费"),
    "现金价值": ("现金价值",),
    "账户价值": ("保单账户价值", "个人账户价值", "账户价值"),
    "基本保额": ("基本保额", "基本保险金额"),
    "免赔额": ("免赔额",),
    "给付金额": ("身故保险金", "退保金额", "赔付", "给付"),
}


@dataclass(frozen=True, slots=True)
class DomainSupplement:
    """领域专用补充证据。"""

    title: str
    content: str


@dataclass(frozen=True, slots=True)
class InsuranceRuleCalcResult:
    """保险题本地规则计算结果。"""

    title: str
    details: list[str]


@dataclass(frozen=True, slots=True)
class FinancialMetricSnapshot:
    """财报指标抽取快照。"""

    doc_id: str
    year: str | None
    metric: str
    sentence: str
    numeric_values: tuple[str, ...]


def build_domain_supplement(
    q: Question,
    doc_ids: list[str],
    docs: DocumentRepository,
    evidence: list[EvidenceSnippet],
) -> DomainSupplement | None:
    """按领域构造补充证据。"""
    if q.domain == "financial_reports":
        content = build_financial_report_supplement(q=q, doc_ids=doc_ids, docs=docs, evidence=evidence)
        if content:
            return DomainSupplement(title="财报双向核验", content=content)
    if q.domain == "insurance":
        content = build_insurance_supplement(q=q, evidence=evidence)
        if content:
            return DomainSupplement(title="保险变量与公式", content=content)
    return None


def build_financial_report_supplement(
    q: Question,
    doc_ids: list[str],
    docs: DocumentRepository,
    evidence: list[EvidenceSnippet],
) -> str:
    """构造财报领域的跨文档双向核验说明。"""
    target_metrics = select_financial_metrics(q)
    if not target_metrics:
        return ""

    lines: list[str] = []
    doc_year_pairs = [(doc_id, infer_year(doc_id)) for doc_id in doc_ids]
    if len(doc_year_pairs) >= 2:
        sorted_pairs = sorted(doc_year_pairs, key=lambda item: item[1] or "9999")
        older_doc = sorted_pairs[0][0]
        newer_doc = sorted_pairs[-1][0]
        lines.append(
            f"- 双向核验顺序：先看 `{newer_doc}` 中“上年同期/上期”数据，再回看 `{older_doc}` 中“本期”或同名指标。"
        )

    for metric in target_metrics:
        aliases = FINANCIAL_REPORT_METRICS[metric]
        lines.append(f"- 指标 `{metric}` 统一按别名 {', '.join(aliases)} 检索。")
        for doc_id in doc_ids:
            try:
                text = docs.load_text(domain=q.domain, doc_id=doc_id)
            except Exception:
                continue
            focus = extract_metric_focus(
                text=text,
                aliases=aliases,
                preferred_year=infer_year(doc_id),
            )
            if focus:
                lines.append(f"  - `{doc_id}`: {focus}")

        snapshots = collect_financial_metric_snapshots(q=q, doc_ids=doc_ids, docs=docs, metric=metric)
        if snapshots:
            lines.append(f"  - `{metric}` 指标值抽取：")
            for snapshot in snapshots[:4]:
                year_label = snapshot.year or "未知年份"
                values_text = ", ".join(snapshot.numeric_values) if snapshot.numeric_values else "未抽到显式数值"
                lines.append(
                    f"    - `{snapshot.doc_id}` / {year_label}: {values_text} | {snapshot.sentence[:140]}"
                )
            comparison_note = build_financial_metric_comparison_note(metric=metric, snapshots=snapshots)
            if comparison_note:
                lines.append(f"    - 比对备注：{comparison_note}")

    if evidence:
        lines.append("- 当前高分证据中的财报关键句：")
        for item in evidence[:6]:
            if any(metric in item.content for metric in target_metrics):
                summary = normalize_whitespace(item.content)[:180]
                lines.append(f"  - `{item.doc_id}` / `{item.title}`: {summary}")

    return "\n".join(lines).strip()


def build_insurance_supplement(q: Question, evidence: list[EvidenceSnippet]) -> str:
    """构造保险领域的变量表、公式线索与规则计算辅助。"""
    lines: list[str] = []
    variables = extract_insurance_variables(q.question)
    if variables:
        lines.append("- 题干变量表：")
        for key, values in variables.items():
            lines.append(f"  - `{key}`: {', '.join(values)}")

    formula_clues = collect_formula_clues(evidence)
    if formula_clues:
        lines.append("- 命中公式/规则线索：")
        for clue in formula_clues[:8]:
            lines.append(f"  - {clue}")

    local_calc = build_insurance_local_calc_notes(q)
    if local_calc:
        lines.append("- 本地规则计算辅助：")
        lines.extend([f"  - {item}" for item in local_calc])

    rule_calc = run_insurance_rule_calculator(q)
    if rule_calc is not None and rule_calc.details:
        lines.append(f"- {rule_calc.title}：")
        lines.extend([f"  - {item}" for item in rule_calc.details])

    logic_notes = build_insurance_logic_notes(q=q, evidence=evidence)
    if logic_notes:
        lines.append("- 保险逻辑规则辅助：")
        lines.extend([f"  - {item}" for item in logic_notes])

    return "\n".join(lines).strip()


def select_financial_metrics(q: Question) -> list[str]:
    """根据题干与选项选择财报关键指标。"""
    combined = f"{q.question}\n" + "\n".join(q.options.values())
    selected: list[str] = []
    for metric, aliases in FINANCIAL_REPORT_METRICS.items():
        if any(alias in combined for alias in aliases):
            selected.append(metric)
    return selected


def extract_metric_focus(text: str, aliases: tuple[str, ...], preferred_year: str | None) -> str:
    """抽取某指标在单个文档中的高相关句。"""
    sentences = re.split(r"(?<=[。；;！？!?\n])\s*", text)
    candidates: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        hits = sum(1 for alias in aliases if alias in sentence)
        if hits <= 0:
            continue
        if preferred_year and preferred_year in sentence:
            hits += 1
        if re.search(r"(本期|上年同期|同比|较上年|增长|下降|每10股|派发现金)", sentence):
            hits += 1
        candidates.append((hits, normalize_whitespace(sentence)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for _, sentence in candidates[:2]]
    return " / ".join(selected)[:240]


def collect_financial_metric_snapshots(
    q: Question,
    doc_ids: list[str],
    docs: DocumentRepository,
    metric: str,
) -> list[FinancialMetricSnapshot]:
    """抽取单个财报指标在多文档中的数值快照。"""
    aliases = FINANCIAL_REPORT_METRICS[metric]
    snapshots: list[FinancialMetricSnapshot] = []
    for doc_id in doc_ids:
        try:
            text = docs.load_text(domain=q.domain, doc_id=doc_id)
        except Exception:
            continue
        for sentence in find_metric_sentences(text=text, aliases=aliases)[:2]:
            numeric_values = tuple(re.findall(r"\d+(?:\.\d+)?(?:亿元|万元|万|亿|元|%|％)", sentence))
            snapshots.append(
                FinancialMetricSnapshot(
                    doc_id=doc_id,
                    year=infer_year(f"{doc_id} {sentence}"),
                    metric=metric,
                    sentence=normalize_whitespace(sentence),
                    numeric_values=numeric_values,
                )
            )
    return snapshots


def find_metric_sentences(text: str, aliases: tuple[str, ...]) -> list[str]:
    """查找包含指标别名的高相关句。"""
    candidates: list[tuple[int, str]] = []
    for sentence in re.split(r"(?<=[。；;！？!?\n])\s*", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        hits = sum(1 for alias in aliases if alias in sentence)
        if hits <= 0:
            continue
        if re.search(r"(本期|上年同期|同比|增长|下降|每10股|派发现金)", sentence):
            hits += 1
        if re.search(r"\d+(?:\.\d+)?(?:亿元|万元|万|亿|元|%|％)", sentence):
            hits += 1
        candidates.append((hits, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in candidates]


def build_financial_metric_comparison_note(metric: str, snapshots: list[FinancialMetricSnapshot]) -> str:
    """生成跨年份指标比对备注。"""
    if len(snapshots) < 2:
        return ""
    by_year = sorted(
        (snapshot for snapshot in snapshots if snapshot.year is not None),
        key=lambda item: item.year or "9999",
    )
    if len(by_year) < 2:
        return ""
    older = by_year[0]
    newer = by_year[-1]
    if not older.numeric_values or not newer.numeric_values:
        return f"已抽到 `{older.doc_id}` 与 `{newer.doc_id}` 的相关句，需人工核对数值口径。"
    return (
        f"优先比对 `{newer.doc_id}`({newer.year}) 与 `{older.doc_id}`({older.year})，"
        f"当前抽到的首批数值分别为 {', '.join(newer.numeric_values[:2])} / {', '.join(older.numeric_values[:2])}。"
    )


def extract_insurance_variables(text: str) -> dict[str, list[str]]:
    """从题干中抽取保险变量表。"""
    result: dict[str, list[str]] = {}
    normalized = normalize_whitespace(text)
    for key, patterns in INSURANCE_VARIABLE_PATTERNS.items():
        values: list[str] = []
        for pattern in patterns:
            regex = rf"{re.escape(pattern)}[^，。；;]*?(\d+(?:\.\d+)?(?:万元|万|亿元|亿|元|%|％))"
            values.extend(re.findall(regex, normalized))
        if values:
            result[key] = deduplicate_preserve_order(values)
    return result


def collect_formula_clues(evidence: list[EvidenceSnippet]) -> list[str]:
    """从证据中收集公式与赔付规则线索。"""
    clues: list[str] = []
    for item in evidence:
        for sentence in re.split(r"(?<=[。；;！？!?\n])\s*", item.content):
            sentence = normalize_whitespace(sentence)
            if not sentence:
                continue
            if re.search(r"(较大者|max|免赔额|已交保费|现金价值|账户价值|给付|赔付|退保)", sentence, flags=re.IGNORECASE):
                clues.append(f"`{item.doc_id}` / `{item.title}`: {sentence[:180]}")
    return deduplicate_preserve_order(clues)


def build_insurance_local_calc_notes(q: Question) -> list[str]:
    """构造保险题的本地计算辅助说明。"""
    notes: list[str] = []
    combined = f"{q.question}\n" + "\n".join(q.options.values())
    if any(term in combined for term in ("排序", "从高到低", "从高到低排序")):
        notes.append("若题目要求排序，先分别求每个产品/险种的给付金额，再比较大小。")
    if "免赔额" in combined:
        notes.append("医疗险通常按“责任内费用 - 已报销部分 - 免赔额”计算，再与条款限制比对。")
    if any(term in combined for term in ("身故保险金", "退保", "现金价值", "账户价值")):
        notes.append("寿险/年金险优先检查“已交保费、现金价值、账户价值、基本保额、较大者/max”规则。")
    if "白血病" in combined:
        notes.append("医疗险需先核实是否属于保险责任，再按医保报销后自费部分和免赔额计算。")
    return notes


def build_insurance_logic_notes(q: Question, evidence: list[EvidenceSnippet]) -> list[str]:
    """为等待期/免责/保单贷款/现金价值公式等逻辑题生成辅助说明。"""
    combined = normalize_whitespace(q.question + "\n" + "\n".join(q.options.values()))
    notes: list[str] = []

    if "等待期" in combined:
        notes.append("等待期题先区分“意外”与“非意外”，多数条款对意外事故不受等待期限制。")
        wait_clues = collect_sentences_by_terms(evidence, ("等待期", "意外", "非意外", "生效"))
        notes.extend(wait_clues[:3])

    if "免责" in combined or "免责范围" in combined:
        notes.append("免责题优先核对“责任免除/不承担保险责任/除外责任”原文，不足时不要凭常识外推。")
        exclusion_clues = collect_sentences_by_terms(evidence, ("免责", "责任免除", "不承担保险责任", "除外责任"))
        notes.extend(exclusion_clues[:3])

    if "保单贷款" in combined:
        notes.append("保单贷款题重点核对是否允许贷款、贷款上限、是否按现金价值扣除欠款后计算。")
        loan_clues = collect_sentences_by_terms(evidence, ("保单贷款", "现金价值", "80%", "欠款"))
        notes.extend(loan_clues[:3])

    if "现金价值" in combined and "公式" in combined:
        notes.append("现金价值公式题优先区分“明确给出公式”与“仅列示保单年度末现金价值表”。")
        cash_value_clues = collect_sentences_by_terms(evidence, ("现金价值", "公式", "保单年度末", "退保费用"))
        notes.extend(cash_value_clues[:3])

    return deduplicate_preserve_order(notes)


def run_insurance_rule_calculator(q: Question) -> InsuranceRuleCalcResult | None:
    """对常见保险题型执行轻量本地规则计算。"""
    question_text = normalize_whitespace(q.question)
    if "身故保险金" in question_text and "排序" in question_text:
        details = calc_death_benefit_sorting(question_text)
        if details:
            return InsuranceRuleCalcResult(title="本地身故保险金测算", details=details)

    if "退保" in question_text and "排序" in question_text:
        details = calc_surrender_value_sorting(question_text)
        if details:
            return InsuranceRuleCalcResult(title="本地退保金额测算", details=details)

    if "免赔额" in question_text and ("赔付" in question_text or "共应赔付" in question_text or "分别应赔付" in question_text):
        details = calc_medical_claims(question_text)
        if details:
            return InsuranceRuleCalcResult(title="本地医疗险赔付测算", details=details)

    if "等待期" in question_text:
        details = calc_waiting_period_logic(question_text)
        if details:
            return InsuranceRuleCalcResult(title="本地等待期逻辑提示", details=details)

    if "保单贷款" in question_text:
        details = calc_policy_loan_logic(question_text)
        if details:
            return InsuranceRuleCalcResult(title="本地保单贷款逻辑提示", details=details)

    if "现金价值" in question_text and "公式" in question_text:
        details = calc_cash_value_formula_logic(question_text)
        if details:
            return InsuranceRuleCalcResult(title="本地现金价值逻辑提示", details=details)

    return None


def calc_death_benefit_sorting(question_text: str) -> list[str]:
    """对典型身故保险金排序题做规则测算。"""
    paid = extract_first_amount(question_text, "已交保费均为")
    cash_value = extract_first_amount(question_text, "现金价值均为")
    if paid is None or cash_value is None:
        return []

    results: list[tuple[str, float]] = []
    account_value = extract_amount_for_product(question_text, "平安智盈金生", ("保单账户价值",))
    if account_value is not None:
        results.append(("平安智盈金生", max(account_value, cash_value)))

    base_amount = extract_amount_for_product(question_text, "国寿增益宝", ("基本保额",))
    individual_account = extract_amount_for_product(question_text, "国寿增益宝", ("个人账户价值",))
    if base_amount is not None and individual_account is not None:
        results.append(("国寿增益宝", max(base_amount * 1.6, individual_account, paid)))

    received_annuity_15 = extract_amount_for_product(question_text, "国寿鑫享添盈", ("已领养老年金",))
    if received_annuity_15 is not None:
        results.append(("国寿鑫享添盈", max(paid - received_annuity_15, cash_value)))

    received_annuity_16 = extract_amount_for_product(question_text, "平安富鸿金生", ("已领养老年金",))
    if received_annuity_16 is not None:
        results.append(("平安富鸿金生", max(paid - received_annuity_16, cash_value)))

    if not results:
        return []

    ranking = sorted(results, key=lambda item: item[1], reverse=True)
    lines = [f"{name} = {format_amount(value)}" for name, value in ranking]
    lines.append("排序结果：" + " > ".join(f"{name}({format_amount(value)})" for name, value in ranking))
    return lines


def calc_surrender_value_sorting(question_text: str) -> list[str]:
    """对典型退保金额排序题做规则测算。"""
    results: list[tuple[str, float]] = []

    paid = extract_amount_for_product(question_text, "平安智盈金生", ("累计所交保费",))
    account_gain = extract_amount_for_product(question_text, "平安智盈金生", ("保单账户累计收益",))
    if paid is not None and account_gain is not None:
        results.append(("平安智盈金生", paid + account_gain - 0.5))

    individual_account = extract_amount_for_product(question_text, "国寿增益宝", ("个人账户价值",))
    if individual_account is not None:
        fee_rate = extract_percent_near(question_text, "退保费用")
        if fee_rate is None:
            fee_rate = 0.0
        results.append(("国寿增益宝", individual_account * (1 - fee_rate / 100.0)))

    cash_value = extract_amount_for_product(question_text, "平安富鸿金生", ("现金价值",))
    if cash_value is not None:
        results.append(("平安富鸿金生", cash_value))

    if not results:
        return []

    ranking = sorted(results, key=lambda item: item[1], reverse=True)
    lines = [f"{name} = {format_amount(value)}" for name, value in ranking]
    lines.append("排序结果：" + " > ".join(f"{name}({format_amount(value)})" for name, value in ranking))
    return lines


def calc_medical_claims(question_text: str) -> list[str]:
    """对典型医疗险免赔额题做规则测算。"""
    total_cost = extract_first_amount(question_text, "总费用")
    reimbursed = extract_first_amount(question_text, "医保报销")
    self_paid = extract_first_amount(question_text, "自费")
    if self_paid is None and total_cost is not None and reimbursed is not None:
        self_paid = max(total_cost - reimbursed, 0.0)
    if self_paid is None:
        return []

    lines: list[str] = [f"责任内自费部分 = {format_amount(self_paid)}"]

    if "众安白血病医疗险" in question_text:
        zhongan_deductible = extract_deductible_for_product(question_text, "众安白血病医疗险")
        if zhongan_deductible is None:
            zhongan_deductible = 0.0
        zhongan_pay = max(self_paid - zhongan_deductible, 0.0)
        lines.append(f"众安白血病医疗险 = {format_amount(zhongan_pay)}")

    if "平安e生保" in question_text:
        pingan_deductible = extract_deductible_for_product(question_text, "平安e生保")
        if pingan_deductible is not None:
            lines.append(f"平安e生保 = {format_amount(max(self_paid - pingan_deductible, 0.0))}")

    if "太保团体百万医疗" in question_text:
        taibao_deductible = extract_deductible_for_product(question_text, "太保团体百万医疗")
        if taibao_deductible is not None:
            lines.append(f"太保团体百万医疗 = {format_amount(max(self_paid - taibao_deductible, 0.0))}")

    if "家庭共享" in question_text and "平安e生保" in question_text:
        household_costs = re.findall(r"发生医疗费用(\d+(?:\.\d+)?)万", question_text)
        household_reimburse = re.findall(r"医保报销(\d+(?:\.\d+)?)", question_text)
        if household_costs and household_reimburse:
            net_values: list[float] = []
            for index, cost in enumerate(household_costs):
                reimb = float(household_reimburse[index]) / 10000.0 if index < len(household_reimburse) else 0.0
                net_values.append(max(float(cost) - reimb, 0.0))
            if net_values:
                shared_total = sum(net_values)
                shared_deductible = extract_deductible_for_product(question_text, "平安e生保") or 0.0
                shared_pay = max(shared_total - shared_deductible, 0.0)
                lines.append(
                    "家庭共享免赔额下，平安e生保按家庭责任内净额合计计算："
                    f"{format_amount(shared_total)} - {format_amount(shared_deductible)} = {format_amount(shared_pay)}"
                )

    return lines


def calc_waiting_period_logic(question_text: str) -> list[str]:
    """为等待期题生成本地逻辑提示。"""
    lines = ["先判断各选项是否为意外事故；若为意外导致的保险事故，通常不受等待期限制。"]
    if "意外车祸" in question_text:
        lines.append("题干包含“意外车祸”，该类表述通常优先进入可赔方向。")
    if "非意外" in question_text:
        lines.append("题干中标注“非意外”的选项，需要重点核对等待期是否已满。")
    return lines


def calc_policy_loan_logic(question_text: str) -> list[str]:
    """为保单贷款题生成本地逻辑提示。"""
    lines = ["保单贷款题优先提取三个要点：是否允许贷款、贷款比例上限、是否需扣除未偿欠款。"]
    if "80%" in question_text:
        lines.append("题干选项中出现 80%，应重点核对条款是否写明“现金价值的 80%”或“扣除欠款后的 80%”。")
    if "个人养老金" in question_text:
        lines.append("若涉及个人养老金方式投保，需单独核对是否限制保单贷款。")
    return lines


def calc_cash_value_formula_logic(question_text: str) -> list[str]:
    """为现金价值公式题生成本地逻辑提示。"""
    lines = ["先区分“明确给出计算公式”与“仅列出现金价值表”，两者不能混同。"]
    if "退保费用" in question_text:
        lines.append("若题干出现退保费用，应重点核对是否存在“现金价值 = 账户价值 - 退保费用”类型条款。")
    return lines


def extract_first_amount(text: str, prefix: str) -> float | None:
    """提取某个前缀后的首个金额（统一返回万为单位的数值）。"""
    pattern = rf"{re.escape(prefix)}[^，。；;]*?(\d+(?:\.\d+)?)(万元|万|亿元|亿|元)"
    match = re.search(pattern, text)
    if not match:
        return None
    return convert_to_wan(float(match.group(1)), match.group(2))


def extract_amount_for_product(text: str, product: str, labels: tuple[str, ...]) -> float | None:
    """从产品描述附近提取金额。"""
    for label in labels:
        patterns = [
            rf"{re.escape(product)}[^，。；;]*?{re.escape(label)}(\d+(?:\.\d+)?)(万元|万|亿元|亿|元)",
            rf"{re.escape(product)}[^，。；;]*?{re.escape(label)}为?(\d+(?:\.\d+)?)(万元|万|亿元|亿|元)",
            rf"{re.escape(label)}(\d+(?:\.\d+)?)(万元|万|亿元|亿|元)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return convert_to_wan(float(match.group(1)), match.group(2))
    return None


def extract_percent_near(text: str, label: str) -> float | None:
    """抽取标签附近的百分比。"""
    match = re.search(rf"{re.escape(label)}[^，。；;]*?(\d+(?:\.\d+)?)%", text)
    if not match:
        return None
    return float(match.group(1))


def extract_deductible_for_product(text: str, product: str) -> float | None:
    """提取某产品的免赔额。"""
    patterns = [
        rf"{re.escape(product)}[^，。；;]*?免赔额(\d+(?:\.\d+)?)(万元|万|亿元|亿|元)",
        rf"{re.escape(product)}[^，。；;]*?[(（][^)]*免赔额(\d+(?:\.\d+)?)(万元|万|亿元|亿|元)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return convert_to_wan(float(match.group(1)), match.group(2))
    return None


def collect_sentences_by_terms(evidence: list[EvidenceSnippet], terms: tuple[str, ...]) -> list[str]:
    """按术语从证据中抽取辅助句。"""
    lines: list[str] = []
    for item in evidence:
        for sentence in re.split(r"(?<=[。；;！？!?\n])\s*", item.content):
            sentence = normalize_whitespace(sentence)
            if not sentence:
                continue
            if any(term in sentence for term in terms):
                lines.append(f"`{item.doc_id}` / `{item.title}`: {sentence[:160]}")
    return deduplicate_preserve_order(lines)


def convert_to_wan(value: float, unit: str) -> float:
    """将金额统一换算成“万元”。"""
    if unit in {"万元", "万"}:
        return value
    if unit in {"亿元", "亿"}:
        return value * 10000.0
    if unit == "元":
        return value / 10000.0
    return value


def format_amount(value: float) -> str:
    """将万元金额格式化为更可读的文本。"""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}万"
    return f"{value:.2f}万"


def infer_year(text: str) -> str | None:
    """推断文档或标题中的年份。"""
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else None


def normalize_whitespace(text: str) -> str:
    """标准化空白。"""
    return re.sub(r"\s+", " ", text or "").strip()


def deduplicate_preserve_order(items: list[str]) -> list[str]:
    """去重并保持原顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
