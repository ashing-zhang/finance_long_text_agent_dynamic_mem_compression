# 技术架构：金融长文本智能体（动态压缩 + 纯词法检索）

本文档描述本项目的主要技术架构与端到端链路，用于后续按架构对实现进行有针对性的优化。

## 1. 目标与硬约束

**目标**：在尽可能低的 Token 成本下，完成金融长文档问答（单选/多选/判断），输出符合评测格式的 `answer.csv`，并具备可审计的证据链与检索/推理 trace。

**硬约束（来自赛题）**：

- 推理问答阶段只能调用 Qwen 系列模型 API（OpenAI-compatible Chat Completions）。
- 严禁使用任何 embedding 模型做检索或 rerank；检索必须走纯文本/词法路线。
- 输出格式严格：答案必须是大写字母；多选必须按字母排序、去重、无分隔符。

本项目的实现策略与上述约束对齐：**BM25 词法检索 + 结构化分块 + 两阶段检索漏斗 + 动态上下文压缩 + 严格输出护栏**。

## 2. 总体架构（数据流）

系统以“漏斗型”成本路由为核心：先用低成本的本地规则/词法检索快速缩小证据范围，再在必要时调用 Qwen 做证据精筛压缩，最后用 Qwen 输出受约束的 JSON 并做答案规范化。

```text
┌────────────────────────────────────────────────────────────────┐
│                        fin_agent.run (入口)                     │
│  - 加载 .env / 读取 configs/agent.toml / 组装各模块配置          │
└────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────────┐
│                      run_evaluation (批处理)                    │
│  - 加载 questions / 构建 DocumentRepository / 遍历题目作答       │
│  - 写 outputs/answer.csv + outputs/logs.csv (+ 可选 evidence)    │
└────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────────┐
│                   FinanceLongTextAgent.answer                   │
│  1) 构造检索计划（全局 query + 分选项 query + 特征抽取）          │
│  2) 文档级粗筛（仅 B 组：全域 doc_profile 做 BM25）               │
│  3) 选项级召回（对每个选项、每篇候选文档做 chunk BM25 + 加权）     │
│  4) 证据链拼装（按选项分组） + 领域补充（domain specialists）     │
│  5) 上下文压缩（本地截断/抽取式压缩；超阈值则 Qwen 精筛）          │
│  6) Qwen 严格推理（强约束 system + JSON 输出格式）                │
│  7) 输出护栏（解析 JSON / 回退推断 / 答案规范化与合法性校验）      │
└────────────────────────────────────────────────────────────────┘
```

## 3. 模块划分与职责（按代码结构）

### 3.1 入口与配置

- [run.py](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/fin_agent/run.py)
  - 从 `.env` 加载 `DASHSCOPE_API_KEY`（不覆盖已有环境变量）。
  - 从 `configs/agent.toml` 解析 `RunConfig / LlmConfig / RetrievalConfig`。
  - 以模块方式启动：`python -m fin_agent.run`。

### 3.2 应用层（核心链路）

- [agent.py](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/fin_agent/application/agent.py)
  - `FinanceLongTextAgent.answer()`：单题端到端主流程（检索→压缩→推理→后处理）。
  - `run_evaluation()`：批处理评测，产出 `answer.csv`、`logs.csv`，可选写 `evidence.jsonl`。
  - 关键机制：
    - **两阶段检索**：文档级粗筛（doc\_profile）→ 选项级 chunk 召回。
    - **检索加权**：BM25 基础分 + 符号化特征加权（年份/金额/条款号/关键词）+ 领域定制 boost。
    - **上下文动态压缩**：先本地压缩，超过阈值再调用 Qwen 做“证据精筛”。
    - **输出护栏**：要求模型仅输出 JSON；本地解析与答案规范化，保障格式可评测。
    - **可观测性**：写入每题 `thought/search/answer` trace 到 `logs.csv` 便于定位问题与优化。

### 3.3 领域专家（Domain Specialists）

- [domain\_specialists.py](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/fin_agent/application/domain_specialists.py)
  - 作用：把“领域定制的证据补充、变量/公式抽取、本地规则计算提示”集中管理，避免核心 Agent 膨胀。
  - 当前覆盖：
    - `financial_reports`：指标别名标准化、跨文档双向核验提示、指标句与数值快照抽取、年份比对备注。
    - `insurance`：题干变量表抽取、公式线索句抽取、本地规则计算辅助与逻辑规则提示。
  - 产物：以“补充证据段”的形式追加到最终上下文中，让模型用更低 token 成本获取结构化提示。

### 3.4 基础设施层（数据访问与文档结构化）

- [data\_access.py](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/fin_agent/infrastructure/data_access.py)
  - `QuestionRepository`：从 `questions/*.json` 加载题目。
  - `DocumentRepository`：按 `doc_id` 解析文档文本、构建结构化分块（chunk）。
  - 核心设计：
    - **结构化分块**：按标题/章节/条款等启发式 heading 规则切块，而不是固定长度硬切。
    - **元数据注入**：chunk 索引文本带 `[DocID | Title]`，提升 BM25 定位与可审计性。
    - **缓存**：文档文本与 chunks 在内存中缓存，降低重复 IO 与重复解析成本。

### 3.5 LLM 客户端

- [openai\_compatible\_client.py](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/fin_agent/infrastructure/llm/openai_compatible_client.py)
  - 通过 OpenAI-compatible `POST /chat/completions` 调用 Qwen。
  - 解析 `usage`，用于 `answer.csv` 统计 Token。
  - 内置重试与超时，避免评测过程中因偶发网络错误中断。

### 3.6 领域模型

- [models.py](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/fin_agent/domain/models.py)
  - `Question / EvidenceSnippet / TokenUsage / AnswerRecord`：贯穿全链路的数据结构。
  - `RunConfig / LlmConfig / RetrievalConfig`：配置驱动，避免参数散落。

## 4. 核心链路细节（单题）

### 4.1 检索计划（Query Plan）

**输入**：题干 + A/B/C/D 选项。\
**输出**：

- `global_query`：用于文档级粗筛（B 组未知 doc\_ids 时）。
- `option_queries`：对每个选项构造更“判别性”的查询（题干 + 选项文本 + 领域扩展词）。
- `QueryFeatures`：抽取年份、数值单位、条款编号、关键词，用于后续加权与放宽检索。

### 4.2 文档级粗筛（只在 B 组触发）

当题目未提供 `doc_ids` 时：

- 对领域下所有文档构造 `doc_profile`（`doc_id + title + outline + head window`）。
- 用 BM25 对 `doc_profile` 排序，再叠加符号化 boost，取 Top-K 作为候选文档。

该步骤的价值是把“全域文档规模”缩到“少量候选文档”，为后续 chunk 级检索节省大量成本。

### 4.3 选项级召回（Chunk Retrieval）

对每个选项、每篇候选文档：

- 加载结构化 chunks，并对 `chunk.to_index_text()` 做 BM25。
- 得分 = BM25 基础分 + 符号化 boost（年份/金额/条款号等）+ 领域定制 boost。
- 对 chunk 内容做“证据聚焦”（减少无关句），形成 `EvidenceSnippet`。
- 最终做两层限额：
  - `per_doc_top_k`：每篇文档最多贡献多少条证据；
  - `per_option_top_k`：每个选项最多保留多少条证据；
  - 最终 `top_k_chunks`：全局证据条数上限。

并支持最多 `max_routing_rounds` 轮路由：第一轮用计划 query，第二轮用“放宽 query”（relaxed）兜底，提高召回鲁棒性。

### 4.4 证据链拼装与动态压缩

证据链构建分三层：

1. **按选项分组拼装**：把证据组织成 A/B/C/D 的“证据链”，提升推理对齐度。
2. **领域补充段**：追加 `domain_specialists` 的结构化提示（变量/公式/核验顺序）。
3. **动态压缩**：
   - `len(context) <= max_context_chars`：直接使用；
   - `len(context) > refine_context_chars`：调用 Qwen 执行“证据精筛”，仅保留能直接支撑/反驳选项的证据；
   - 否则走本地抽取式压缩/截断，控制上下文大小。

### 4.5 Qwen 严格推理 + 输出护栏

推理阶段采用强约束：

- system 强制“只基于证据，不得用常识；证据不足则判错”。
- user 明确 JSON schema，要求只输出 JSON。

后处理护栏：

- 优先解析 JSON 的 `final_answer`；
- 若 `final_answer` 缺失，则根据 `option_evaluation` 中 `verdict=true` 的选项回推出答案；
- 仍失败则从原始文本中抽取字母并做规范化；
- 统一做大写、合法性过滤、多选排序去重，确保符合 `SPEC.md`。

## 5. Token 经济学（成本控制点）

本项目的 Token 消耗主要来自两次模型调用（可能只发生一次）：

- **精筛压缩调用（可选）**：仅在上下文超过 `refine_context_chars` 时触发。
- **最终推理调用（必选）**：生成受约束的 JSON 与答案。

关键控制旋钮在 `RetrievalConfig`：

- `doc_top_k`：B 组候选文档数（越小越省，但漏召回风险更高）。
- `chunk_max_chars`：chunk 粒度（过大导致噪声与 token；过小导致割裂）。
- `per_doc_top_k / per_option_top_k / top_k_chunks`：证据条数上限（直接影响上下文体积）。
- `refine_context_chars / max_context_chars`：控制是否触发精筛与最终上下文上限。
- `max_routing_rounds`：检索轮数上限（控制检索鲁棒性与计算耗时）。

同时，`DocumentRepository` 的文本与 chunk 缓存减少重复读取；`doc_profile` 使用窗口截断（`coarse_doc_window_chars`）避免全量参与粗筛。

## 6. 可优化的方向（按架构定位）

以下方向可以直接映射到架构模块与配置旋钮，便于逐项优化：

- **文档结构化（infrastructure）**
  - 改进 heading 识别规则、表格与编号条款的切分策略（提升召回精度，减少无效 chunk）。
  - 针对不同领域使用不同 chunk 粒度（如研报更大窗口、法规更强条款对齐）。
- **检索与加权（application/agent）**
  - 优化符号化 boost 的特征：金额单位一致性、比例/期限的近邻匹配、条款号的强匹配。
  - 放宽检索（relaxed queries）策略可更“可控”：只在命中失败时引入少量扩展词，避免噪声。
- **压缩与提示（application + domain\_specialists）**
  - 将压缩 prompt 进一步结构化为“保留/丢弃”的判别式输出，减少模型自由发挥。
  - 领域专家产出更规范的“可引用证据要点”，降低最终推理的 token 需求。
- **输出护栏与可观测性**
  - 对 JSON 解析失败与异常样本做聚类（基于 `logs.csv`），定位最常见失败模式。
  - 对高 token 样本做 topN 分析（结合 `context_chars`、检索 hit 分布、是否触发精筛）。

## 7. 配置与产物（工程接口）

**配置入口**：`configs/agent.toml`（也可用环境变量覆盖路径）

- `FIN_AGENT_DOTENV`：指定 `.env` 路径（默认根目录 `.env`）
- `FIN_AGENT_CONFIG`：指定配置文件路径（默认 `configs/agent.toml`）

**输出产物**（默认在 `outputs/`）：

- `answer.csv`：提交文件（含 `summary` 行与每题 token 统计）
- `logs.csv`：每题检索/推理 trace（用于审计与定位问题）
- `output_evidence_jsonl`（可选）：每题证据片段明细（便于对比不同检索策略）

