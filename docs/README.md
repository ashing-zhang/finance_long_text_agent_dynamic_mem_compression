# fin_agent 运行指南

## 1. 前置准备

### 1.1 配置 API Key

本项目通过 OpenAI-compatible Chat Completions 接口调用 Qwen（DashScope / Model Studio）。

需要设置环境变量：

- `DASHSCOPE_API_KEY`：你的 DashScope API Key

也可以在项目根目录创建 `.env` 文件并写入 `DASHSCOPE_API_KEY=...`，程序启动时会自动加载该文件（不覆盖已存在的环境变量）。

### 1.2 可选依赖：PDF 文本抽取

`fin_agent` 读取 PDF 时需要 `pypdf` 或 `PyPDF2`（二选一）。如果你主要跑 `regulatory`（监管法规）领域，优先读取已提供的 `.txt`，通常不依赖 PDF 抽取。

建议安装：

```bash
pip install pypdf
```

## 2. 配置说明（configs/agent.toml）

默认配置文件位于 [configs/agent.toml](file:///d:/codes/finance_long_text_agent_dynamic_mem_compression/configs/agent.toml)。

关键配置项：

- `[run]`
  - `dataset_root`：数据根目录（默认指向 `data/public_dataset_a/public_dataset_upload`）
  - `questions_subdir`：题目目录（默认 `questions/group_a`）
  - `raw_docs_subdir`：原始文档目录（默认 `raw`）
  - `split`：题组（`A` / `B`）
  - `output_csv`：输出的 `answer.csv` 路径
  - `output_evidence_jsonl`：证据输出 jsonl 路径；为空则不输出
- `[llm]`
  - `base_url`：默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `model`：默认 `qwen-plus`
  - `api_key_env`：读取 API Key 的环境变量名（默认 `DASHSCOPE_API_KEY`）
  - `timeout_seconds`：请求超时（秒）
  - `max_retries`：失败重试次数
- `[retrieval]`
  - `chunk_max_chars`：文档分块最大字符数
  - `top_k_chunks`：最终取用的证据块数量
  - `max_context_chars`：拼接后上下文最大字符数（超出会做抽取式压缩）
  - `per_doc_top_k`：每个文档最多贡献的证据块数量

## 3. 运行方式（模块化）

### 3.1 使用默认配置运行

在项目根目录执行：

```bash
python -m fin_agent.run
```

### 3.2 指定配置文件运行

通过环境变量指定配置文件路径：

```bash
FIN_AGENT_CONFIG=./configs/agent.toml python -m fin_agent.run
```

## 4. 输出说明

### 4.1 answer.csv

会在 `output_csv` 指定位置生成 `answer.csv`，包含：

- `summary` 行：汇总本次评测的 `prompt_tokens / completion_tokens / total_tokens`
- 每题一行：`qid / answer / prompt_tokens / completion_tokens / total_tokens`

答案格式：

- 单选题：输出一个大写字母（A/B/C/D）
- 判断题：输出一个大写字母（A 或 B）
- 多选题：输出多个大写字母，按字母顺序排列，不使用分隔符（例如 `ABC`）

### 4.2 证据（可选）

如果 `output_evidence_jsonl` 配置为非空路径，会追加写入每题的证据片段信息（`doc_id / chunk_id / score / quoted_clause`），用于审计与调试。
