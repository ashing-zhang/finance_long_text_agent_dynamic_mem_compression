# 金融长文本 Agentic RAG 技术方案（增强版）

## ------借鉴 Claude Code Agentic Search 的精确检索、上下文管理与多 Agent 架构

## 1. 方案定位

本方案针对 SPEC.md 中金融长文本问答任务设计。任务要求：

- 文档规模大，包含 PDF、法规、合同、年报、研报等长文本；
- 问题需要精确定位事实、数字、条件和例外条款；
- 禁止 embedding 模型参与正式检索；
- 推理阶段只能调用 Qwen 系列模型 API；
- 需要同时优化准确率和 Token 消耗。

因此，本方案不采用传统 Vector RAG，而采用：

> Agentic Search + Lexical Retrieval + Structured Evidence Memory + Qwen
> Reasoning Agent

架构。

Claude Code 的核心能力并非简单 RAG，而是通过 Agent Loop 持续执行：

```
Gather Context
      ↓
Take Action
      ↓
Verify Result
      ↓
Repeat
```

其工具驱动模式包括文件搜索、内容搜索、读取、执行和验证。该思想可迁移到金融文档理解场景。

***

# 2. 总体架构

```
                  Question
                     |
                     v

            Query Planning Agent

                     |
        +------------+-------------+
        |                          |
        v                          v

 Document Discovery Agent     Reasoning Agent


        |
        v

 +---------------------------+
 | Agentic Search Engine     |
 |                           |
 | File Search               |
 | Exact Search              |
 | Section Navigation        |
 | Evidence Expansion        |
 +---------------------------+

        |
        v

 +---------------------------+
 | Evidence Memory            |
 |                            |
 | Facts                     |
 | Numbers                   |
 | Conditions                |
 | Exceptions                |
 | Sources                   |
 +---------------------------+

        |
        v

          Qwen Reasoning

        |
        v

       answer.csv
```

***

# 3. Claude Code 思想映射

Claude Code 使用工具让模型主动探索代码库，而不是一次性读取整个项目。

对应关系：

Claude Code          金融 Agent

***

Repository           金融文档集合
Glob                 文档定位
Grep                 条款关键词搜索
Read                 章节读取
Context Compaction   Evidence Compression
Sub Agent            专业分析 Agent
Tool Loop            检索-验证循环

Claude Code 官方架构强调：

- Agent 根据任务决定调用哪些工具；
- 每次工具返回结果都会影响下一步行动；
- 通过持续上下文收集和验证完成任务。

***

# 4. 文档结构化层

## 4.1 PDF解析

允许使用：

- MinerU
- OCR
- Layout Parser

完成：

```
PDF

↓

Markdown

↓

Document Block
```

***

## 4.2 Block 数据模型

```json
{
"doc_id":"byd_2025",
"section":"研发投入",
"page":120,
"block_id":"b120_003",
"text":"研发费用占营业收入比例..."
}
```

Block 是 Agent 最小读取单元。

***

# 5. Agentic Search Engine

## 5.1 为什么不用 Embedding

金融任务存在：

- 数字精确匹配；
- 条款编号；
- 日期；
- 比例；
- 法律限定词。

例如：

问题：

> 等待期内发生身故是否赔付？

关键词：

```
等待期
身故
保险责任
```

比语义相似更可靠。

***

# 5.2 多级搜索体系

## Level 1：Document Search

目标：

找到相关文档。

技术：

SQLite Metadata

字段：

```
doc_id
domain
title
year
company
```

***

## Level 2：Exact Search

类似 Claude Code Grep。

技术：

- SQLite FTS5
- ripgrep
- Regex

例如：

```
search(
"现金价值"
)
```

返回：

```
doc:
insurance001

section:
第五条 退保

match:
现金价值表
```

***

## Level 3：Structure Search

金融文档天然具有结构：

```
第一章
  1.1 定义

第二章
  2.1 权利义务
```

Agent 可以：

```
open_section(
"2.1 权利义务"
)
```

***

# 6. Agent Loop设计

## Planner Agent

输入：

```
问题 + 选项
```

输出：

```json
{
"task":"compare",
"documents":[
"annual2024",
"annual2025"
],
"keywords":[
"营业收入",
"研发投入"
]
}
```

***

## Retriever Agent

负责：

- 搜索
- 阅读
- 扩展上下文

循环：

```
Search

↓

Read

↓

Need more context?

↓

Search again
```

***

## Analyst Agent

负责：

- 数值计算
- 条件判断
- 选项验证

输出：

```json
{
"A":true,
"B":false,
"C":true
}
```

***

## Reviewer Agent

检查：

1. 是否有证据支持；
2. 是否存在反例；
3. 输出格式是否符合要求。

***

# 7. Evidence Memory设计

不要保存全文，而保存：

```
Evidence
 |
 + Fact
 + Value
 + Condition
 + Source
 + Confidence
```

示例：

```json
{
"fact":
"等待期180天",

"source":
"保险合同第三章",

"confidence":
0.98
}
```

***

# 8. 动态上下文压缩

采用类似 Claude Code Context Compaction：

## 原始：

100页合同

↓

## Evidence Extraction：

20个关键事实

↓

## Working Memory：

5个相关事实

模型上下文：

```
Question

+

Evidence

+

Reasoning
```

而不是：

```
Question

+

100页PDF
```

***

# 9. Token优化策略

## 9.1 Lazy Reading

默认：

不读取全文。

流程：

```
标题

↓

命中章节

↓

上下文扩展

↓

完整条款
```

***

## 9.2 Cache

缓存：

```
doc_id

section

facts
```

多个问题共享。

***

## 9.3 Query Decomposition

复杂问题拆解：

例如：

"比较两年年报经营变化"

拆：

```
收入变化

利润变化

现金流变化

研发变化
```

***

# 10. 推荐工程实现

## Backend

Python

## Agent Framework

推荐：

LangGraph

状态：

```python
AgentState:

question

search_history

evidence

memory

answer
```

***

## Search Layer

```
SQLite

 ├── metadata table

 ├── document_blocks

 └── FTS5 index
```

***

## Model Layer

唯一推理：

Qwen API

用于：

- Planning
- Tool Selection
- Reasoning
- Verification

***

# 11. 运行流程

```
Question

↓

Planner

↓

Generate Search Plan

↓

Search Tool

↓

Read Evidence

↓

Memory Update

↓

Reasoning

↓

Reviewer

↓

Answer
```

***

# 12. 相比传统RAG优势

能力            Vector RAG   本方案

***

数字准确性      一般         强
条款定位        一般         强
无需Embedding   否           是
解释性          弱           强
Token控制       一般         强
跨文档比较      中           强

***

# 13. 最终提交架构

```
                 Qwen API

                    |
                    |

        Agent Orchestrator

                    |

  +-------------------------------+

  Search      Memory      Reason

  FTS5        Evidence    Qwen

  +-------------------------------+

                    |

              answer.csv
```

本方案满足 SPEC.md：

- 金融长文本理解；
- 多文档精确检索；
- 动态记忆压缩；
- Token成本控制；
- 仅使用Qwen推理；
- 不使用embedding检索。

