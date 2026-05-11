# DEWO-Set

**DEWO-Set** 是论文 *DEWO: An LLM-Based Agent System for Dynamic Model Hubs and Real-World Inference Services* 中为 **动态模型中心（Model Hub）与真实推理服务（Inference Providers）** 场景构建的端到端评测基准。

它与系统 **DEWO** 配套：用于在统一协议下检验智能体能否在候选可用性、服务可执行性与运行时条件变化的环境中，完成从自然语言指令与多模态资源到可验收产物的闭环执行。

代码与数据发布见论文声明仓库：  
[https://github.com/DEWO-code/DEWO](https://github.com/DEWO-code/DEWO)

---

## 1. DEWO-Set与 DEWO 的配套关系

论文关注的问题是：在 **动态模型仓库** 与 **真实推理服务** 并存的环境中，候选模型是否可用、服务是否可调用、运行时状态会变化；智能体需要在 **自然语言指令** 与 **多模态文件输入** 下，规划并执行带依赖的工作流（子任务节点、数据依赖边、以及逐节点的模型与服务绑定），并尽可能提高 **端到端执行成功率**。  
**DEWO-Set** 在上述设定下提供固定 **306** 条样本，作为 **统一任务集**：各方法共享相同输入、验收检查与日志协议，在 Hugging Face 生态所代表的动态 Hub 与真实推理接口上进行公平对比。

数据集在 **任务依赖结构** 上的划分受 TaskBench 等结构化依赖建模工作启发；样本覆盖 **单节点（Single）、链式（Chain）、图式（Graph）** 三种形态，分别对应一步调用、串行依赖、以及含并行分支与汇聚的图结构。

---

## 2. 规模与结构统计（与论文 Table 1 一致）

### 2.1 按任务结构

| 类型 | 说明 | 条数 | 占比 |
|------|------|------|------|
| Single | 单步 / 单节点调用 | 126 | 41.2% |
| Chain | 多步串行依赖 | 90 | 29.4% |
| Graph | 含并行分支与依赖汇聚 | 90 | 29.4% |
| **合计** | — | **306** | **100%** |

仓库内与上表一一对应：`datasets/single.jsonl`、`datasets/chain.jsonl`、`datasets/graph.jsonl`。

### 2.2 按输入模态（论文统计；部分样本可含多模态）

| 模态 | 样本数（论文报告） |
|------|---------------------|
| Image | 107 |
| Audio | 23 |
| Table | 20 |

实际资源文件置于 `assets/`，与每条样本 `inputs` 中的引用一致；评测时由统一协议解析路径并构造多模态上下文。

### 2.3 任务类型与能力覆盖

全集共 **21** 类下游任务类型，横跨 **文本、视觉、语音、表格** 等技能，与开放模型生态中常见 pipeline 能力对齐；部分样本带 **严格输出格式** 约束，用于检验机器可解析结果与格式合规性。

---

## 3. 与论文评测指标的对应（复现实验时可参照）

论文在统一协议下报告 **Execution Success（ES）** 与 **Workflow Score（WS）**：

- **ES**：严格规则型通过——综合执行证据、非空输出、机器可解码结果、格式合规与多模态产物等；通过率定义为「通过样本数 / 总任务数」。
- **WS**：由 **LLM 裁判** 对每条轨迹从工作流连贯性、完整性（路径、跨节点传递、节点执行等）打分后在样本上取平均，越高越好。

此外论文还报告端到端 **Latency（s）** 与 **Controller LLM Tokens（k）**；复现 DEWO 与基线（如 HuggingGPT、ReAct、AutoGen、smolagents）时需与论文 **相同任务集、相同外部工具与推理接口** 对齐。

---

## 4. 目录结构

本目录为数据集根 **DEWO-Set**：

```text
DEWO-Set/
├─ assets/                 # 多模态文件资源（与 JSONL 中 inputs 引用一致）
├─ demo_data.jsonl         # 用于展示的示例数据（少量样例，非完整 306 条）
├─ datasets/               # 按结构划分的 JSONL
│  ├─ single.jsonl         # Single，difficulty: simple
│  ├─ chain.jsonl          # Chain，difficulty: normal
│  └─ graph.jsonl          # Graph，difficulty: complex
└─ README.md
```

根目录下的 **`demo_data.jsonl`** 仅含少量行，字段形态与 `datasets/*.jsonl` 一致，便于演示或快速核对 JSONL 格式；**论文对齐的完整评测** 请始终使用 `datasets/` 下三个文件（合计 **306** 条）。

---

## 5. 样本格式（JSONL）

每行一条 JSON，与 DEWO 流水线消费字段对齐的基础形态如下：

```json
{
  "id": "req_xxx",
  "split": "test",
  "difficulty": "simple|normal|complex",
  "task": ["one_or_more_supported_tasks"],
  "query": "自然语言任务描述（可含语言/格式/字段/数量等约束）",
  "inputs": {
    "image": "可选",
    "audio": "可选",
    "video": "可选",
    "table": "可选"
  },
  "expected_output_type": "text|json|image|audio|video",
  "json_format": ["仅当 expected_output_type=json 时建议提供"]
}
```

- **difficulty**：`simple` / `normal` / `complex` 分别对应 Single / Chain / Graph 三类结构。  
- **task**：节点级任务类型列表，约束执行轨迹中的能力调用，须属于论文与实现共同支持的 **21** 类之一。  
- **json_format**：在 JSON 输出场景下提供字段路径等约束，便于 ES 侧结构化验收。

---

## 6. 21类任务类型（与开放 Hub 能力对齐）

- **文本**：`text_generation`、`summarization`、`translation`、`question_answering`、`sentence_similarity`、`feature_extraction`、`fill_mask`、`token_classification`、`text_classification`、`zero_shot_classification`  
- **表格**：`table_question_answering`  
- **图文/视觉**：`image_text_to_text`、`image_classification`、`object_detection`、`image_segmentation`  
- **生成**：`text_to_image`、`image_to_image`、`text_to_video`、`image_to_video`  
- **语音**：`automatic_speech_recognition`、`text_to_speech`

---

## 7. 构造原则（与 DEWO 问题设定一致）

- **可执行优先**：子任务组合满足数据依赖与 I/O 可连通，避免仅堆叠语义却不可在真实服务上跑通。  
- **结构真实**：Chain 强调串行中间态传递；Graph 强调并行分支与汇聚，便于检验错误沿依赖传播与分层恢复。  
- **输出可验收**：在需要处给出明确类型与格式约束，支撑 **ES** 的规则检查与 **WS** 的轨迹评判。  
- **资产一致**：`inputs` 引用文件须存在于 `assets/` 且模态与任务一致。  
- **语义合理**：每个中间节点应对最终可交付结果有实质贡献。

---

## 8. 使用提示

- 按行流式读取 JSONL；`id` 建议作为日志与对论文表格的样本键。  
- 与 **DEWO** 或自建评测器对接时，将 `inputs` 中相对路径解析到 `assets/`（或配置中的资源根目录）。  
- 生成类任务建议以 **artifact 路径** 落盘，便于 ES 检查多模态产物与复现。  
- 建议完整记录 **执行轨迹**（绑定、调用参数、耗时、失败类型），以支撑 WS 与失败分析。

---

## 9. 版本与引用

- 当前基准样本集中于 `datasets/` 三文件，合计 **306** 条；若需与论文数值严格对齐，请以本 README 与论文 Table 1 为准并对仓库打 **Git 标签** 冻结快照。  
- 学术引用请使用论文正式题目与会议/期刊信息（以录用版本为准）。
