# DEWO

面向 **动态模型中心（Hugging Face Model Hub）** 与 **真实推理服务** 的 LLM 智能体系统 **DEWO**（*An LLM-Based Agent System for Dynamic Model Hubs and Real-World Inference Services*）：将自然语言指令与多模态输入编排为可执行的 DAG 工作流，在节点级联合考虑任务对齐与服务可执行性，并通过节点级恢复与图级修复提升端到端成功率。

本仓库为 **论文配套的开发与评测布局**：核心实现在 **`DEWO-code`**，基准数据集在 **`DEWO-Set`**。

<p align="center">
  <a href="https://github.com/DEWO-code/DEWO">GitHub</a>
</p>

---

## ✨ 特性

- **绑定待定的工作流**：先产出 DAG 结构与任务语义，再在线检索与绑定具体模型。
- **节点级选模**：结合可执行性、语义对齐、稳定性与活跃度等信号排序候选。
- **分层恢复**：节点内重试 / 修参 / 换模；必要时图级补丁与受影响子图增量重跑。
- **评测基准 DEWO-Set**：306 条样本，覆盖 Single / Chain / Graph 与 21 类下游任务。

---

## 📁 仓库结构

```text
DEWO/
├── DEWO-code/          # 系统实现（LangGraph 主图、Runner、HF 工具封装等）
│   ├── run.py          # CLI 入口
│   ├── requirements.txt # pip 依赖（与常用 conda 环境版本对齐，见文件头注释）
│   ├── app/            # configs、agent 模块、dewo_logging、tool_hf 等
│   └── README.md       # 架构说明、依赖、调试与环境变量
├── DEWO-demo-web/      # 可选：浏览器流式演示（FastAPI + 静态页，见该目录 README）
├── DEWO-Set/           # 数据集（JSONL + assets）
│   ├── demo_data.jsonl # 展示用示例数据（少量样例；完整评测见 datasets/）
│   ├── datasets/       # single / chain / graph（306 条）
│   ├── assets/         # 多模态文件资源
│   └── README.md       # 规模统计、字段说明与论文指标对齐说明
└── README.md           # 本文件
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/DEWO-code/DEWO.git
cd DEWO
```

### 2. 环境与依赖

推荐使用 **Python 3.12**。进入 **`DEWO-code`** 后安装依赖：

```bash
cd DEWO-code
python -m pip install -U pip
pip install -r requirements.txt
```
### 3. 配置密钥与端点

**（1）控制器 LLM（LiteLLM）**：请在 [`DEWO-code/app/configs.py`](DEWO-code/app/configs.py)进行控制器LLM的相关配置，API Key 通过环境变量读取。

**（2）Hugging Face**：Hub / 推理 token 请自行配置环境变量 **`HF_TOKEN`**。

### 4. 多模态文件资源路径

默认将 **`DEWO-Set/assets`** 作为多模态输入根目录（由 `configs.input_assets_base_dir` 解析，与 **`DEWO-code` 同级** 的目录布局）。

### 5. 运行评测

```bash
cd DEWO-code
python run.py --data ../DEWO-Set/datasets/single.jsonl --max-samples 1
```

---

## 🎬 功能演示（示例数据）

使用 **`DEWO-Set/demo_data.jsonl`** 中的内置样例快速跑通流水线（需已完成上文环境与 **`HF_TOKEN`** 等配置）。`--start-index` 为 **1-based**，与 `--max-samples` 配合可只执行其中一条。

### 示例数据 1（`req_graph_000011` · 音频转写 + 实体 + 与笔记语义对齐）

```bash
cd DEWO-code
python run.py --data ../DEWO-Set/demo_data.jsonl --start-index 1 --max-samples 1
```

> 动态演示图（示例数据 1）：*[待补充 — 可替换为 GIF / 视频或 `![](相对路径)`]*

<!-- 发布时可将上一行替换为图片，例如: ![示例数据1](docs/demo/demo-01.gif) -->

### 示例数据 2（`req_graph_000021` · 图像场景描述 + 目标检测 + 结构化报告）

```bash
cd DEWO-code
python run.py --data ../DEWO-Set/demo_data.jsonl --start-index 2 --max-samples 1
```

> 动态演示图（示例数据 2）：*[待补充 — 可替换为 GIF / 视频或 `![](相对路径)`]*

<!-- 发布时可将上一行替换为图片，例如: ![示例数据2](docs/demo/demo-02.gif) -->

### 浏览器交互演示（可选）

在配置好控制器 LLM 与 `HF_TOKEN` 后，可在本地启动 **`DEWO-demo-web`**，在页面中**选择示例并流式查看**与终端一致的 `run.py` 输出。详见 [**`DEWO-demo-web/README.md`**](DEWO-demo-web/README.md)。

---

## 📊 数据集（DEWO-Set）

| 划分 | 文件 | 样本数 |
|------|------|--------|
| Single | `datasets/single.jsonl` | 126 |
| Chain | `datasets/chain.jsonl` | 90 |
| Graph | `datasets/graph.jsonl` | 90 |
| **合计** | — | **306** |

更多统计与 JSONL 字段说明见 [**`DEWO-Set/README.md`**](DEWO-Set/README.md)。  
**`DEWO-Set/demo_data.jsonl`** 为用于展示的少量示例数据，格式与正式集一致；完整 306 条任务请使用上表中的 `datasets/*.jsonl`。

## 📜 引用

若在研究中使用了本仓库，请引用论文正式版本（题目见上文）。录用信息与 BibTeX 以 Camera-ready 为准。

---

## 📄 许可证

<!-- 发布前请替换为实际许可证，例如 MIT / Apache-2.0 -->
待定（请在仓库根目录添加 `LICENSE` 并更新本节）。

---

## 🙏 致谢

任务依赖结构的建模思路受 TaskBench 等相关工作启发；实验与推理接口基于 Hugging Face 生态。
