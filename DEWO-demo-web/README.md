# DEWO 演示 Web（交互式）

用于在浏览器中**流式展示** `DEWO-code/run.py` 的终端输出（模块 1→5、Binder 修参、HF 推理与 `final_result`），与你在 CLI 中看到的体验一致。

## 前置条件

- 已按主仓库说明配置 **`DEWO-code`** 依赖、**控制器 LLM**（`app/configs.py`）与 **`HF_TOKEN`**。
- 仓库目录保持 **`DEWO-code/`** 与 **`DEWO-Set/`** 与主 `README` 一致（本服务从仓库根解析路径）。

## 安装与启动

在 **`DEWO-demo-web`** 目录下（可与 `DEWO-code` 使用同一 Python 环境，或单独 venv）：

```bash
cd DEWO-demo-web
pip install -r requirements.txt
python -m uvicorn server.main:app --host 127.0.0.1 --port 8765
```

浏览器打开：**http://127.0.0.1:8765/**

- 选择 **`demo_data.jsonl`** 中的示例卡片，点击 **运行示例**。
- 右侧 **最终结果** 会尝试从日志中解析 `final_result:` 与下一段分隔线之间的文本。

## API（本地）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 检查 `run.py` 与 `demo_data.jsonl` 是否存在 |
| GET | `/api/presets` | 返回 `demo_data.jsonl` 每行摘要 |
| GET | `/api/run/stream?preset=1&max_samples=1` | SSE：流式输出子进程 stdout（`preset` 为 JSONL 行号，1-based） |

## 安全说明

本演示在本地启动子进程执行 **真实** `run.py`，会消耗 LLM 与 HF 推理额度。**勿将 `0.0.0.0` 暴露到公网**。若需对外演示，请加反向代理、鉴权与限流。
