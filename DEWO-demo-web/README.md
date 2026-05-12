# DEWO 演示 Web（前后端分离）

在浏览器中以**类 LLM 对话**形式提交任务：用户与助手均以**气泡**展示；一次执行中，**主流程进度、DAG 工作流、用时与 Token** 均在**同一条助手气泡**内实时更新，最终回复也在该气泡底部展示。

## 目录

| 路径 | 说明 |
|------|------|
| [demo-dewo-code/](demo-dewo-code/) | 演示用 DEWO 代码副本（与 CLI `run.py` 同源逻辑） |
| [backend/](backend/) | FastAPI + SSE |
| [frontend/](frontend/) | Vite + React 浅色界面 |

## 快速启动

1. 配置环境变量（与 `demo-dewo-code/app/configs.py` 一致）：控制器 API Key、`HF_TOKEN` 等。

2. 启动后端（端口 **8765**）：

```powershell
cd D:\Project\YTY\DEWO-TEST\DEWO-demo-web
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765
```

3. 启动前端（端口 **5173**）：

```powershell
cd D:\Project\YTY\DEWO-TEST\DEWO-demo-web\frontend
npm install
npm run dev
```

浏览器访问前端地址即可。

示例卡片数据来自 **`../DEWO-Set/demo_data.jsonl` 前两条**（`req_graph_000011` 音频+笔记对齐、`req_graph_000021` 图像描述+检测）；请确保 **`DEWO-Set/assets`** 下存在 `graph_011_audio_1.wav`、`graph_021_image_1.png` 等资源。

## 安全说明

本演示会调用真实 LLM 与 Hugging Face 推理。**不要**将后端 `0.0.0.0` 暴露到公网。
