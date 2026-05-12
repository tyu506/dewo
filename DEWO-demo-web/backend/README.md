# DEWO 演示后端

FastAPI 服务：将浏览器请求转为 `demo-dewo-code` 的 LangGraph 主图执行，并通过 **SSE**（`text/event-stream`）推送阶段状态、DAG 节点进度与用量摘要。

## 环境

- Python 3.12 推荐（与 `demo-dewo-code` 一致）。
- 与 `demo-dewo-code/app/configs.py` 一致的控制器密钥（如 `DEEPSEEK_API_KEY`）及 Hugging Face `HF_TOKEN`（或 `HUGGING_FACE_HUB_TOKEN`）。

## 安装

在仓库根目录 `DEWO-demo-web` 下：

```powershell
pip install -r backend/requirements.txt
```

## 启动

```powershell
cd D:\Project\YTY\DEWO-TEST\DEWO-demo-web
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 路径与密钥探测 |
| GET | `/api/examples` | 示例列表：默认从 **`DEWO-Set/demo_data.jsonl` 前两条** 读取；每条可含 **`title_zh` / `description_zh` / `query_zh`**（由 `examples_loader` 与样本 id 对齐注入），供前端卡片中英切换；若 jsonl 不存在则回退 `app/examples.json` |
| POST | `/api/run/stream` | `multipart/form-data`：`query`、`inputs_json`（JSON 字符串）、以及任意文件字段（键名进入 `inputs`） |

SSE 每行一条 JSON：`{"type":"meta|phase|dag_node|done|error","data":{...}}`。

**安全**：请勿将服务绑定到公网；本机演示即可。
