# DEWO 演示前端

Vite + React + TypeScript，浅色主题；开发时通过 Vite 代理访问 `http://127.0.0.1:8765` 的后端 API。

## 要求

- **Node.js 18+**（推荐；Node 16 请至少使用本仓库锁定的 Vite 4.x 依赖）。

## 安装与开发

```powershell
cd D:\Project\YTY\DEWO-TEST\DEWO-demo-web\frontend
npm install
npm run dev
```

浏览器打开控制台提示的地址（一般为 `http://127.0.0.1:5173`）。请先启动后端（见 `../backend/README.md`）。

## 构建

```powershell
npm run build
```

产物在 `frontend/dist/`，可由任意静态服务器或后端挂载提供。
