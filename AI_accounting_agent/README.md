# AI 记账与投资助手

本项目在本地环境中同时提供 **AI 记账服务** 与 **AI 投资助手** 两条能力线：
- 票据 / 自然语言入账 → 结构化账单数据库 → 财务报告、行为分析与建议；
- 多 Agent 投资研究链路（Analyst / Risk / Trader），提供自选股日报、交易建议、智能调仓和移动端仪表盘。

## 功能速览
- **本地部署**：所有模型、数据与 API 全程离线运行，可按需切换 `QWEN_MODEL_DIR`。
- **会计代理**：`DEFAULT_SYSTEM_PROMPT`（`prompts.py`）注入 `/chat`，强制输出简体中文且结构化 JSON 合规。
- **投资代理**：基于 `investment_app.py` 提供 Watchlist、个股日报、调仓建议与通用投顾对话。
- **前端体验**：`frontend/accounting` 与 `frontend/investment` 均为移动优先的 Tailwind 静态页面，可直接通过静态服务器访问。

## 目录总览
```
AI_accounting_agent/
├── backend/
│   ├── app.py                 # 记账服务 FastAPI
│   ├── investment_app.py      # 投资服务 FastAPI
│   ├── start.sh               # 8010 启动脚本
│   └── start_investment.sh    # 8020 启动脚本
├── frontend/
│   ├── accounting/            # 账务可视化、建议等页面
│   └── investment/            # 开始/风险/智能投资/问答/仪表盘
└── README.md
```

## 运行方式

### 会计服务（端口 8010）
```bash
cd /home/bld/data/data4/admin/fintechcom/AI_accounting_agent/backend
./start.sh
```
脚本会自动释放 8010 端口、激活 `.venv`、设置 `QWEN_MODEL_DIR=/home/bld/data/data4/admin/fintechcom/Qwen2.5-VL-3B-Instruct/model`，并运行 `app.py`。

**手动步骤**：
```bash
cd /home/bld/data/data4/admin/fintechcom/AI_accounting_agent/backend
lsof -ti:8010 | xargs -r kill -9
source .venv/bin/activate
export QWEN_MODEL_DIR="/home/bld/data/data4/admin/fintechcom/Qwen2.5-VL-3B-Instruct/model"
export PORT=8010
python app.py
```

### 投资服务（端口 8020）
```bash
cd /home/bld/data/data4/admin/fintechcom/AI_accounting_agent/backend
./start_investment.sh
```
脚本会终止 8020 端口占用，启用 `.venv`，设置 `QWEN_MODEL_DIR=/home/bld/data/data4/admin/fintechcom/Qwen3-4B`，并运行 `investment_app.py`。

**手动步骤**：
```bash
cd /home/bld/data/data4/admin/fintechcom/AI_accounting_agent/backend
lsof -ti:8020 | xargs -r kill -9
source .venv/bin/activate
export QWEN_MODEL_DIR="/home/bld/data/data4/admin/fintechcom/Qwen3-4B"
export PORT=8020
python investment_app.py
```

### 静态前端
```bash
cd /home/bld/data/data4/admin/fintechcom/AI_accounting_agent/frontend
fuser -k 5500/tcp
python3 -m http.server 5500 --bind 0.0.0.0
```
浏览器访问 `http://<HOST>:5500` 并打开对应页面（示例见“前端体验”章节）。

## 接口说明

### 会计服务 REST（`http://HOST:8010`）
- `GET /health`：返回模型设备、账单数量、数据库路径。
- `GET /bills`：支持 `limit`、`category`、`type`、`start_date`、`end_date` 等过滤。
- `POST /bills`：接收 OCR / 聊天结果写入账本。
- `GET /bills/summary`：最近账单概览。
- `POST /chat`：`system_prompt` 可覆盖，`include_bill_context` 默认注入最新 20 条账单。
- `GET /reports/aggregate`：返回周 / 月 / 年收入支出柱状图、净储蓄对比、分类对比、饼图及 `custom` 报表。
- `GET /advice/context`：输出 `overview`（30 天）、`behavior`（180 天）、`advice`（90 天）数据，前端可结合 `/chat` 生成自然语言洞察。

**示例：**
```json
{
  "messages": [
    {"role": "user", "content": "总结最近的餐饮支出"}
  ],
  "max_new_tokens": 256
}
```

### 投资服务 REST（`http://HOST:8020`）
- `GET /health`：服务状态。
- `GET /trader/watchlist`：返回 A10 自选股元数据。
- `GET /trader/portfolio`：读取 `data/portfolio_snapshot.json`，输出资产概览、持仓、行业分布与 NAV 历史。
- `GET /trader/stock/{code}/kline?days=10`：最近 K 线。
- `GET /trader/stock/{code}/news`：最新新闻与研报摘要。
- `POST /trader/stock/{code}/daily_report`：基于 K 线与资讯生成“基本面 / 社会舆情 / 风险提示”三段式日报（中文、Markdown）。
- `POST /trader/analyst/report`：以整体组合为输入生成 Analyst 报告。
- `POST /chat`：通用金融对话，支持自定义 `system_prompt`、`max_new_tokens`、`temperature`、`top_p`。

所有响应都会移除 `<think>`、“思考过程”等隐私内容，并保持正式 Markdown 样式。

## 前端体验

> 所有页面默认读取 `localStorage` 中的 API 地址：
> - `AI_BOOKKEEPER_API` → 会计后端（8010）
> - `AI_TRADER_API` → 投资后端（8020）

### 会计前端（`frontend/accounting`）
- `visualization/code.html`：按周 / 月 / 年渲染柱状图、净储蓄对比与饼图。
- `financial_advice/code.html`：加载 `/advice/context` 并通过 `/chat` 给出 AI 总览、行为分析、财务建议。
- `transaction_list/code.html`：账单列表、筛选、AI 总结。
- `bill_auto-scan_&_entry_1`、`welcome_&_setup_4` 等页面默认调用 `/chat`→`/bills` 完成票据自动入账。

### 投资前端（`frontend/investment`）
- `welcome/code.html`：移动端入口，展示“一键风险画像 / 智能投资研究 / 调仓与仪表盘”三大卖点，CTA 跳转智能投资。
- `risk/code.html`：风险评估动画页，与后端账号画像打通（可自行接入 API）。
- `report/code.html`：核心 Analyst 报告 + Trader 建议 + Smart Rebalance，需用户选择个股并手动点击按钮触发生成。
- `trader_chat/code.html`：金融问答界面，可绑定 `/chat` 进行自由提问。
- `visual/code.html`：组合仪表盘、情绪卡片、收益折线，适配移动端竖屏。
所有页面均采用统一底部导航（开始 / 风险评估 / 智能投资 / 金融问答 / 仪表盘），方便在手机中切换。

## 数据与环境变量
- `AI_BOOKKEEPER_DATA_DIR`：会计数据库目录（默认 `backend/data/bills.db`）。
- `AI_TRADER_API`：前端指向投资后端的地址，缺省为 `http://localhost:8020`。
- `QWEN_MODEL_DIR`：两个服务分别在启动脚本中指定，可根据显存替换为其他模型目录。
- `PORT`：允许在启动脚本或手动模式下覆盖。

## 常见操作
- **健康检查**：`curl http://localhost:8010/health` 与 `curl http://localhost:8020/health`。
- **日志查看**：前台运行直接查看终端输出；后台部署可使用 `nohup python app.py > server.log 2>&1 &` 与 `tail -f server.log`。
- **前端调试**：`localStorage.setItem('AI_BOOKKEEPER_API','http://HOST:8010')`、`localStorage.setItem('AI_TRADER_API','http://HOST:8020')` 后刷新页面即可。

