# 本地 AI 记账助手服务

## 系统提示词

存放于 `prompts.py` 的 `DEFAULT_SYSTEM_PROMPT` 会作为默认 System Prompt 注入模型，可覆盖至 `/chat` 请求的 `system_prompt` 字段。

核心目标：
- 将自然语言、票据等信息抽取为结构化账单；
- 基于本地账单数据库进行分析与建议；
- 全程使用简体中文；
- 输出结构化数据时必须返回有效 JSON。

## 完整运行步骤

### 快速启动（推荐）

使用提供的启动脚本一键启动：

```bash
cd /home/bld/data/data4/admin/fintechcom/stitch_bill_auto_scan_entry/server
./start.sh
```

脚本会自动：
- 检查并停止占用 8010 端口的进程
- 激活虚拟环境
- 设置环境变量
- 启动服务

### 手动启动步骤

### 1. 停止占用端口的进程（如需要）

如果端口 8010 已被占用，先停止相关进程：

```bash
# 查找占用 8010 端口的进程
lsof -ti:8010

# 停止进程（将 PID 替换为实际进程号）
kill -9 $(lsof -ti:8010)

# 或者使用 fuser（如果可用）
fuser -k 8010/tcp
```

### 2. 启动后端服务

```bash
# 进入服务目录
cd /home/bld/data/data4/admin/fintechcom/stitch_bill_auto_scan_entry/server

# 激活虚拟环境
source .venv/bin/activate

# 设置环境变量
export QWEN_MODEL_DIR="/home/bld/data/data4/admin/fintechcom/Qwen2.5-VL-3B-Instruct/model"
export PORT=8010

# 启动服务
python app.py
```

服务启动后，你应该看到类似输出：
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8010 (Press CTRL+C to quit)
```

### 3. 验证服务运行

在另一个终端窗口测试服务：

```bash
# 健康检查
curl http://localhost:8010/health

# 或使用浏览器访问
# http://10.120.17.24:8010/health
```

### 4. 启动前端静态服务器（可选）

如果需要访问前端页面，在项目根目录启动 HTTP 服务器：

```bash
# 进入项目根目录
cd /home/bld/data/data4/admin/fintechcom/stitch_bill_auto_scan_entry/stitch_bill_auto_scan_entry
fuser -k 5500/tcp
# 启动静态服务器（端口 5500）
python3 -m http.server 5500 --bind 0.0.0.0
```

然后在浏览器访问：
- 财务报告：`http://10.120.17.24:5500/visualization/code.html`
- 财务顾问：`http://10.120.17.24:5500/financial_advice/code.html`
- 交易列表：`http://10.120.17.24:5500/transaction_list/code.html`

### 5. 后台运行（生产环境）

如果需要后台运行服务：

```bash
# 使用 nohup 后台运行
cd /home/bld/data/data4/admin/fintechcom/stitch_bill_auto_scan_entry/server
source .venv/bin/activate
export QWEN_MODEL_DIR="/home/bld/data/data4/admin/fintechcom/Qwen2.5-VL-3B-Instruct/model"
export PORT=8010
nohup python app.py > server.log 2>&1 &

# 查看日志
tail -f server.log

# 停止服务
pkill -f "python app.py"
```

默认数据目录：`server/data/bills.db`，可通过环境变量 `AI_BOOKKEEPER_DATA_DIR` 自定义。

## REST 接口

### 健康检查
```
GET /health
```
返回模型设备、账单数量、数据库路径。

### 账单管理
```
GET /bills?limit=100&category=餐饮&type=expense&start_date=2025-11-01&end_date=2025-11-07
POST /bills
GET /bills/summary
```

`POST /bills` 请求体：
```json
{
  "event_date": "2025-11-07",
  "category": "餐饮",
  "type": "expense",
  "amount": 48.00,
  "currency": "CNY",
  "description": "午餐",
  "source": "ocr",
  "metadata": {"raw": "原始文本"}
}
```

### 聊天接口
```
POST /chat
```
- `system_prompt` 可选，默认使用 `DEFAULT_SYSTEM_PROMPT`
- `include_bill_context` 默认为 `true`，会把最新 20 条账单以系统消息注入模型上下文

示例：
```json
{
  "messages": [
    {"role": "user", "content": "总结最近的餐饮支出"}
  ],
  "max_new_tokens": 256
}
```

### 财务报告接口
```
GET /reports/aggregate
```
- 数据来源：`server/data/synthetic_bank_bills.jsonl` 过去一年的账单
- 返回内容：
  - `week` / `month` / `year` 三个维度的收入、支出柱状图数据
  - 本期净储蓄与上一期对比
  - 支出类别（餐饮 / 出行 / 购物 / 生活缴费 / 娱乐 / 其他）本期与上一期对比
  - 支出占比饼图数据
- 前端可直接根据 `bars`、`net`、`categories`、`pie`、`summary` 字段渲染图表
- 自定义报表：可携带 `start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` 查询参数，接口会根据时间跨度自动决定按日 / 周 / 月聚合，并返回 `custom` 字段

### 财务建议接口
```
GET /advice/context
```
- 数据来源：`server/data/bills.db` 数据库
- 返回内容：
  - `overview`：最近30天的财务概览数据（收入、支出、净储蓄、分类变化等）
  - `behavior`：过去180天的消费行为分析数据（月度统计、风险标志、波动性等）
  - `advice`：最近90天的财务建议数据（收入/支出结构、储蓄率、压力点等）
- 前端可基于这些数据调用 `/chat` 接口生成 AI 总览、行为分析和财务建议

## 前端集成

所有页面可通过 `localStorage.setItem('AI_BOOKKEEPER_API', 'http://HOST:PORT')` 或在"设置与集成"页面中的"后端服务"卡片配置后端地址。
- `bill_auto-scan_&_entry_1`：三种录入方式会调用 `/chat` → `/bills` 自动解析并入库。
- `transaction_list/code.html`：加载 `/bills` 与 `/bills/summary`，支持搜索筛选与 AI 总结。
- `visualization/code.html`：使用 `/reports/aggregate` 渲染周 / 月 / 年财务报告及储蓄目标。
- `financial_advice/code.html`：使用 `/advice/context` 获取数据，调用 `/chat` 生成 AI 总览、行为分析和财务建议。
- `welcome_&_setup_4`：聊天页面默认自动带入账单上下文。

静态页面访问示例：
```
python3 -m http.server 5500 --bind 0.0.0.0
# Windows 浏览器访问 http://10.120.17.24:5500/welcome_%26_setup_4/code.html
```

