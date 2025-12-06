import csv
import json
import os
import re
import uvicorn
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_DIR = os.environ.get("QWEN_MODEL_DIR")
DATA_DIR = Path(os.environ.get("AI_BOOKKEEPER_DATA_DIR", Path(__file__).resolve().parent / "data"))
PORTFOLIO_PATH = DATA_DIR / "portfolio_snapshot.json"
A10_KLINE_DIR = DATA_DIR / "Financial" / "A10"
REPORTS_DIR = DATA_DIR / "Financial" / "reports"

# Stock metadata
STOCK_METADATA = {
    "600036": "招商银行",
    "600519": "贵州茅台", 
    "600900": "长江电力",
    "601088": "中国神华",
    "601138": "工业富联",
    "601288": "农业银行",
    "601398": "工商银行",
    "601628": "中国人寿",
    "601857": "中国石油",
    "601988": "中国银行"
}

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str


def _load_portfolio_snapshot() -> dict:
    if not PORTFOLIO_PATH.exists():
        raise HTTPException(status_code=404, detail="Portfolio data not found")
    with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_portfolio_payload(raw: dict) -> dict:
    total_balance = float(raw.get("total_balance", 0))
    day_change_pct = float(raw.get("day_change_pct", 0)) * 100
    ytd_return_pct = float(raw.get("ytd_return_pct", 0)) * 100
    last_update = raw.get("last_update") or raw.get("as_of")
    if last_update and isinstance(last_update, str) and len(last_update) == 10:
        last_update = f"{last_update}T00:00:00Z"
    summary = {
        "total_balance": total_balance,
        "day_change_pct": day_change_pct,
        "ytd_return_pct": ytd_return_pct,
        "last_update": last_update,
        "base_currency": raw.get("base_currency", "CNY"),
    }

    holdings = []
    for item in raw.get("holdings", []):
        weight_pct = float(item.get("weight_pct", 0))
        weight = weight_pct / 100
        market_value = total_balance * weight
        holdings.append({
            "code": item.get("symbol", ""),
            "name": item.get("name", ""),
            "weight": weight,
            "market_value": round(market_value, 2),
            "current_price": item.get("last_price"),
            "pnl_pct": item.get("pnl_pct", 0),
            "concept": item.get("theme", "--"),
        })
    holdings.sort(key=lambda h: h.get("weight", 0), reverse=True)

    allocation = {
        "by_sector": {
            entry.get("sector", "--"): entry.get("weight_pct", 0)
            for entry in raw.get("sector_allocation", [])
        }
    }

    return {
        "summary": summary,
        "holdings": holdings,
        "allocation": allocation,
        "nav_history": raw.get("nav_history", []),
        "insights": raw.get("insights", {}),
    }


def _read_kline_records(csv_path: Path, limit: int) -> List[Dict[str, str]]:
    if limit <= 0:
        return []

    buffer: Deque[Dict[str, str]] = deque(maxlen=limit)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            buffer.append({key: row.get(key) for key in reader.fieldnames})
    return list(buffer)

# App Setup
app = FastAPI(title="Investment Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Variables
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_if_needed():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    
    if not MODEL_DIR:
        raise RuntimeError("QWEN_MODEL_DIR environment variable not set")

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()
    print("Model loaded.")

def _sanitize_reply(text: str) -> str:
    if not text:
        return ""
    # Remove explicit <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    block_prefixes = (
        "思考",
        "推理",
        "reasoning",
        "思考过程",
        "推理过程",
        "分析过程",
    )

    blocks = re.split(r"\n\s*\n", text)
    filtered_blocks: List[str] = []
    for block in blocks:
        stripped_block = block.strip()
        if not stripped_block:
            continue
        first_line = stripped_block.splitlines()[0].strip()
        lower_first = first_line.lower()
        if any(
            first_line.startswith(prefix) or lower_first.startswith(prefix)
            for prefix in block_prefixes
        ):
            continue
        filtered_blocks.append(stripped_block)

    cleaned = "\n\n".join(filtered_blocks).strip()

    # Ensure key headings start on a new line to avoid run-on sentences
    headings = ("基本面分析", "社会舆情", "风险提示", "投资意见")
    for heading in headings:
        cleaned = re.sub(rf"(?<!\n){heading}", f"\n{heading}", cleaned)

    cleaned = re.sub(r"\*\*\s+", "**", cleaned)
    cleaned = re.sub(r"(\*\*[^*]+\*\*)(?!\n)", r"\1\n", cleaned)

    return cleaned.strip()

@app.get("/health")
def health():
    return {"status": "ok", "service": "investment-agent"}

@app.get("/trader/portfolio")
def get_portfolio():
    raw = _load_portfolio_snapshot()
    return _format_portfolio_payload(raw)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    load_model_if_needed()
    
    system_prompt = req.system_prompt or "You are a helpful financial investment assistant."
    
    messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})
        
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0
        )
        
    new_tokens = generated_ids[0, input_ids.shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return ChatResponse(reply=_sanitize_reply(reply))

@app.post("/trader/analyst/report")
def generate_analyst_report():
    if not PORTFOLIO_PATH.exists():
        raise HTTPException(status_code=404, detail="Portfolio data not found")
    
    with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
        portfolio = json.load(f)
        
    prompt = f"""
    You are a professional financial analyst agent. 
    Based on the following portfolio snapshot, please write a comprehensive daily analysis report.
    
    Portfolio Data:
    {json.dumps(portfolio, indent=2, ensure_ascii=False)}
    
    Report Structure:
    1. **Market & Portfolio Overview**: Summary of total balance, daily change, and YTD return.
    2. **Performance Drivers**: Analyze top gainers and losers.
    3. **Risk & Allocation**: Comment on sector allocation and concentration.
    
    
    Format the output in clean Markdown.
    """
    
    req = ChatRequest(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="You are an expert financial analyst.",
        max_new_tokens=1500
    )
    
    return chat(req)


@app.get("/trader/watchlist")
def get_watchlist():
    """Return A10 stock watchlist with metadata"""
    return [
        {"code": code, "name": name} 
        for code, name in STOCK_METADATA.items()
    ]


@app.get("/trader/stock/{stock_code}/kline")
def get_stock_kline(stock_code: str, days: int = 10):
    """Get recent K-line data for a stock"""
    csv_path = A10_KLINE_DIR / f"{stock_code}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"K-line data not found for {stock_code}")

    records = _read_kline_records(csv_path, max(days, 1))
    return records


@app.get("/trader/stock/{stock_code}/news")
def get_stock_news(stock_code: str):
    """Get news and research reports for a stock"""
    report_path = REPORTS_DIR / f"{stock_code}_news_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"News/report data not found for {stock_code}")
    
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract recent news and reports
    news_items = data.get("news", {}).get("data", [])[:5]  # Top 5 news
    reports = data.get("research_reports", {}).get("data", [])[:3]  # Top 3 reports
    
    return {
        "news": news_items,
        "reports": reports
    }


@app.post("/trader/stock/{stock_code}/daily_report")
def generate_stock_daily_report(stock_code: str):
    """Generate daily analysis report for a specific stock"""
    
    # Load K-line data
    try:
        kline_data = get_stock_kline(stock_code, days=10)
    except HTTPException:
        kline_data = []
    
    # Load news and reports
    try:
        news_data = get_stock_news(stock_code)
    except HTTPException:
        news_data = {"news": [], "reports": []}
    
    stock_name = STOCK_METADATA.get(stock_code, stock_code)
    
    # Build context
    kline_summary = f"Recent 10-day K-line data (latest first):\n"
    if kline_data:
        for i, row in enumerate(kline_data[-5:]):  # Last 5 days
            kline_summary += f"- {row.get('date')}: Open {row.get('open')}, High {row.get('high')}, Low {row.get('low')}, Close {row.get('close')}, Volume {row.get('volume')}\n"
    
    news_summary = "Recent News:\n"
    for item in news_data.get("news", [])[:3]:
        news_summary += f"- [{item.get('发布时间')}] {item.get('新闻标题')}\n"
    
    reports_summary = "Research Reports:\n"
    for report in news_data.get("reports", [])[:2]:
        reports_summary += f"- {report.get('报告名称')} ({report.get('机构')}, {report.get('东财评级')})\n"
    
    prompt = f"""
你是一名资深证券研究员，需要用中文输出一份结构化的个股研究日报，禁止透露任何思考过程或中间推理，只展示最终结论。

基础数据：
{kline_summary}

最新新闻：
{news_summary}

最新研报：
{reports_summary}

    撰写要求：
    1. 仅输出“**基本面分析**”“**社会舆情**”“**风险提示**”三个章节，每个章节另起一段，并给出2-3条要点，禁止包含“投资意见”或其他章节。
    2. 所有结论都需引用上述数据或事实，不可凭空臆测。
    3. 全文控制在400字以内，保持专业、凝练的中文表述，并确保章节标题前后各留一行空行。
"""

    req = ChatRequest(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="你是资深证券分析师，只输出中文正式报告，不得展现思考过程。",
        max_new_tokens=800,
        temperature=0.6
    )
    
    return chat(req)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8020"))
    uvicorn.run("investment_app:app", host="0.0.0.0", port=port, reload=False)
