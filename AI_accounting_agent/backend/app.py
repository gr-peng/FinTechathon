import json
import os
import sqlite3
import statistics
import uvicorn
import calendar
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
except Exception:
    AutoModelForVision2Seq = None  # type: ignore
    AutoProcessor = None  # type: ignore

from prompts import DEFAULT_SYSTEM_PROMPT


MODEL_DIR = os.environ.get(
    "QWEN_MODEL_DIR",
    #"/home/bld/data/data4/admin/fintechcom/Qwen3-4B",
    #"/home/bld/data/data4/admin/fintechcom/Qwen2.5-VL-3B-Instruct",
)
DATA_DIR = Path(
    os.environ.get(
        "AI_BOOKKEEPER_DATA_DIR",
        Path(__file__).resolve().parent / "data",
    )
)
DB_PATH = DATA_DIR / "bills.db"
JSONL_PATH = DATA_DIR / "synthetic_bank_bills.jsonl"

DISPLAY_CATEGORIES = ["餐饮", "出行", "购物", "生活缴费", "娱乐", "其他"]
CATEGORY_ALIASES = {
    "餐": "餐饮",
    "饭": "餐饮",
    "外卖": "餐饮",
    "餐饮": "餐饮",
    "出行": "出行",
    "交通": "出行",
    "打车": "出行",
    "购物": "购物",
    "数码": "购物",
    "生活缴费": "生活缴费",
    "缴费": "生活缴费",
    "话费": "生活缴费",
    "水电": "生活缴费",
    "娱乐": "娱乐",
    "影视": "娱乐",
    "游戏": "娱乐",
}
WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
MONTH_LABELS = [f"{i}月" for i in range(1, 13)]

# 默认是否启用全库检索（RAG）
DEFAULT_RETRIEVAL = os.environ.get("AI_BOOKKEEPER_DEFAULT_RETRIEVAL", "true").lower() in {"1", "true", "yes", "on"}
GEN_DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("AI_BOOKKEEPER_DEFAULT_MAX_NEW_TOKENS", "1024"))
GEN_MAX_NEW_TOKENS_CAP = int(os.environ.get("AI_BOOKKEEPER_MAX_NEW_TOKENS_CAP", "4096"))


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 3000
    min_new_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    system_prompt: Optional[str] = None
    include_bill_context: bool = True
    # RAG：是否构建全库检索上下文（聚合 + 相关片段），None 表示走默认开关
    retrieval: Optional[bool] = None
    # RAG：限制聚合时间窗口的月数（例如 12 表示过去 12 个月；None 表示不限制）
    retrieval_months: Optional[int] = 12


class ChatResponse(BaseModel):
    reply: str


class BillBase(BaseModel):
    event_date: date = Field(..., description="发生日期")
    category: str = Field(..., min_length=1, max_length=64)
    type: str = Field("expense", description="expense 或 income")
    amount: float = Field(..., description="正数金额")
    currency: str = Field("CNY", min_length=1, max_length=8)
    description: str = Field(..., min_length=1, max_length=512)
    source: Optional[str] = Field(None, description="数据来源，如OCR/手动")
    metadata: Optional[dict] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        value = value.lower()
        if value not in {"expense", "income"}:
            raise ValueError("type 必须为 expense 或 income")
        return value

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: float) -> float:
        if value < 0:
            raise ValueError("amount 必须为正数")
        return round(float(value), 2)


class BillCreate(BillBase):
    pass


class Bill(BillBase):
    id: int
    created_at: datetime

    model_config = {
        "json_encoders": {
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
        }
    }


app = FastAPI(title="Local Qwen3-4B Chat Server")

DATA_DIR.mkdir(parents=True, exist_ok=True)
_db_lock = Lock()
_db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_db_conn.row_factory = sqlite3.Row

with _db_lock:
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL,
            category TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'expense',
            amount REAL NOT NULL,
            currency TEXT NOT NULL DEFAULT 'CNY',
            description TEXT NOT NULL,
            source TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _db_conn.commit()

# 从 JSONL 导入账单至 SQLite（仅当数据库为空时触发）
def _import_jsonl_if_needed() -> None:
    jsonl_path = os.environ.get("AI_BOOKKEEPER_JSONL_PATH")
    if not jsonl_path:
        return
    p = Path(jsonl_path)
    if not p.exists():
        return
    with _db_lock:
        count = _db_conn.execute("SELECT COUNT(*) AS c FROM bills").fetchone()["c"]
    if count and count > 0:
        return
    inserted = 0
    now = datetime.utcnow().isoformat()
    with p.open("r", encoding="utf-8") as f, _db_lock:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # 兼容 synthetic 数据格式：{"messages":[{"role":"assistant","content":"{...}"}]}
            payload = None
            if isinstance(obj, dict) and "messages" in obj:
                for m in obj.get("messages", []):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        try:
                            payload = json.loads(m.get("content", "{}"))
                        except Exception:
                            payload = None
                        break
            # 如果已经是结构化对象，也尝试直接使用
            if payload is None and isinstance(obj, dict):
                payload = obj
            if not isinstance(payload, dict):
                continue
            # 兼容中文键
            try:
                event_date = payload.get("日期") or payload.get("event_date")
                category = payload.get("类别") or payload.get("category") or "未分类"
                amount = payload.get("金额") or payload.get("amount")
                description = payload.get("描述") or payload.get("description") or ""
                merchant = payload.get("商户") or payload.get("merchant")
                payment = payload.get("支付方式") or payload.get("payment_method")
                if not event_date or amount is None:
                    continue
                amount = float(amount)
                metadata = {
                    "merchant": merchant,
                    "payment_method": payment,
                    "raw": payload,
                }
                metadata_json = json.dumps(metadata, ensure_ascii=False)
                _db_conn.execute(
                    """
                    INSERT INTO bills
                    (event_date, category, type, amount, currency, description, source, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_date,
                        category,
                        "expense",
                        amount,
                        "CNY",
                        description if description else (f"{merchant or ''}消费").strip(),
                        "jsonl",
                        metadata_json,
                        now,
                    ),
                )
                inserted += 1
            except Exception:
                # 单条失败忽略，继续导入其它行
                continue
        _db_conn.commit()
    # 可在启动日志上体现
    print(f"[bootstrap] Imported {inserted} bills from JSONL: {p}")


def _normalize_category(raw: Optional[str]) -> str:
    if not raw:
        return "其他"
    text = str(raw)
    for key, value in CATEGORY_ALIASES.items():
        if key in text:
            return value
    return text if text in DISPLAY_CATEGORIES else "其他"


def _extract_payload_from_jsonl(line: str) -> Optional[dict]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if isinstance(obj, dict) and "messages" in obj:
        for message in obj.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "assistant":
                try:
                    return json.loads(message.get("content", "{}"))
                except Exception:
                    return None
    if isinstance(obj, dict):
        return obj
    return None


def _load_synthetic_records() -> List[dict]:
    records: List[dict] = []
    if not JSONL_PATH.exists():
        return records
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = _extract_payload_from_jsonl(line)
            if not isinstance(payload, dict):
                continue
            date_str = payload.get("日期") or payload.get("event_date")
            amount = payload.get("金额") or payload.get("amount")
            type_str = str(payload.get("type", "expense")).lower()
            category = _normalize_category(payload.get("类别") or payload.get("category"))
            if not date_str or amount is None:
                continue
            try:
                event_date = datetime.fromisoformat(date_str).date()
            except ValueError:
                continue
            try:
                amount_value = float(amount)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "event_date": event_date,
                    "amount": amount_value,
                    "type": type_str,
                    "category": category,
                }
            )
    return records


_RAW_SYNTHETIC_RECORDS = _load_synthetic_records()
REPORT_REFERENCE_DATE = max((r["event_date"] for r in _RAW_SYNTHETIC_RECORDS), default=date.today())
SYNTHETIC_RECORDS = [
    r
    for r in _RAW_SYNTHETIC_RECORDS
    if REPORT_REFERENCE_DATE - timedelta(days=366) <= r["event_date"] <= REPORT_REFERENCE_DATE
]


def _filter_records(records: List[dict], start: date, end: date) -> List[dict]:
    return [r for r in records if start <= r["event_date"] <= end]


def _sum_income_expense(records: List[dict]) -> tuple[float, float]:
    income = sum(r["amount"] for r in records if r["type"] == "income")
    expense = sum(r["amount"] for r in records if r["type"] == "expense")
    return round(income, 2), round(expense, 2)


def _category_totals(records: List[dict]) -> dict[str, float]:
    totals = {cat: 0.0 for cat in DISPLAY_CATEGORIES}
    for r in records:
        if r["type"] != "expense":
            continue
        totals[r["category"]] = totals.get(r["category"], 0.0) + r["amount"]
    return {k: round(v, 2) for k, v in totals.items()}


def _build_category_comparison(curr_records: List[dict], prev_records: List[dict]) -> List[dict]:
    current = _category_totals(curr_records)
    previous = _category_totals(prev_records)
    items = []
    for cat in DISPLAY_CATEGORIES:
        items.append(
            {
                "name": cat,
                "current": round(current.get(cat, 0.0), 2),
                "previous": round(previous.get(cat, 0.0), 2),
            }
        )
    return items


def _build_pie_data(curr_records: List[dict]) -> List[dict]:
    totals = _category_totals(curr_records)
    data = []
    for cat, value in totals.items():
        if value > 0:
            data.append({"name": cat, "value": value})
    return data


def _build_net_summary(curr_records: List[dict], prev_records: List[dict]) -> dict:
    curr_income, curr_expense = _sum_income_expense(curr_records)
    prev_income, prev_expense = _sum_income_expense(prev_records)
    current = round(curr_income - curr_expense, 2)
    previous = round(prev_income - prev_expense, 2)
    return {
        "current": current,
        "previous": previous,
        "difference": round(current - previous, 2),
    }


def _build_summary(records: List[dict]) -> dict:
    income, expense = _sum_income_expense(records)
    return {"income_total": income, "expense_total": expense}


def _format_range(start: date, end: date) -> str:
    return f"{start.isoformat()} ~ {end.isoformat()}"


def _shift_month(base: date, months: int) -> date:
    month = base.month - 1 + months
    year = base.year + month // 12
    month = month % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _build_week_report(records: List[dict], ref_date: date) -> dict:
    current_start = ref_date - timedelta(days=ref_date.weekday())
    current_end = current_start + timedelta(days=6)
    previous_start = current_start - timedelta(days=7)
    previous_end = current_start - timedelta(days=1)

    current_records = _filter_records(records, current_start, current_end)
    previous_records = _filter_records(records, previous_start, previous_end)

    income_series = [0.0] * 7
    expense_series = [0.0] * 7
    for r in current_records:
        idx = (r["event_date"] - current_start).days
        if 0 <= idx < 7:
            if r["type"] == "income":
                income_series[idx] += r["amount"]
            else:
                expense_series[idx] += r["amount"]

    return {
        "title": "本周",
        "range": _format_range(current_start, current_end),
        "bars": {
            "labels": WEEKDAY_LABELS,
            "income": [round(v, 2) for v in income_series],
            "expense": [round(v, 2) for v in expense_series],
            "unit": "天",
        },
        "net": _build_net_summary(current_records, previous_records),
        "categories": _build_category_comparison(current_records, previous_records),
        "pie": _build_pie_data(current_records),
        "summary": _build_summary(current_records),
    }


def _build_month_report(records: List[dict], ref_date: date) -> dict:
    month_start = ref_date.replace(day=1)
    next_month_start = _shift_month(month_start, 1)
    month_end = next_month_start - timedelta(days=1)
    previous_month_start = _shift_month(month_start, -1)
    previous_month_end = month_start - timedelta(days=1)

    current_records = _filter_records(records, month_start, month_end)
    previous_records = _filter_records(records, previous_month_start, previous_month_end)

    labels: List[str] = []
    income_series: List[float] = []
    expense_series: List[float] = []

    segment_start = month_start
    week_index = 1
    while segment_start <= month_end:
        segment_end = min(segment_start + timedelta(days=6), month_end)
        segment_records = _filter_records(records, segment_start, segment_end)
        income, expense = _sum_income_expense(segment_records)
        labels.append(f"第{week_index}周")
        income_series.append(income)
        expense_series.append(expense)
        week_index += 1
        segment_start = segment_end + timedelta(days=1)

    return {
        "title": "本月",
        "range": _format_range(month_start, month_end),
        "bars": {
            "labels": labels,
            "income": [round(v, 2) for v in income_series],
            "expense": [round(v, 2) for v in expense_series],
            "unit": "周",
        },
        "net": _build_net_summary(current_records, previous_records),
        "categories": _build_category_comparison(current_records, previous_records),
        "pie": _build_pie_data(current_records),
        "summary": _build_summary(current_records),
    }


def _build_year_report(records: List[dict], ref_date: date) -> dict:
    year_start = date(ref_date.year, 1, 1)
    next_year_start = date(ref_date.year + 1, 1, 1)
    year_end = next_year_start - timedelta(days=1)
    previous_year_start = date(ref_date.year - 1, 1, 1)
    previous_year_end = year_start - timedelta(days=1)

    current_records = _filter_records(records, year_start, year_end)
    previous_records = _filter_records(records, previous_year_start, previous_year_end)

    income_series = []
    expense_series = []
    for month in range(1, 13):
        month_start = date(ref_date.year, month, 1)
        month_end = _shift_month(month_start, 1) - timedelta(days=1)
        month_records = _filter_records(records, month_start, month_end)
        income, expense = _sum_income_expense(month_records)
        income_series.append(income)
        expense_series.append(expense)

    return {
        "title": "本年",
        "range": _format_range(year_start, year_end),
        "bars": {
            "labels": MONTH_LABELS,
            "income": [round(v, 2) for v in income_series],
            "expense": [round(v, 2) for v in expense_series],
            "unit": "月",
        },
        "net": _build_net_summary(current_records, previous_records),
        "categories": _build_category_comparison(current_records, previous_records),
        "pie": _build_pie_data(current_records),
        "summary": _build_summary(current_records),
    }


def _build_custom_report(records: List[dict], start: date, end: date) -> dict:
    if start > end:
        start, end = end, start
    span_days = (end - start).days + 1
    previous_end = start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=span_days - 1)

    current_records = _filter_records(records, start, end)
    previous_records = _filter_records(records, previous_start, previous_end)

    if span_days <= 31:
        unit = "天"
        labels = []
        income_series = []
        expense_series = []
        day = start
        while day <= end:
            day_records = _filter_records(records, day, day)
            income, expense = _sum_income_expense(day_records)
            labels.append(day.strftime("%m-%d"))
            income_series.append(income)
            expense_series.append(expense)
            day += timedelta(days=1)
    elif span_days <= 180:
        unit = "周"
        labels = []
        income_series = []
        expense_series = []
        segment_start = start
        index = 1
        while segment_start <= end:
            segment_end = min(segment_start + timedelta(days=6), end)
            segment_records = _filter_records(records, segment_start, segment_end)
            income, expense = _sum_income_expense(segment_records)
            labels.append(f"第{index}周")
            income_series.append(income)
            expense_series.append(expense)
            index += 1
            segment_start = segment_end + timedelta(days=1)
    else:
        unit = "月"
        labels = []
        income_series = []
        expense_series = []
        month_cursor = start.replace(day=1)
        while month_cursor <= end:
            month_end = _shift_month(month_cursor, 1) - timedelta(days=1)
            if month_end < start:
                month_cursor = _shift_month(month_cursor, 1)
                continue
            if month_cursor > end:
                break
            segment_start = max(month_cursor, start)
            segment_end = min(month_end, end)
            segment_records = _filter_records(records, segment_start, segment_end)
            income, expense = _sum_income_expense(segment_records)
            labels.append(month_cursor.strftime("%Y-%m"))
            income_series.append(income)
            expense_series.append(expense)
            month_cursor = _shift_month(month_cursor, 1)

    return {
        "title": "自定义区间",
        "range": _format_range(start, end),
        "bars": {
            "labels": labels,
            "income": [round(v, 2) for v in income_series],
            "expense": [round(v, 2) for v in expense_series],
            "unit": unit,
        },
        "net": _build_net_summary(current_records, previous_records),
        "categories": _build_category_comparison(current_records, previous_records),
        "pie": _build_pie_data(current_records),
        "summary": _build_summary(current_records),
    }


def _empty_report(unit: str) -> dict:
    return {
        "title": "",
        "range": "",
        "bars": {"labels": [], "income": [], "expense": [], "unit": unit},
        "net": {"current": 0.0, "previous": 0.0, "difference": 0.0},
        "categories": [
            {"name": cat, "current": 0.0, "previous": 0.0} for cat in DISPLAY_CATEGORIES
        ],
        "pie": [],
        "summary": {"income_total": 0.0, "expense_total": 0.0},
    }


@lru_cache(maxsize=1)
def build_report_payload() -> dict:
    if not SYNTHETIC_RECORDS:
        return {
            "week": _empty_report("天"),
            "month": _empty_report("周"),
            "year": _empty_report("月"),
        }
    reference = max((r["event_date"] for r in SYNTHETIC_RECORDS), default=REPORT_REFERENCE_DATE)
    return {
        "week": _build_week_report(SYNTHETIC_RECORDS, reference),
        "month": _build_month_report(SYNTHETIC_RECORDS, reference),
        "year": _build_year_report(SYNTHETIC_RECORDS, reference),
    }


def _fetch_db_records(start: date, end: date) -> List[dict]:
    with _db_lock:
        rows = _db_conn.execute(
            """
            SELECT event_date, category, type, amount, description
            FROM bills
            WHERE date(event_date) BETWEEN date(?) AND date(?)
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    records: List[dict] = []
    for row in rows:
        try:
            event_date = datetime.fromisoformat(row["event_date"]).date()
        except ValueError:
            continue
        records.append(
            {
                "event_date": event_date,
                "category": _normalize_category(row["category"]),
                "type": row["type"],
                "amount": float(row["amount"]),
                "description": row["description"],
            }
        )
    return records


def _category_summary(records: List[dict]) -> dict:
    totals = defaultdict(float)
    for r in records:
        if r["type"] == "expense":
            totals[r["category"]] += r["amount"]
    return {k: round(v, 2) for k, v in totals.items()}


def _build_overview_context(ref_date: date) -> dict:
    current_end = ref_date
    current_start = current_end - timedelta(days=29)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=30)

    curr_records = _fetch_db_records(current_start, current_end)
    prev_records = _fetch_db_records(previous_start, previous_end)

    curr_income, curr_expense = _sum_income_expense(curr_records)
    prev_income, prev_expense = _sum_income_expense(prev_records)
    curr_categories = _category_summary(curr_records)
    prev_categories = _category_summary(prev_records)

    category_changes = []
    for cat in DISPLAY_CATEGORIES:
        curr_val = curr_categories.get(cat, 0.0)
        prev_val = prev_categories.get(cat, 0.0)
        if curr_val == 0 and prev_val == 0:
            continue
        change_pct = 0.0
        if prev_val > 0:
            change_pct = (curr_val - prev_val) / prev_val
        elif curr_val > 0:
            change_pct = 1.0
        category_changes.append(
            {
                "category": cat,
                "current": curr_val,
                "previous": prev_val,
                "change_pct": round(change_pct, 4),
            }
        )
    category_changes.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    anomalies = sorted(
        [r for r in curr_records if r["type"] == "expense"],
        key=lambda x: x["amount"],
        reverse=True,
    )[:5]
    anomaly_payload = [
        {
            "date": r["event_date"].isoformat(),
            "category": r["category"],
            "amount": round(r["amount"], 2),
            "description": r["description"],
        }
        for r in anomalies
    ]

    return {
        "period": {"start": current_start.isoformat(), "end": current_end.isoformat()},
        "summary": {
            "income": round(curr_income, 2),
            "expense": round(curr_expense, 2),
            "net": round(curr_income - curr_expense, 2),
        },
        "previous_summary": {
            "income": round(prev_income, 2),
            "expense": round(prev_expense, 2),
            "net": round(prev_income - prev_expense, 2),
        },
        "category_changes": category_changes[:6],
        "largest_transactions": anomaly_payload,
    }


def _build_behavior_context(ref_date: date) -> dict:
    end = ref_date
    start = end - timedelta(days=179)
    records = _fetch_db_records(start, end)
    monthly_totals = defaultdict(lambda: {"income": 0.0, "expense": 0.0})
    category_monthly = defaultdict(lambda: defaultdict(float))

    for r in records:
        month_key = r["event_date"].strftime("%Y-%m")
        if r["type"] == "income":
            monthly_totals[month_key]["income"] += r["amount"]
        else:
            monthly_totals[month_key]["expense"] += r["amount"]
            category_monthly[r["category"]][month_key] += r["amount"]

    months_sorted = sorted(monthly_totals.keys())
    patterns = []
    risk_flags = []
    for cat in DISPLAY_CATEGORIES:
        series = [category_monthly[cat].get(m, 0.0) for m in months_sorted]
        if not any(series):
            continue
        avg = statistics.mean(series)
        std = statistics.pstdev(series) if len(series) > 1 else 0.0
        trend = "stable"
        if len(series) >= 2:
            if series[-1] > (series[0] * 1.15):
                trend = "up"
            elif series[-1] < (series[0] * 0.85):
                trend = "down"
        patterns.append(
            {
                "category": cat,
                "average": round(avg, 2),
                "volatility": round(std, 2),
                "trend": trend,
            }
        )
        if avg > 0 and std > avg * 0.5:
            risk_flags.append(
                {
                    "category": cat,
                    "std": round(std, 2),
                    "avg": round(avg, 2),
                    "message": "波动显著，建议监控",
                }
            )

    behavior_monthly = [
        {
            "month": month,
            "income": round(monthly_totals[month]["income"], 2),
            "expense": round(monthly_totals[month]["expense"], 2),
        }
        for month in months_sorted
    ]

    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "monthly_totals": behavior_monthly,
        "category_patterns": patterns,
        "risk_flags": risk_flags[:5],
    }


def _build_advice_center_context(ref_date: date) -> dict:
    end = ref_date
    start = end - timedelta(days=89)
    records = _fetch_db_records(start, end)
    income, expense = _sum_income_expense(records)
    net = income - expense
    savings_rate = (net / income) if income > 0 else 0.0

    category_totals = _category_summary(records)
    total_expense = sum(category_totals.values())
    category_share = []
    for cat, value in category_totals.items():
        share = (value / total_expense) if total_expense else 0
        category_share.append(
            {"category": cat, "amount": round(value, 2), "share": round(share, 4)}
        )
    category_share.sort(key=lambda x: x["share"], reverse=True)

    fixed_candidates = [
        cat
        for cat, value in category_totals.items()
        if value > 0 and value >= total_expense * 0.15
    ]
    pressure_points = [
        item["category"] for item in category_share if item["share"] > 0.3
    ]

    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "income_total": round(income, 2),
        "expense_total": round(expense, 2),
        "net": round(net, 2),
        "savings_rate": round(savings_rate, 4),
        "category_share": category_share[:6],
        "fixed_candidates": fixed_candidates,
        "pressure_points": pressure_points,
    }


def build_advice_payload() -> dict:
    today = date.today()
    return {
        "overview": _build_overview_context(today),
        "behavior": _build_behavior_context(today),
        "advice": _build_advice_center_context(today),
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForCausalLM] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"
_resolved_model_dir: Optional[Path] = None
_is_vl_model: bool = False
_processor = None


def _row_to_bill(row: sqlite3.Row) -> Bill:
    metadata = json.loads(row["metadata"]) if row["metadata"] else None
    return Bill(
        id=row["id"],
        event_date=datetime.fromisoformat(row["event_date"]).date(),
        category=row["category"],
        type=row["type"],
        amount=row["amount"],
        currency=row["currency"],
        description=row["description"],
        source=row["source"],
        metadata=metadata,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def fetch_recent_bills(limit: int = 20) -> List[Bill]:
    with _db_lock:
        cur = _db_conn.execute(
            """
            SELECT * FROM bills
            ORDER BY datetime(event_date) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [_row_to_bill(row) for row in rows]


def build_bill_context(limit: int = 20) -> str:
    bills = fetch_recent_bills(limit)
    if not bills:
        return ""
    lines = ["以下为系统已记录的最新账单（最多20条，时间倒序）："]
    for bill in bills:
        lines.append(
            f"- {bill.event_date.isoformat()} | {bill.category} | {bill.type} | {bill.amount:.2f}{bill.currency} | {bill.description}"
        )
    return "\n".join(lines)


def build_retrieval_context(month_window: Optional[int] = 12) -> str:
    """
    基于整个数据库构建检索上下文：
    - 全局总览（总收入/支出）
    - 按类别汇总（收入/支出）
    - 按月汇总（最近 N 个月，或全量）
    - Top 商户（支出 Top 10）
    """
    with _db_lock:
        totals = _db_conn.execute(
            """
            SELECT 
                SUM(CASE WHEN type='income' THEN amount ELSE 0 END) AS total_income,
                SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) AS total_expense
            FROM bills
            """
        ).fetchone()
        by_category = _db_conn.execute(
            """
            SELECT category, type, SUM(amount) AS total
            FROM bills
            GROUP BY category, type
            ORDER BY total DESC
            """
        ).fetchall()
        # 按月汇总
        if month_window and month_window > 0:
            by_month = _db_conn.execute(
                f"""
                SELECT strftime('%Y-%m', date(event_date)) AS ym, type, SUM(amount) AS total
                FROM bills
                GROUP BY ym, type
                ORDER BY ym DESC
                """
            ).fetchall()
        else:
            by_month = _db_conn.execute(
                """
                SELECT strftime('%Y-%m', date(event_date)) AS ym, type, SUM(amount) AS total
                FROM bills
                GROUP BY ym, type
                ORDER BY ym DESC
                """
            ).fetchall()
        # Top 商户（从 metadata.raw/merchant 或 metadata.merchant 中提取）
        top_merchants = _db_conn.execute(
            """
            SELECT 
                COALESCE(
                    json_extract(metadata, '$.merchant'),
                    json_extract(metadata, '$.raw.商户'),
                    json_extract(metadata, '$.raw.merchant'),
                    '未知商户'
                ) AS merchant,
                SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) AS expense_total,
                COUNT(*) AS cnt
            FROM bills
            GROUP BY merchant
            ORDER BY expense_total DESC
            LIMIT 10
            """
        ).fetchall()

    lines = []
    lines.append("以下为基于全库的检索上下文：")
    lines.append(f"- 总收入：{(totals['total_income'] or 0):.2f}")
    lines.append(f"- 总支出：{(totals['total_expense'] or 0):.2f}")

    # 类别汇总
    lines.append("\n[按类别汇总]")
    # 合并为 dict[(category, type)] = total
    category_map = {}
    for row in by_category:
        category_map.setdefault(row["category"], {}).update({row["type"]: row["total"]})
    for cat, tmap in category_map.items():
        income = tmap.get("income", 0) or 0
        expense = tmap.get("expense", 0) or 0
        lines.append(f"- {cat} | 收入：{income:.2f} | 支出：{expense:.2f}")

    # 月度汇总（可截断最近 N 个月）
    lines.append("\n[按月汇总]")
    # 聚合为 dict[ym] = {"income": x, "expense": y}
    month_map = {}
    for row in by_month:
        ym = row["ym"]
        month_map.setdefault(ym, {}).update({row["type"]: row["total"]})
    # 只保留最近 month_window 个月（如果设置了）
    months_sorted = sorted(month_map.keys(), reverse=True)
    if month_window and month_window > 0:
        months_sorted = months_sorted[:month_window]
    for ym in months_sorted:
        income = month_map[ym].get("income", 0) or 0
        expense = month_map[ym].get("expense", 0) or 0
        lines.append(f"- {ym} | 收入：{income:.2f} | 支出：{expense:.2f}")

    # Top 商户
    lines.append("\n[支出 Top 商户]")
    for row in top_merchants:
        lines.append(f"- {row['merchant']} | 支出：{(row['expense_total'] or 0):.2f} | 笔数：{row['cnt']}")

    return "\n".join(lines)


def _force_no_cot_instruction() -> str:
    return (
        "重要：不要输出思考过程、草稿或 <think> 内容；只输出清晰、可核对的最终答案。"
    )


def _sanitize_reply(text: str) -> str:
    if not text:
        return text
    # 去除明显的 <think> 开头段
    if text.lstrip().startswith("<think>"):
        # 尝试剪掉直到 </think>，若没有则剪掉到第一个空行
        start = text.find("<think>")
        end = text.find("</think>", start + 7)
        if end != -1:
            text = text[end + len("</think>") :].lstrip()
        else:
            # 没有闭合标签，剪到第一个双换行或整段
            sep = "\n\n"
            pos = text.find(sep, start + 7)
            text = text[pos + len(sep) :].lstrip() if pos != -1 else ""
    # 清理残余的 think 标签
    text = text.replace("<think>", "").replace("</think>", "")
    # 兜底：若清理后为空，返回简短答复提示
    return text.strip() or "（已根据数据库给出结果，未显示思考过程）"


def load_model_if_needed() -> None:
    global tokenizer, model, _resolved_model_dir, _is_vl_model, _processor
    if tokenizer is not None and model is not None:
        return

    # 解析模型目录：若根目录无 config.json 而存在子目录 model/config.json，则使用子目录
    base_dir = Path(MODEL_DIR)
    cfg_at_root = (base_dir / "config.json").exists()
    cfg_at_sub = (base_dir / "model" / "config.json").exists()
    _resolved_model_dir = base_dir / "model" if (not cfg_at_root and cfg_at_sub) else base_dir

    # 判断是否为 Qwen2.5-VL 等多模态模型
    model_type = ""
    try:
        with open(_resolved_model_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            model_type = str(cfg.get("model_type", "")).lower()
    except Exception:
        pass
    _is_vl_model = model_type in {"qwen2_5_vl", "qwen2_vl"}

    if _is_vl_model and AutoModelForVision2Seq is not None:
        tokenizer = AutoTokenizer.from_pretrained(_resolved_model_dir, trust_remote_code=True)
        try:
            _processor = AutoProcessor.from_pretrained(_resolved_model_dir, trust_remote_code=True) if AutoProcessor else None
        except Exception:
            _processor = None
        model = AutoModelForVision2Seq.from_pretrained(
            _resolved_model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(_resolved_model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            _resolved_model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()


@app.get("/health")
def health():
    try:
        load_model_if_needed()
        # 确保在第一次健康检查前尝试导入（若尚未导入）
        _import_jsonl_if_needed()
        with _db_lock:
            count = (
                _db_conn.execute("SELECT COUNT(*) AS c FROM bills").fetchone()["c"]
            )
        return {
            "status": "ok",
            "device": device,
            "bill_count": count,
            "database": str(DB_PATH),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/bills", response_model=List[Bill])
def list_bills(
    limit: int = Query(100, ge=1, le=500),
    category: Optional[str] = None,
    type: Optional[str] = Query(None, pattern="^(expense|income)$"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    sql = "SELECT * FROM bills WHERE 1=1"
    params: List = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if type:
        sql += " AND type = ?"
        params.append(type)
    if start_date:
        sql += " AND date(event_date) >= date(?)"
        params.append(start_date.isoformat())
    if end_date:
        sql += " AND date(event_date) <= date(?)"
        params.append(end_date.isoformat())
    sql += " ORDER BY datetime(event_date) DESC, id DESC LIMIT ?"
    params.append(limit)

    with _db_lock:
        rows = _db_conn.execute(sql, params).fetchall()
    return [_row_to_bill(row) for row in rows]


@app.post("/bills", response_model=Bill, status_code=201)
def create_bill(bill: BillCreate):
    metadata_json = (
        json.dumps(bill.metadata, ensure_ascii=False) if bill.metadata else None
    )
    now = datetime.utcnow().isoformat()

    with _db_lock:
        cursor = _db_conn.execute(
            """
            INSERT INTO bills (event_date, category, type, amount, currency, description, source, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bill.event_date.isoformat(),
                bill.category,
                bill.type,
                bill.amount,
                bill.currency,
                bill.description,
                bill.source,
                metadata_json,
                now,
            ),
        )
        _db_conn.commit()
        new_id = cursor.lastrowid
        row = _db_conn.execute("SELECT * FROM bills WHERE id = ?", (new_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="账单写入失败")
    return _row_to_bill(row)


@app.delete("/bills/{bill_id}")
def delete_bill(bill_id: int):
    with _db_lock:
        # 先检查账单是否存在
        row = _db_conn.execute("SELECT id FROM bills WHERE id = ?", (bill_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="账单不存在")
        # 删除账单
        _db_conn.execute("DELETE FROM bills WHERE id = ?", (bill_id,))
        _db_conn.commit()
    return {"message": "账单已删除", "id": bill_id}


@app.post("/bills/ocr")
async def ocr_bill(file: UploadFile = File(...)):
    """
    上传图片进行OCR识别，返回解析后的账单信息
    """
    import io
    from PIL import Image
    
    # 检查是否为图片文件
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    # 读取图片
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        # 转换为RGB格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片读取失败: {str(e)}")
    
    # 检查是否是多模态模型
    load_model_if_needed()
    if not _is_vl_model or _processor is None:
        # 如果不是视觉模型，使用文本描述的方式
        # 这里可以集成其他OCR服务，暂时返回提示
        raise HTTPException(
            status_code=501, 
            detail="当前模型不支持图片识别，请使用文本描述或配置视觉模型"
        )
    
    # 使用视觉模型进行识别
    try:
        # 构建提示词
        prompt = "请识别这张票据图片中的账单信息，包括日期、金额、类别、商户名称等。请以JSON格式输出，包含以下字段：event_date(YYYY-MM-DD格式的日期), category(类别), type(expense或income), amount(金额数字), currency(币种，默认CNY), description(描述信息)。仅输出JSON，不要其他内容。"
        
        # 使用processor处理图片和文本
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 使用processor处理
        try:
            # 尝试使用processor处理图片和文本
            if hasattr(_processor, 'apply_chat_template'):
                text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = _processor.process_vision_info(messages)
                inputs = _processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)
                
                # 生成回复
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                reply = _processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            else:
                # 如果processor不支持，使用tokenizer和模型直接处理
                # 这里需要根据具体模型调整
                raise NotImplementedError("Processor不支持图片处理")
        except Exception as proc_error:
            # 如果视觉处理失败，尝试使用文本描述方式
            print(f"[WARN] Vision processing failed: {proc_error}, falling back to text description")
            # 使用chat API来解析图片描述
            fallback_prompt = f"这是一张票据图片（文件名：{file.filename}），请根据常见的票据格式，生成一个示例账单JSON，包含event_date(YYYY-MM-DD), category, type(expense或income), amount(数字), currency(默认CNY), description。仅输出JSON。"
            chat_req = ChatRequest(
                messages=[ChatMessage(role="user", content=fallback_prompt)],
                include_bill_context=False
            )
            chat_resp = chat(chat_req)
            reply = chat_resp.reply
        
        # 解析JSON
        import re
        import json
        # 尝试找到JSON对象（支持嵌套）
        json_match = None
        # 先尝试找到 ```json 代码块
        json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', reply, re.DOTALL)
        if json_block_match:
            json_match = json_block_match.group(1)
        else:
            # 尝试找到第一个 { 到最后一个 } 之间的内容
            first_brace = reply.find('{')
            last_brace = reply.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_match = reply[first_brace:last_brace+1]
        
        if json_match:
            try:
                bill_data = json.loads(json_match)
                # 验证和规范化数据
                if 'event_date' not in bill_data or 'amount' not in bill_data:
                    raise ValueError("缺少必要字段")
                return {"parsed": bill_data, "raw_text": reply}
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=500, detail=f"解析JSON失败: {str(e)}, 原始回复: {reply[:200]}")
        else:
            raise HTTPException(status_code=500, detail=f"未找到有效的JSON，原始回复: {reply[:200]}")
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] OCR failed: {error_detail}")
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")


@app.get("/bills/summary")
def bill_summary():
    with _db_lock:
        totals = _db_conn.execute(
            """
            SELECT 
                SUM(CASE WHEN type='income' THEN amount ELSE 0 END) AS total_income,
                SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) AS total_expense
            FROM bills
            """
        ).fetchone()
        by_category = _db_conn.execute(
            """
            SELECT category, type, SUM(amount) AS total
            FROM bills
            GROUP BY category, type
            ORDER BY total DESC
            """
        ).fetchall()
    return {
        "total_income": totals["total_income"] or 0,
        "total_expense": totals["total_expense"] or 0,
        "by_category": [
            {"category": row["category"], "type": row["type"], "total": row["total"]}
            for row in by_category
        ],
    }


@app.get("/reports/aggregate")
def aggregate_reports(
    start_date: Optional[date] = Query(None, description="自定义开始日期 YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="自定义结束日期 YYYY-MM-DD"),
):
    if start_date and end_date:
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        records = _fetch_db_records(start_date, end_date)
        # 支持空数据直接返回空报表，避免前端 404
        return {"custom": _build_custom_report(records, start_date, end_date)}
    return build_report_payload()


@app.get("/advice/context")
def advice_context():
    try:
        return build_advice_payload()
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] /advice/context failed: {error_detail}")
        raise HTTPException(status_code=500, detail=f"生成财务建议上下文失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    load_model_if_needed()

    system_prompt = req.system_prompt or DEFAULT_SYSTEM_PROMPT
    # 注入禁止 CoT 的系统指令
    system_prompt = f"{system_prompt}\n\n{_force_no_cot_instruction()}".strip()
    # RAG：如果启用检索，则使用全库检索上下文；否则按原逻辑使用最新 20 条
    use_retrieval = req.retrieval if req.retrieval is not None else DEFAULT_RETRIEVAL
    if use_retrieval:
        bill_context = build_retrieval_context(month_window=req.retrieval_months)
    else:
        bill_context = build_bill_context() if req.include_bill_context else ""

    conversation_parts: List[str] = []
    if system_prompt:
        conversation_parts.append(f"<|system|>\n{system_prompt}")
    if bill_context:
        conversation_parts.append(f"<|system|>\n{bill_context}")
    for m in req.messages:
        role = m.role.strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        if role == "system":
            conversation_parts.append(f"<|system|>\n{m.content}")
        elif role == "user":
            conversation_parts.append(f"<|user|>\n{m.content}")
        else:
            conversation_parts.append(f"<|assistant|>\n{m.content}")

    if hasattr(tokenizer, "apply_chat_template"):
        hf_messages = []
        if system_prompt:
            hf_messages.append({"role": "system", "content": system_prompt})
        if bill_context:
            hf_messages.append({"role": "system", "content": bill_context})
        for m in req.messages:
            hf_messages.append({"role": m.role, "content": m.content})
        input_ids = tokenizer.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        input_text = "\n".join(conversation_parts) + "\n<|assistant|>\n"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # 服务端统一长度策略（默认/上限/下限）
    eff_max_new = req.max_new_tokens if req.max_new_tokens and req.max_new_tokens > 0 else GEN_DEFAULT_MAX_NEW_TOKENS
    eff_max_new = max(32, min(eff_max_new, GEN_MAX_NEW_TOKENS_CAP))
    eff_min_new = None
    if req.min_new_tokens is not None:
        try:
            eff_min_new = max(0, min(req.min_new_tokens, eff_max_new // 2))
        except Exception:
            eff_min_new = None

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=eff_max_new,
            min_new_tokens=eff_min_new,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0, input_ids.shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    reply = _sanitize_reply(reply)

    return ChatResponse(reply=reply.strip())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)





