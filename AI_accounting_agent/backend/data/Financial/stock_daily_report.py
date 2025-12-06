#!/usr/bin/env python3
"""Fetch per-stock news and research report datasets for downstream agents."""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List

import akshare as ak
import pandas as pd

try:
    import numpy as np
except Exception:  # numpy may be unavailable in some environments
    np = None  # type: ignore

try:
    import requests
except Exception:  # requests may be unavailable in some environments
    requests = None  # type: ignore

# Default lookback window for news (days)
DEFAULT_NEWS_DAYS = 7
DEFAULT_REPORT_DAYS = 90
EASTMONEY_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
EASTMONEY_DEFAULT_PAGE_SIZE = 20
EASTMONEY_MAX_PAGES = 3
HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class FetchResult:
    name: str
    data: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def safe_fetch(name: str, func: Callable, *args, **kwargs) -> FetchResult:
    """Execute an AkShare call defensively, capturing exceptions as needed."""

    try:
        data = func(*args, **kwargs)
        if data is None or (hasattr(data, "empty") and getattr(data, "empty")):
            logging.warning("%s 返回空数据", name)
        return FetchResult(name=name, data=data)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("%s 调用失败: %s", name, exc)
        return FetchResult(name=name, error=str(exc))


def fetch_if_available(name: str, *args, **kwargs) -> FetchResult:
    """Fetch via AkShare only if the endpoint exists in the current version."""

    func = getattr(ak, name, None)
    if func is None:
        msg = "接口在当前 AkShare 版本中不可用"
        logging.warning("%s: %s", name, msg)
        return FetchResult(name=name, error=msg)
    return safe_fetch(name, func, *args, **kwargs)


def serialize_for_json(obj: Any) -> Any:
    """Convert pandas objects into JSON-friendly structures without NaN."""

    if isinstance(obj, pd.DataFrame):
        cleaned = obj.replace({pd.NA: None})
        if np is not None:
            cleaned = cleaned.replace({np.nan: None})
        return serialize_for_json(cleaned.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        cleaned = obj.replace({pd.NA: None})
        if np is not None:
            cleaned = cleaned.replace({np.nan: None})
        return serialize_for_json(cleaned.to_dict())
    if isinstance(obj, FetchResult):
        return {
            "name": obj.name,
            "ok": obj.ok,
            "error": obj.error,
            "data": serialize_for_json(obj.data),
        }
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if np is not None:
        numpy_scalars = (np.integer, np.floating, np.bool_)
        if isinstance(obj, numpy_scalars):  # type: ignore[arg-type]
            value = obj.item()
            return None if isinstance(value, float) and math.isnan(value) else value
    return obj


def _clean_text(text: Any) -> str:
    if not text:
        return ""
    cleaned = HTML_TAG_RE.sub("", str(text))
    cleaned = cleaned.replace("\u3000", " ").replace("\r\n", " ")
    return " ".join(cleaned.split())


def filter_news_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or days <= 0:
        return df
    cutoff = datetime.now() - timedelta(days=days)
    if "发布时间" not in df.columns:
        return df
    published = pd.to_datetime(df["发布时间"], errors="coerce")
    mask = published >= cutoff
    filtered = df[mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def filter_research_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or days <= 0:
        return df
    cutoff = datetime.now() - timedelta(days=days)
    candidate_cols = ["发布日期", "发布时间", "date", "报告日期"]
    date_col = next((col for col in candidate_cols if col in df.columns), None)
    if date_col is None:
        return df
    published = pd.to_datetime(df[date_col], errors="coerce")
    mask = published >= cutoff
    filtered = df[mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def _strip_jsonp_payload(raw_text: str, callback: str) -> Dict[str, Any]:
    payload = raw_text.strip()
    if payload.startswith("/**/"):
        payload = payload[4:].lstrip()
    prefix = f"{callback}("
    if not payload.startswith(prefix):
        raise ValueError("Unexpected JSONP callback format")
    payload = payload[len(prefix):].rstrip()
    if payload.endswith(");"):
        payload = payload[:-2].rstrip()
    elif payload.endswith(")"):
        payload = payload[:-1].rstrip()
    else:
        raise ValueError("Unexpected JSONP tail format")
    return json.loads(payload)


def _fetch_eastmoney_news_page(symbol: str, page_index: int, page_size: int) -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests 未安装，无法抓取东方财富新闻")
    if page_index < 1:
        raise ValueError("page_index must be >= 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    timestamp_ms = int(time.time() * 1000)
    callback = f"jQuery3510{random.randint(10**11, 10**12 - 1)}_{timestamp_ms}"
    inner_param = {
        "uid": "",
        "keyword": symbol,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": page_index,
                "pageSize": page_size,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    params = {
        "cb": callback,
        "param": json.dumps(inner_param, ensure_ascii=False),
        "_": str(timestamp_ms),
    }
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "cache-control": "no-cache",
        "connection": "keep-alive",
        "host": "search-api-web.eastmoney.com",
        "pragma": "no-cache",
        "referer": f"https://so.eastmoney.com/news/s?keyword={symbol}",
        "sec-ch-ua": '\"Chromium\";v=\"142\", \"Google Chrome\";v=\"142\", \"Not_A Brand\";v=\"99\"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '\"Windows\"',
        "sec-fetch-dest": "script",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    }
    response = requests.get(EASTMONEY_NEWS_URL, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data_json = _strip_jsonp_payload(response.text, callback)
    return data_json.get("result", {}).get("cmsArticleWebOld", [])


def fetch_eastmoney_news(symbol: str, max_pages: int = EASTMONEY_MAX_PAGES, page_size: int = EASTMONEY_DEFAULT_PAGE_SIZE) -> FetchResult:
    name = "eastmoney_news"
    try:
        clean_symbol = symbol.strip()
        if not clean_symbol:
            raise ValueError("股票代码不能为空")
        all_rows: List[Dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            rows = _fetch_eastmoney_news_page(clean_symbol, page, page_size)
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
        df = pd.DataFrame(all_rows)
        if df.empty:
            return FetchResult(name=name, data=pd.DataFrame())
        df["url"] = "http://finance.eastmoney.com/a/" + df["code"].astype(str).str.strip() + ".html"
        df.rename(
            columns={
                "date": "发布时间",
                "mediaName": "文章来源",
                "title": "新闻标题",
                "content": "新闻内容",
                "url": "新闻链接",
            },
            inplace=True,
        )
        df["新闻标题"] = df["新闻标题"].apply(_clean_text)
        df["新闻内容"] = df["新闻内容"].apply(_clean_text)
        df["关键词"] = clean_symbol
        desired_cols = [
            "关键词",
            "新闻标题",
            "新闻内容",
            "发布时间",
            "文章来源",
            "新闻链接",
        ]
        available = [col for col in desired_cols if col in df.columns]
        df = df[available]
        return FetchResult(name=name, data=df)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("东方财富新闻抓取失败: %s", exc)
        return FetchResult(name=name, error=str(exc))


def gather_stock_payload(symbol: str, max_news_pages: int, news_page_size: int) -> Dict[str, FetchResult]:
    """Collect news and research report datasets for the chosen stock symbol."""

    payload: Dict[str, FetchResult] = {}

    news_result = fetch_eastmoney_news(
        symbol,
        max_pages=max_news_pages,
        page_size=news_page_size,
    )
    needs_fallback = not (
        news_result.ok
        and isinstance(news_result.data, pd.DataFrame)
        and not news_result.data.empty
    )
    if needs_fallback:
        fallback_news = fetch_if_available(
            "stock_news_em",
            symbol=symbol,
        )
        if fallback_news.ok:
            news_result = fallback_news
    payload["news"] = news_result

    research_result = fetch_if_available(
        "stock_research_report_em",
        symbol=symbol,
    )
    payload["research_reports"] = research_result

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a stock daily report via AkShare")
    parser.add_argument("symbol", help="股票代码，例如 603777 或 000001")
    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="输出目录 (默认: ./reports)",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="历史行情回看天数 (默认: 30)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志等级",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    symbol = args.symbol.strip()
    if not symbol:
        raise SystemExit("股票代码不能为空")

    payload = gather_stock_payload(symbol, args.history_days)
    output_paths = save_outputs(symbol, payload, Path(args.output_dir))

    print("报告生成完成：")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
