#!/usr/bin/env python3
"""Batch generator for stock news & research report files."""

import argparse
import json
import logging
from pathlib import Path
from typing import List

from stock_daily_report import (
    DEFAULT_NEWS_DAYS,
    DEFAULT_REPORT_DAYS,
    EASTMONEY_DEFAULT_PAGE_SIZE,
    EASTMONEY_MAX_PAGES,
    gather_stock_payload,
    serialize_for_json,
    filter_news_by_days,
    filter_research_by_days,
)


def generate_for_symbol(
    symbol: str,
    output_dir: Path,
    news_days: int,
    report_days: int,
    max_pages: int,
    page_size: int,
) -> None:
    payload = gather_stock_payload(symbol, max_news_pages=max_pages, news_page_size=page_size)
    news_result = payload.get("news")
    if news_result and news_result.ok and hasattr(news_result.data, "copy"):
        news_result.data = filter_news_by_days(news_result.data, news_days)
    report_result = payload.get("research_reports")
    if report_result and report_result.ok and hasattr(report_result.data, "copy"):
        filtered_reports = filter_research_by_days(report_result.data, report_days)
        report_result.data = filtered_reports.head(15).reset_index(drop=True)
    serializable = {k: serialize_for_json(v) for k, v in payload.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{symbol}_news_report.json"
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("生成完成：%s", json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate news/research payloads for multiple stocks")
    parser.add_argument("symbols", nargs="+", help="一个或多个股票代码")
    parser.add_argument("--output-dir", default="./reports", help="输出目录")
    parser.add_argument("--news-days", type=int, default=DEFAULT_NEWS_DAYS, help="新闻回看天数")
    parser.add_argument("--report-days", type=int, default=DEFAULT_REPORT_DAYS, help="研报回看天数")
    parser.add_argument("--max-pages", type=int, default=EASTMONEY_MAX_PAGES, help="东方财富新闻分页数")
    parser.add_argument("--page-size", type=int, default=EASTMONEY_DEFAULT_PAGE_SIZE, help="每页新闻条数")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志等级")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    for symbol in args.symbols:
        symbol_clean = symbol.strip()
        if not symbol_clean:
            logging.warning("跳过空白股票代码")
            continue
        try:
            generate_for_symbol(
                symbol_clean,
                output_dir,
                args.news_days,
                args.report_days,
                args.max_pages,
                args.page_size,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("%s 生成失败: %s", symbol_clean, exc)


if __name__ == "__main__":
    main()
