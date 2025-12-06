#!/usr/bin/env bash
# Simple wrapper around generate_reports.py.
# Adjust the variables below to control which symbols are processed
# and how the Eastmoney crawler behaves.

set -euo pipefail

# ==== User-configurable parameters ====
SYMBOLS=(
  601288  # 农业银行
  601398  # 工商银行
  600519  # 贵州茅台
  601857  # 中国石油
  601988  # 中国银行
  601138  # 工业富联
  601628  # 中国人寿
  600036  # 招商银行
  600900  # 长江电力
  601088  # 中国神华
)     # Space-separated list of tickers
OUTPUT_DIR="./reports"      # Directory to store JSON outputs
NEWS_DAYS=7                 # Lookback window for news filtering (days)
REPORT_DAYS=90              # Lookback window for research reports (days)
MAX_PAGES=3                 # Eastmoney pagination depth
PAGE_SIZE=20                # Results per page (1-100)
LOG_LEVEL="INFO"           # DEBUG, INFO, WARNING, or ERROR
# ====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python generate_reports.py \
  "${SYMBOLS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --news-days "$NEWS_DAYS" \
  --report-days "$REPORT_DAYS" \
  --max-pages "$MAX_PAGES" \
  --page-size "$PAGE_SIZE" \
  --log-level "$LOG_LEVEL"
