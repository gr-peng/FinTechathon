import akshare as ak
import pandas as pd

def safe(func, *args, **kw):
    try:
        return func(*args, **kw)
    except Exception as e:
        print(f"[Error] {func.__name__}: {e}")
        return None


def fetch_all(stock_code):
    result = {}

    print("[*] 获取股票实时行情（新浪接口，100% 稳定）")
    result["quote"] = safe(ak.stock_zh_a_spot, stock=stock_code)

    print("[*] 获取日K（东财最新接口）")
    result["kline"] = safe(
        ak.stock_zh_a_hist,
        symbol=stock_code,
        period="daily",
        start_date="20220101",
        end_date="20251231",
        adjust="qfq",
    )

    print("[*] 获取公告（东财 datacenter 最新接口）")
    result["notice"] = safe(
        ak.stock_notice_report_em,
        stock=stock_code
    )

    print("[*] 获取研报（东财 datacenter 最新接口）")
    result["research"] = safe(
        ak.stock_research_report_em,
        stock=stock_code,
        indicator="最新"
    )

    print("[*] 获取财务摘要（同花顺接口）")
    result["financial"] = safe(
        ak.stock_financial_abstract_ths,
        symbol=stock_code
    )

    print("[*] 获取利润预测（同花顺）")
    result["forecast"] = safe(
        ak.stock_profit_forecast_ths,
        symbol=stock_code
    )

    return result


def save_doc(result, path="report.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for key, df in result.items():
            f.write(f"===== {key} =====\n")
            if isinstance(df, pd.DataFrame):
                f.write(df.to_string())
            else:
                f.write(str(df))
            f.write("\n\n")

    print(f"[Saved] {path}")


if __name__ == "__main__":
    stock = "600519"  # 茅台
    data = fetch_all(stock)
    save_doc(data)
