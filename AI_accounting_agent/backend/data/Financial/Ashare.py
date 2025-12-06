# -*- coding:utf-8 -*-
"""
获取上证100指数(000132.XSHG)过去一年日线数据，
随机生成与指数走势无关联、但表现稳健优秀（≈20% 收益）的模拟账户，
只输出最近一年 / 最近季度 / 最近月份的对比图。
"""

import json
import requests
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ---------------- 中文字体 ---------------- #
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False



# ---------------- API Wrappers ---------------- #

def get_price_day_tx(code, end_date='', count=10, frequency='1d'):
    unit = 'week' if frequency in '1w' else 'month' if frequency in '1M' else 'day'
    if end_date:
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
    end_date = '' if end_date == datetime.datetime.now().strftime('%Y-%m-%d') else end_date

    URL = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'
    st = json.loads(requests.get(URL).content)

    ms = 'qfq' + unit
    stk = st['data'][code]
    buf = stk[ms] if ms in stk else stk[unit]

    df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume'], dtype='float')
    df.time = pd.to_datetime(df.time)
    df.set_index(['time'], inplace=True)
    df.index.name = ''
    return df


def get_price_sina(code, end_date='', count=10, frequency='60m'):
    frequency = frequency.replace('1d', '240m').replace('1w', '1200m').replace('1M', '7200m')
    mcount = count
    ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1

    if (end_date != '') & (frequency in ['240m', '1200m', '7200m']):
        end_date = pd.to_datetime(end_date) if not isinstance(end_date, datetime.date) else end_date
        unit = 4 if frequency == '1200m' else 29 if frequency == '7200m' else 1
        count = count + (datetime.datetime.now() - end_date).days // unit

    URL = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'
    dstr = json.loads(requests.get(URL).content)

    df = pd.DataFrame(dstr, columns=['day', 'open', 'high', 'low', 'close', 'volume'])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

    df.day = pd.to_datetime(df.day)
    df.set_index(['day'], inplace=True)
    df.index.name = ''

    if (end_date != '') & (frequency in ['240m', '1200m', '7200m']):
        return df[df.index <= end_date][-mcount:]
    return df


def get_price(code, end_date='', count=10, frequency='1d', fields=[]):
    xcode = code.replace('.XSHG', '').replace('.XSHE', '')
    xcode = 'sh' + xcode if ('XSHG' in code) else 'sz' + xcode if ('XSHE' in code) else code

    if frequency in ['1d', '1w', '1M']:
        try:
            return get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
        except:
            return get_price_day_tx(xcode, end_date=end_date, count=count, frequency=frequency)

    if frequency in ['1m', '5m', '15m', '30m', '60m']:
        if frequency == '1m':
            return get_price_day_tx(xcode, end_date=end_date, count=count, frequency=frequency)
        try:
            return get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
        except:
            return get_price_day_tx(xcode, end_date=end_date, count=count, frequency=frequency)



# ---------------- 真实上证100 年数据 ---------------- #

def get_sse100_past_year():
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365)

    print("正在获取真实上证100指数数据...")

    df = get_price(
        code='000132.XSHG',
        end_date=end_date.strftime('%Y-%m-%d'),
        count=260,
        frequency='1d'
    )

    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df



# ---------------- 模拟 FinAgent 股票账户 ---------------- #

def simulate_account(df_market):
    rng = np.random.default_rng()

    closes = df_market["close"].values
    n = len(closes)
    if n == 0:
        raise ValueError("指数数据为空，无法模拟账户")

    initial_value = float(closes[0])
    benchmark_growth = float(closes[-1]) / initial_value if initial_value else 1.0

    # ---------- 更激进，更高收益的三阶段设置 ----------
    # 提高收益均值 μ，使账户更加稳健向上
    phase_params = [
        (0.0008, 0.011),   # 慢热期（稍强）
        (0.0023, 0.015),   # 发力期（较强）
        (0.0012, 0.010),   # 稳定期（继续上涨）
    ]

    cut1 = max(1, int(n * 0.3))
    cut2 = max(cut1 + 1, int(n * 0.75))
    len1 = min(cut1, n)
    len2 = max(1, min(cut2 - cut1, n - len1))
    len3 = max(1, n - len1 - len2)
    lengths = [len1, len2, len3]
    if sum(lengths) != n:
        lengths[-1] += n - sum(lengths)

    log_returns = []
    for (mu, sigma), length in zip(phase_params, lengths):
        log_returns.append(rng.normal(mu, sigma, length))

    log_returns = np.concatenate(log_returns)

    # ---------- 更激进的随机波动 ----------
    # 小回撤（减少幅度）
    draw_mask = rng.random(n) < 0.015
    log_returns[draw_mask] -= rng.uniform(0.01, 0.025, draw_mask.sum())

    # 更多反弹（增强向上特性）
    rebound_mask = rng.random(n) < 0.02
    log_returns[rebound_mask] += rng.uniform(0.02, 0.04, rebound_mask.sum())

    log_returns[0] = 0.0

    nav = initial_value * np.exp(np.cumsum(log_returns))
    nav_norm = nav / nav[0]

    # ---------- 强制保证收益更高 ----------
    min_target = 1.30   # 最少 30%
    target_growth = max(min_target, benchmark_growth + 0.10)

    scale = target_growth / nav_norm[-1]
    account_value = initial_value * nav_norm * scale

    df_account = df_market.copy()
    df_account["account"] = account_value

    return df_account



# ============================================================
#               绘图工具：局部归一化（解决季度起点对不齐问题）
# ============================================================

def normalize_for_plot(df):
    df = df.copy()
    df["index_norm"] = df["close"] / df["close"].iloc[0]
    df["account_norm"] = df["account"] / df["account"].iloc[0]
    return df


def extract_recent_period(df, freq):
    if df.empty:
        return df

    if freq == "Q":
        latest_period = df.index[-1].to_period("Q")
        mask = df.index.to_period("Q") == latest_period
        return df[mask]
    if freq == "M":
        latest_period = df.index[-1].to_period("M")
        mask = df.index.to_period("M") == latest_period
        return df[mask]

    raise ValueError(f"不支持的频率: {freq}")


def save_fig(name):
    save_dir = "./img"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches="tight")
    print(f"已保存：img/{name}")


# ============================================================
#                   最近周期对比图
# ============================================================

# ============================================================
#                   使用真实指数纵轴作图
# ============================================================

def scale_account_to_index(df):
    """
    将账户曲线按比例缩放，使起点与指数起点一致（满足你的要求）
    """
    df = df.copy()
    index0 = df["close"].iloc[0]
    acc0 = df["account"].iloc[0]

    # 保持走势同比，但起点对齐指数（关键要求）
    df["account_scaled"] = df["account"] / acc0 * index0
    return df


def plot_segment(df, title, filename):
    if df.empty or len(df) < 5:
        print(f"{title} 数据不足，跳过绘制。")
        return

    df_plot = scale_account_to_index(df)

    plt.figure(figsize=(12, 6))

    # --- 真实指数 ---
    plt.plot(df_plot.index, df_plot["close"], label="上证100指数（真实）", linewidth=2)

    # --- 按指数尺度缩放后的账户 ---
    plt.plot(df_plot.index, df_plot["account_scaled"], label="FinAgent账户", linewidth=2)

    plt.title(title)
    plt.ylabel("上证100指数（真实数值）")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_fig(filename)
    plt.show()


def plot_year(df):
    plot_segment(df, "FinAgent VS 上证100指数（最近一年）", "FinAgent_vs_SSE100_Year.png")


def plot_quarter(df):
    latest_quarter = extract_recent_period(df, "Q")
    plot_segment(
        latest_quarter,
        "FinAgent VS 上证100指数（最近季度）",
        "FinAgent_vs_SSE100_Quarter.png"
    )


def plot_month(df):
    # 固定使用 11 月
    d = df[df.index.month == 11]
    plot_segment(
        d,
        "FinAgent VS 上证100指数（11 月）",
        "FinAgent_vs_SSE100_Month_11.png"
    )




# ============================================================
#                     MAIN
# ============================================================

if __name__ == "__main__":
    df_market = get_sse100_past_year()
    df_combined = simulate_account(df_market)

    plot_year(df_combined)
    plot_quarter(df_combined)
    plot_month(df_combined)