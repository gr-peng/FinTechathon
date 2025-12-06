import pandas as pd
import yfinance as yf
import time
import os

# --- 配置 ---
TARGET_DIR = os.path.join('fintechcom', 'AI_accounting_agent', 'backend', 'data', '理财', '黄金')
PROXY = None  # 如果需要代理，填入 "http://127.0.0.1:7890"

# --- 优化的黄金代码列表 ---
GOLD_MAP = {
    'GC=F':     'COMEX黄金期货_主力',   # 【最推荐】数据质量通常最高
    'GLD':      '黄金ETF_SPDR',         # 【最稳定】作为美股ETF，数据绝不会错
    'XAUUSD=X': '伦敦金_现货_修正版'     # 【尝试】换个代码写法试试
}

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_gold_data_fixed():
    ensure_directory_exists(TARGET_DIR)
    print(f"🚀 [Yahoo Fix] 开始下载修复版黄金数据...")
    
    for ticker, name in GOLD_MAP.items():
        try:
            print(f"⏳ 请求: {name} ({ticker})...")
            
            # 下载数据
            df = yf.Ticker(ticker).history(period="1y", proxy=PROXY, auto_adjust=False)
            
            # --- 关键检查步骤 ---
            if df.empty:
                print(f"⚠️ 数据为空: {name}")
                continue
                
            # 检查收盘价是否全部为 0 (解决你刚才遇到的问题)
            if (df['Close'] == 0).all():
                print(f"❌ 数据无效 (价格全为0): {name} ({ticker}) - 已跳过")
                continue

            # 数据清洗
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['code'] = ticker
            df['name'] = name
            
            # 保存
            file_name = f"{name}.csv"
            save_path = os.path.join(TARGET_DIR, file_name)
            df.to_csv(save_path, encoding='utf_8_sig')
            
            latest_price = round(df['close'].iloc[-1], 2)
            print(f"✅ 成功保存: {file_name} | 最新价: {latest_price}")
            
        except Exception as e:
            print(f"❌ 下载异常: {name} ({ticker}) - {e}")
            
        time.sleep(1)

    print("-" * 30)
    print("🎉 任务结束。建议优先使用 'COMEX黄金期货' 或 '黄金ETF' 的数据。")

if __name__ == '__main__':
    download_gold_data_fixed()