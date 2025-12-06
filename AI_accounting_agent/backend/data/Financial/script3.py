import pandas as pd
import yfinance as yf
import time
import os

# --- 1. 配置 ---
# 为了不覆盖之前腾讯源的数据，我们新建一个文件夹
TARGET_DIR = os.path.join('fintechcom', 'AI_accounting_agent', 'backend', 'data', '理财', '美股')

# 如果你的环境需要代理，请在这里设置。
# 如果你的容器环境已经配置了全局代理，可以注释掉这两行。
# PROXY = "http://127.0.0.1:7890"  
PROXY = None # 如果上面的 requests 报错，尝试把上一行解除注释并填入你的代理IP

# --- 2. 美股 Top 100 名单 (复用) ---
TOP_100_US_MAP = {
    'AAPL': '苹果', 'MSFT': '微软', 'NVDA': '英伟达', 'GOOG': '谷歌C', 'GOOGL': '谷歌A',
    'AMZN': '亚马逊', 'META': 'Meta', 'TSLA': '特斯拉', 'AVGO': '博通', 'AMD': '超威半导体',
    'TSM': '台积电', 'QCOM': '高通', 'TXN': '德州仪器', 'INTC': '英特尔', 'MU': '美光科技',
    'AMAT': '应用材料', 'LRCX': '泛林集团', 'ADI': '亚德诺', 'IBM': 'IBM', 'ORCL': '甲骨文',
    'CSCO': '思科', 'ACN': '埃森哲', 'ADBE': 'Adobe', 'CRM': '赛富时', 'INTU': '财捷',
    'NOW': 'ServiceNow', 'UBER': '优步', 'NFLX': '奈飞', 'BRK-B': '伯克希尔B', # Yahoo用BRK-B而不是BRK.B
    'JPM': '摩根大通', 'V': '威士', 'MA': '万事达', 'BAC': '美国银行', 'WFC': '富国银行',
    'MS': '摩根士丹利', 'GS': '高盛', 'C': '花旗集团', 'BLK': '贝莱德', 'AXP': '美国运通',
    'SPGI': '标普全球', 'CB': '安达保险', 'MMC': '威达信', 'PGR': '前进保险', 'LLY': '礼来',
    'UNH': '联合健康', 'JNJ': '强生', 'MRK': '默沙东', 'ABBV': '艾伯维', 'TMO': '赛默飞',
    'ABT': '雅培', 'PFE': '辉瑞', 'AMGN': '安进', 'ISRG': '直觉外科', 'DHR': '丹纳赫',
    'BMY': '百时美施贵宝', 'GILD': '吉利德', 'SYK': '史赛克', 'ELV': 'Elevance', 'VRTX': '福泰制药',
    'REGN': '再生元', 'ZTS': '硕腾', 'WMT': '沃尔玛', 'PG': '宝洁', 'COST': '好市多',
    'HD': '家得宝', 'KO': '可口可乐', 'PEP': '百事', 'MCD': '麦当劳', 'NKE': '耐克',
    'SBUX': '星巴克', 'DIS': '迪士尼', 'PM': '菲利普莫里斯', 'LOW': '劳氏', 'TJX': 'TJX公司',
    'BKNG': '缤客', 'MAR': '万豪国际', 'XOM': '埃克森美孚', 'CVX': '雪佛龙', 'COP': '康菲石油',
    'SLB': '斯伦贝谢', 'EOG': 'EOG能源', 'GE': '通用电气', 'CAT': '卡特彼勒', 'DE': '迪尔',
    'HON': '霍尼韦尔', 'UNP': '联合太平洋', 'UPS': '联合包裹', 'RTX': '雷神技术', 'LMT': '洛克希德马丁',
    'BA': '波音', 'ADP': '自动数据处理', 'WM': '废弃物管理', 'VZ': '威瑞森', 'T': 'AT&T',
    'CMCSA': '康卡斯特', 'TMUS': 'T-Mobile', 'LIN': '林德', 'SHW': '宣伟', 'NEE': 'NextEra能源',
    'SO': '南方公司'
}

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 目录已创建: {directory}")

def download_yahoo_data():
    ensure_directory_exists(TARGET_DIR)
    print(f"🚀 [Yahoo Finance] 准备下载 {len(TOP_100_US_MAP)} 只美股数据...")
    
    success_count = 0
    
    for ticker, name in TOP_100_US_MAP.items():
        try:
            # 使用 yfinance 下载
            # period="1y" 表示过去一年
            # auto_adjust=True 表示自动处理拆股和分红(复权)
            ticker_obj = yf.Ticker(ticker)
            
            # 下载历史数据
            # 如果有代理，可以在这里通过 proxy 参数传入，但 yfinance 通常会自动读取环境变量
            df = ticker_obj.history(period="1y", proxy=PROXY)
            
            if not df.empty:
                # Yahoo 的列名通常是 Open, High, Low, Close, Volume, Dividends, Stock Splits
                # 我们只要前5列，且统一列名为小写，保持和之前 A股/港股数据格式一致
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # 增加标识列
                df['code'] = ticker
                df['name'] = name
                
                # 索引已经是 Date 类型了，不需要转换
                # df.index.name = 'date' 
                
                # 保存
                file_name = f"{name}.csv"
                save_path = os.path.join(TARGET_DIR, file_name)
                df.to_csv(save_path, encoding='utf_8_sig')
                
                print(f"✅ [{success_count+1}/{len(TOP_100_US_MAP)}] {name} ({ticker})")
                success_count += 1
            else:
                print(f"⚠️ 数据为空: {name} ({ticker})")
                
        except Exception as e:
            print(f"❌ 下载失败: {name} ({ticker}) - {e}")
            
        # 礼貌性延时
        time.sleep(0.2)
        
    print("-" * 30)
    print(f"🎉 Yahoo 任务完成！成功 {success_count} 个。")

if __name__ == '__main__':
    download_yahoo_data()