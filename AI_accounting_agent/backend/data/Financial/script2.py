import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os

# --- 1. 配置保存路径 ---
TARGET_DIR = os.path.join('fintechcom', 'AI_accounting_agent', 'backend', 'data', '理财', '港股')

# --- 2. 核心数据获取函数 (带重试机制版) ---
def get_price_hk_safe_with_retry(code, frequency='1d', count=250):
    # 构造 URL
    url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{count},qfq'
    
    # --- 重试机制配置 ---
    # total=5: 最多重试 5 次
    # backoff_factor=1: 重试间隔 (1s, 2s, 4s...)，避免频繁请求被封
    # status_forcelist: 针对哪些状态码进行重试
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    # 创建一个 Session 会话
    with requests.Session() as session:
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        try:
            # timeout=15: 将超时时间从 4s 增加到 15s，防止代理慢导致断连
            response = session.get(url, timeout=15)
            
            # 如果状态码不是 200，主动抛出异常触发重试
            response.raise_for_status() 
            
            data = response.json()
            
            # 解析逻辑
            if code in data['data'] and 'qfqday' in data['data'][code]:
                buf = data['data'][code]['qfqday']
            elif code in data['data'] and 'day' in data['data'][code]:
                buf = data['data'][code]['day']
            else:
                return None

            # 强制只取前6列
            buf_fixed = [item[:6] for item in buf]
            
            # 转换为 DataFrame
            df = pd.DataFrame(buf_fixed, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            # 如果重试 5 次后依然失败，才会打印这个错误
            print(f"    ❌ [最终失败] {code}: {e}")
            return None

# --- 3. 港股核心资产 TOP 100 名单 ---
TOP_100_HK_MAP = {
    # --- 科技互联 & 新经济 ---
    'hk00981': '中芯国际', 'hk00386': '中国石油化工'
    
}

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"📁 目录已创建: {directory}")
        except Exception as e:
            print(f"❌ 无法创建目录: {e}")
            return False
    return True

def main():
    if not ensure_directory_exists(TARGET_DIR):
        return

    print(f"🚀 准备下载 {len(TOP_100_HK_MAP)} 只港股数据 (带重试机制)...")
    print(f"📂 目标文件夹: {TARGET_DIR}")
    
    success_count = 0
    
    for code, name in TOP_100_HK_MAP.items():
        # 检查文件是否已存在，如果已存在且大小正常，可以选择跳过（这里暂时先覆盖，保证数据最新）
        
        # 调用带重试的函数
        df = get_price_hk_safe_with_retry(code, frequency='1d', count=250)
        
        if df is not None and not df.empty:
            df['code'] = code
            df['name'] = name
            
            file_name = f"{name}.csv"
            save_path = os.path.join(TARGET_DIR, file_name)
            
            df.to_csv(save_path, encoding='utf_8_sig')
            
            print(f"✅ [{success_count+1}/{len(TOP_100_HK_MAP)}] {name} ({code})")
            success_count += 1
        else:
            print(f"⚠️ 跳过: {name} ({code}) - 数据获取失败")
            
        # 即使有重试，这里也保留一个小延时
        time.sleep(0.1)

    print("-" * 30)
    print(f"🎉 任务完成！应下载 {len(TOP_100_HK_MAP)} 个，成功 {success_count} 个。")
    
    if os.path.exists(TARGET_DIR):
        file_count = len([f for f in os.listdir(TARGET_DIR) if f.endswith('.csv')])
        print(f"📁 最终文件数量检查: {file_count}")

if __name__ == '__main__':
    main()