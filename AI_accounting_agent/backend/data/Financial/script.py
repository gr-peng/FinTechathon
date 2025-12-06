import pandas as pd
import requests
import json
import time
import os

# --- 1. 配置 ---
TARGET_DIR = os.path.join('fintechcom', 'AI_accounting_agent', 'backend', 'data', '理财', 'A股')

# --- 2. 修复版 Ashare 核心函数 ---
# 这是一个替代 Ashare.get_price 的函数，解决了“7列数据报错”的问题
def get_price_safe(code, frequency='1d', count=250):
    try:
        # 统一代码格式
        code = code.lower()
        # 腾讯财经接口 (Ashare 原理)
        url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{count},qfq'
        response = requests.get(url, timeout=3)
        data = response.json()
        
        # 解析数据
        # 腾讯接口返回路径通常是 data[code]['qfqday'] 或 data[code]['day']
        if 'qfqday' in data['data'][code]:
            buf = data['data'][code]['qfqday']
        elif 'day' in data['data'][code]:
            buf = data['data'][code]['day']
        else:
            return None

        # --- 关键修复步骤 ---
        # 强制只取前6列 (日期, 开, 高, 低, 收, 量)
        # 解决部分股票返回7列导致 pandas 报错的问题
        buf_fixed = [item[:6] for item in buf]
        
        # 转换为 DataFrame
        df = pd.DataFrame(buf_fixed, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
        
    except Exception as e:
        print(f"    接口请求异常: {e}")
        return None

# --- 3. 完整的 TOP 100 股票名单 (已补全) ---
TOP_100_MAP = {
      
    'sh601211': '国泰君安'
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

    print(f"🚀 准备下载 {len(TOP_100_MAP)} 只股票数据...")
    print(f"📂 目标文件夹: {TARGET_DIR}")
    
    success_count = 0
    
    for code, name in TOP_100_MAP.items():
        try:
            # 使用我们自定义的 get_price_safe 函数
            df = get_price_safe(code, frequency='1d', count=250)
            
            if df is not None and not df.empty:
                df['code'] = code
                df['name'] = name
                
                # 构造保存路径
                file_name = f"{name}.csv"
                save_path = os.path.join(TARGET_DIR, file_name)
                
                # 保存
                df.to_csv(save_path, encoding='utf_8_sig')
                
                print(f"✅ [{success_count+1}/{len(TOP_100_MAP)}] {name} ({code})")
                success_count += 1
            else:
                print(f"⚠️ 数据为空或接口限制: {name} ({code})")
                
        except Exception as e:
            print(f"❌ 下载失败: {name} ({code}) - {e}")
        
        # 稍微延时
        time.sleep(0.1)

    print("-" * 30)
    print(f"🎉 任务完成！应下载 100 个，成功 {success_count} 个。")

if __name__ == '__main__':
    main()