import json
import os
import random
import datetime
import sqlite3
from pathlib import Path

# ============ 类别配置 ============

EXPENSE_CATEGORIES = {
    "餐饮": ["星巴克", "肯德基", "麦当劳", "海底捞", "喜茶", "瑞幸咖啡", "外卖", "全家便利店"],
    "出行": ["滴滴出行", "12306", "高德打车", "共享单车", "携程"],
    "购物": ["淘宝", "京东", "拼多多", "天猫超市", "唯品会"],
    "生活缴费": ["国家电网", "中国移动", "中国联通", "自来水公司", "燃气公司"],
    "娱乐": ["腾讯视频", "爱奇艺", "Bilibili", "Steam", "Netflix"],
    "其他": ["快递费", "打印店", "本地小店", "便利店"]
}

INCOME_SOURCES = {
    "工资": ["公司工资发放"],
    "奖金": ["绩效奖金", "季度奖金"],
    "副业": ["兼职收入", "外包项目"],
    "理财收益": ["余额宝收益", "基金分红"]
}

payment_methods = ["微信支付", "支付宝", "银行卡", "信用卡"]

# ============ 模板 ============

expense_templates = [
    "{date}在{merchant}消费了{amount}元，使用{payment}支付。",
    "{date}通过{payment}在{merchant}花了{amount}元。",
    "{date}我在{merchant}花费{amount}元（{payment}）。",
]

income_templates = [
    "{date}收到来自{merchant}的收入，共计{amount}元。",
    "{date}获得{merchant}发放的{amount}元入账。",
    "{date}账户增加收入{amount}元，来源：{merchant}。",
]

# ============ 月度分布（真实感更强） ============
# 每月消费次数权重
MONTH_EXPENSE_WEIGHTS = {
    1: 0.9, 2: 0.8, 3: 1.0, 4: 1.1, 5: 1.0, 6: 1.1,
    7: 0.9, 8: 0.8, 9: 1.2, 10: 1.3, 11: 1.1, 12: 1.4
}

# 每类支出占比（依据一般人花费结构）
CATEGORY_DISTRIBUTION = {
    "餐饮": 0.35,
    "出行": 0.10,
    "购物": 0.25,
    "生活缴费": 0.10,
    "娱乐": 0.15,
    "其他": 0.05,
}

# ============ 支出金额范围 ============
EXPENSE_AMOUNT_RANGE = {
    "餐饮": (20, 150),
    "出行": (3, 300),
    "购物": (20, 2000),
    "生活缴费": (50, 600),
    "娱乐": (10, 800),
    "其他": (5, 200),
}

# ============ 收入金额范围 ============
INCOME_AMOUNT_RANGE = {
    "工资": (8000, 20000),
    "奖金": (1000, 10000),
    "副业": (200, 3000),
    "理财收益": (5, 200),
}

# ============ 生成单条账单 ============

def generate_expense(date):
    category = random.choices(
        list(CATEGORY_DISTRIBUTION.keys()),
        list(CATEGORY_DISTRIBUTION.values())
    )[0]

    merchant = random.choice(EXPENSE_CATEGORIES[category])
    payment = random.choice(payment_methods)
    low, high = EXPENSE_AMOUNT_RANGE[category]
    amount = round(random.uniform(low, high), 2)

    template = random.choice(expense_templates)
    text = template.format(date=date, merchant=merchant, amount=amount, payment=payment)

    return text, {
        "日期": date,
        "金额": amount,
        "类别": category,
        "商户": merchant,
        "支付方式": payment,
        "描述": f"{category}消费",
        "type": "expense"
    }


def generate_income(date):
    category = random.choice(list(INCOME_SOURCES.keys()))
    merchant = random.choice(INCOME_SOURCES[category])
    low, high = INCOME_AMOUNT_RANGE[category]
    amount = round(random.uniform(low, high), 2)

    template = random.choice(income_templates)
    text = template.format(date=date, merchant=merchant, amount=amount)

    return text, {
        "日期": date,
        "金额": amount,
        "类别": category,
        "商户": merchant,
        "支付方式": "银行入账",
        "描述": f"{category}收入",
        "type": "income"
    }


# ============ 数据库 ============

def _ensure_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL,
            category TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT NOT NULL DEFAULT 'CNY',
            description TEXT NOT NULL,
            source TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


# ============ 主程序 ============

def main():

    TOTAL = 1000  # 一万条记录

    data_dir = Path(os.environ.get(
        "AI_BOOKKEEPER_DATA_DIR",
        Path(__file__).resolve().parent / "data"
    ))
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    rows_for_db = []

    today = datetime.date.today()

    for i in range(TOTAL):
        # —— 平均分到 365 天 —— #
        days_ago = random.randint(1, 365)
        date = (today - datetime.timedelta(days=days_ago)).isoformat()

        # 收入概率 7%（可调）
        if random.random() < 0.07:
            text, output = generate_income(date)
        else:
            text, output = generate_expense(date)

        dataset.append({
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}
            ]
        })

        # 写入数据库
        now = datetime.datetime.utcnow().isoformat()
        rows_for_db.append((
            date,
            output["类别"],
            output["type"],
            float(output["金额"]),
            "CNY",
            output["描述"],
            "synthetic",
            json.dumps(output, ensure_ascii=False),
            now
        ))

    # 写 JSONL
    jsonl_path = data_dir / "synthetic_bank_bills.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写入 SQLite
    db_path = data_dir / "bills.db"
    conn = sqlite3.connect(db_path)
    _ensure_db(conn)
    conn.executemany("""
        INSERT INTO bills
        (event_date, category, type, amount, currency, description, source, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_for_db)
    conn.commit()
    conn.close()

    print("✅ 数据生成完毕")
    print(" - JSONL:", jsonl_path)
    print(" - SQLite:", db_path)
    print(" - 总记录数:", TOTAL)


if __name__ == "__main__":
    main()
