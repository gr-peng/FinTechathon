#!/usr/bin/env python3
"""Generate user persona and personalized reply by calling local /chat with bills context."""

import json
from pathlib import Path
import requests

API_BASE = "http://localhost:8010"
CHAT_ENDPOINT = f"{API_BASE}/chat"
BILLS_PATH = Path("/home/bld/data/data4/admin/fintechcom/AI_accounting_agent/backend/data/test.jsonl")
OUTPUT_PATH = Path("/home/bld/data/data4/admin/fintechcom/AI_accounting_agent/backend/profile_output.txt")

SYSTEM_PROMPT = (
    "你是财务分析助手。你会读取用户最近的消费账单，先在内部形成用户画像，"
    "再据此给出个性化回答，但不要直接说‘你的画像是……’。"
)

USER_TEMPLATE = """账单数据如下（JSONL，每行一条记录）：\n{bills}\n\n请根据用户最近的消费账单生成如下格式的用户画像，并据此给出个性化回答：\n\n# User Profile Context\n**1. 基本人口统计 (Demographics)**\n- 年龄/性别/职业：[ ______________ ]\n- 居住地与居住形式：[ ______________ ]\n- 通勤方式与车辆情况：[ ______________ ]\n\n**2. 财务状况 (Financial Status)**\n- 收入水平与稳定性：[ ______________ ]\n- 月度结余与资金流向：[ ______________ ]\n- 主要固定支出项目：[ ______________ ]\n\n**3. 生活方式 (Lifestyle)**\n- 兴趣爱好与休闲方式：[ ______________ ]\n- 饮食习惯与时间偏好：[ ______________ ]\n- 社交频率与偏好：[ ______________ ]\n\n**4. 消费特征 (Consumption Habits)**\n- 主要消费类别占比：[ ______________ ]\n- 特殊/大额消费习惯：[ ______________ ]\n\n**5. 现状评估 (Current Assessment)**\n- 财务健康/抗风险能力：[ ______________ ]\n- 健康风险或需关注点：[ ______________ ]\n- 潜在机会与改进方向：[ ______________ ]\n\n个性化要求：无论是建议还是与用户对话，都要在内部基于用户画像和账单上下文进行定制化回答，但不要直接透露或复述‘你的画像是……’这类措辞，只呈现个性化建议和语气。"""


def load_bills_text() -> str:
    if not BILLS_PATH.exists():
        raise FileNotFoundError(f"账单文件不存在: {BILLS_PATH}")
    return BILLS_PATH.read_text(encoding="utf-8")


def build_payload(bills_text: str) -> dict:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "include_bill_context": False,  # 避免后端再注入最近账单，直接用本地文件
        "messages": [
            {"role": "user", "content": USER_TEMPLATE.format(bills=bills_text)},
        ],
        "temperature": 0.4,
        "max_new_tokens": 800,
    }


def call_chat(payload: dict) -> str:
    resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("reply") or data


def main() -> None:
    bills_text = load_bills_text()
    payload = build_payload(bills_text)
    reply = call_chat(payload)
    OUTPUT_PATH.write_text(str(reply), encoding="utf-8")
    print(reply)
    print(f"\n已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
