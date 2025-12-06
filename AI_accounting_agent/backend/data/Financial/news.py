#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Date: 2025/11/20 120:15
Desc: 个股新闻数据
https://so.eastmoney.com/news/s?keyword=603777
"""

import argparse
import json
import random
import time

import pandas as pd
import requests


def stock_news_em(symbol: str = "603777", page_index: int = 1, page_size: int = 10) -> pd.DataFrame:
    """
    东方财富-个股新闻-最近 100 条新闻
    https://so.eastmoney.com/news/s?keyword=603777
    :param symbol: 股票代码
    :type symbol: str
    :return: 个股新闻
    :rtype: pandas.DataFrame
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    if not symbol:
        raise ValueError("symbol must not be empty")
    if page_index < 1:
        raise ValueError("page_index must be >= 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    keyword = symbol.strip()

    inner_param = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": page_index,
                "pageSize": page_size,
                "preTag": "<em>",
                "postTag": "</em>"
            }
        }
    }
    timestamp_ms = int(time.time() * 1000)
    callback = f"jQuery3510{random.randint(10**11, 10**12 - 1)}_{timestamp_ms}"
    params = {
        "cb": callback,
        "param": json.dumps(inner_param, ensure_ascii=False),  # 保留中文,
        "_": str(timestamp_ms)
    }
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "cache-control": "no-cache",
        "connection": "keep-alive",
        "host": "search-api-web.eastmoney.com",
        "pragma": "no-cache",
        "referer": f"https://so.eastmoney.com/news/s?keyword={keyword}",
        "sec-ch-ua": "\"Chromium\";v=\"142\", \"Google Chrome\";v=\"142\", \"Not_A Brand\";v=\"99\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "script",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    }
    r = requests.get(url, params=params, headers=headers)
    data_text = r.text.strip()
    if data_text.startswith("/**/"):
        data_text = data_text[4:].lstrip()

    jsonp_prefix = f"{callback}("
    if not data_text.startswith(jsonp_prefix):
        raise ValueError("Unexpected JSONP callback format")

    payload = data_text[len(jsonp_prefix):].rstrip()
    if payload.endswith(");"):
        payload = payload[:-2].rstrip()
    elif payload.endswith(")"):
        payload = payload[:-1].rstrip()
    else:
        raise ValueError("Unexpected JSONP closing pattern")

    data_json = json.loads(payload)
    temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])
    temp_df["url"] = "http://finance.eastmoney.com/a/" + temp_df["code"] + ".html"
    temp_df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "code": "-",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
            "image": "-",
        },
        inplace=True,
    )
    temp_df["关键词"] = symbol
    temp_df = temp_df[
        [
            "关键词",
            "新闻标题",
            "新闻内容",
            "发布时间",
            "文章来源",
            "新闻链接",
        ]
    ]
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\u3000", "", regex=True)
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\r\n", " ", regex=True)
    return temp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Eastmoney news for a stock symbol.")
    parser.add_argument("symbol", nargs="?", default="603777", help="Stock symbol, e.g., 603777")
    parser.add_argument("--page-index", type=int, default=1, dest="page_index", help="Page index, default 1")
    parser.add_argument("--page-size", type=int, default=10, dest="page_size", help="Entries per page (1-100)")
    cli_args = parser.parse_args()
    stock_news_em_df = stock_news_em(
        symbol=cli_args.symbol,
        page_index=cli_args.page_index,
        page_size=cli_args.page_size,
    )
    print(stock_news_em_df)