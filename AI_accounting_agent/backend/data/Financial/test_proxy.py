import requests
import os

print("Testing with proxy from env...")
print(f"http_proxy: {os.environ.get('http_proxy')}")
print(f"https_proxy: {os.environ.get('https_proxy')}")

url = "https://www.baidu.com"
try:
    r = requests.get(url, timeout=10)
    print(f"Status Code: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
