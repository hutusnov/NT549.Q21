import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
import asyncio

async def test():
    url = "http://127.0.0.1:8000/ask"
    payload = {"query": "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận mãn tính thì nên điều trị như thế nào?"}
    
    print(f"Gửi câu hỏi: {payload['query']}")
    print("-" * 50)
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=30.0) as response:
                if response.status_code != 200:
                    print(f"Lỗi HTTP {response.status_code}: {await response.aread()}")
                    return
                
                async for line in response.aiter_lines():
                    if line:
                        print(line)
    except Exception as e:
        print(f"Lỗi kết nối: {e}")

if __name__ == "__main__":
    asyncio.run(test())
