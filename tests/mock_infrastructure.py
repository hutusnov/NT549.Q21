import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json
import random
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Mock Infrastructure")

# Trạng thái giả lập
edge_pending = 0
cloud_pending = 0

async def mock_llm_stream(node_name: str, base_lat: float, req: Request):
    global edge_pending, cloud_pending
    
    if node_name == "edge":
        edge_pending += 1
        lat = base_lat + edge_pending * 1.5
    else:
        cloud_pending += 1
        lat = base_lat + cloud_pending * 0.5
        
    try:
        try:
            body = await req.json()
        except Exception:
            body = {}
        prompt = body.get("prompt", "")
        
        # Mô phỏng độ trễ sinh token
        words = f"Đây là câu trả lời giả lập từ {node_name.upper()} cho câu hỏi: '{prompt[:30]}...'".split()
        
        # Chờ time to first token
        await asyncio.sleep(lat * 0.3)
        
        for w in words:
            await asyncio.sleep((lat * 0.7) / len(words))
            yield json.dumps({"response": w + " "}, ensure_ascii=False) + "\n"
            
    finally:
        if node_name == "edge":
            edge_pending = max(0, edge_pending - 1)
        else:
            cloud_pending = max(0, cloud_pending - 1)

@app.post("/edge/api/generate")
async def edge_api(req: Request):
    return StreamingResponse(mock_llm_stream("edge", 1.0, req), media_type="application/x-ndjson")

@app.post("/cloud/api/generate")
async def cloud_api(req: Request):
    return StreamingResponse(mock_llm_stream("cloud", 3.0, req), media_type="application/x-ndjson")

@app.get("/api/v1/query")
async def prometheus_api(query: str):
    # Fake CPU load từ 0.1 đến 0.9 tuỳ vào số request đang xử lý
    global edge_pending, cloud_pending
    
    if "100.73.54.43" in query:  # Edge
        val = min(0.95, 0.1 + edge_pending * 0.15 + random.uniform(0, 0.05))
    elif "100.110.167.50" in query:  # Cloud
        val = min(0.90, 0.1 + cloud_pending * 0.05 + random.uniform(0, 0.05))
    else:
        val = 0.5
        
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {},
                    "value": [time.time(), str(val)]
                }
            ]
        }
    }

if __name__ == "__main__":
    print("🚀 Khởi chạy Mock Infrastructure (Ollama + Prometheus)...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
