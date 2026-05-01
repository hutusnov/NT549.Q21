# app.py (ACTIVE PROBE + DYNAMIC STRATEGY + FIXED LOG)
from fastapi import FastAPI, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import time
import asyncio
import numpy as np
import json
import aiofiles
import os
from feature_extractor import analyze_query
from rl_agent import DQNAgent
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter("gateway_requests_total", "Tổng số request đã nhận", ["routed_to"])
LATENCY_HIST = Histogram("gateway_request_latency_seconds", "Độ trễ thực tế của request", ["routed_to"])

TEST_STRATEGY = os.environ.get("ROUTING_STRATEGY", "DQN")
round_robin_counter = 0

history_latency = {
    "edge": 5.0,    # 1.5B model: ~3-8s
    "cloud": 3.5    # 7B model: ~2-5s
}

log_lock = asyncio.Lock()
latency_lock = asyncio.Lock()

http_client: httpx.AsyncClient = None
app = FastAPI(title="Medical LLM Gateway (Self-Healing)")

EDGE_NODE_URL = "http://100.73.54.43:11434/api/generate"
CLOUD_NODE_URL = "http://100.110.167.50:11434/api/generate"
PROMETHEUS_URL = "http://localhost:9090/api/v1/query" 

agent = DQNAgent(state_dim=6, action_dim=2)
agent.load_model("dqn_model.pth")

class MedicalQuery(BaseModel):
    query: str

# --- ACTIVE PROBING ---
async def network_monitor_probe():
    print("Robot thám thính mạng đã bắt đầu hoạt động...")
    while True:
        try:
            for node, url in [("edge", EDGE_NODE_URL), ("cloud", CLOUD_NODE_URL)]:
                probe_start = time.time()
                try:
                    model_name = "qwen2:1.5b-instruct-q8_0" if node == "edge" else "qwen2:7b"
                    payload = {
                        "model": model_name,
                        "prompt": "Hi",  # Dummy prompt siêu ngắn
                        "stream": False,
                        "options": {"num_predict": 5}
                    }
                    
                    resp = await http_client.post(url, json=payload, timeout=15.0)
                    
                    real_lat = time.time() - probe_start
                    
                    if resp and resp.status_code == 200:
                        async with latency_lock:
                            history_latency[node] = (0.5 * real_lat) + (0.5 * history_latency[node])
                    else:
                        async with latency_lock: 
                            history_latency[node] = 30.0 
                except Exception as e:
                    async with latency_lock:
                        history_latency[node] = 30.0
                    print(f"⚠️Probe {node} failed: {e}")
                        
            await asyncio.sleep(15) 
        except Exception as e:
            print(f"⚠️Monitor loop error: {e}")
            await asyncio.sleep(15)

@app.on_event("startup")
async def startup_event():
    global http_client
    # Mở sẵn bể chứa 200 kết nối cùng lúc
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
    http_client = httpx.AsyncClient(limits=limits)
    
    asyncio.create_task(network_monitor_probe())

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/set-strategy")
async def set_strategy(strategy: str):
    global TEST_STRATEGY
    if strategy not in ["DQN", "RoundRobin"]:
        return {"error": f"Unknown strategy: {strategy}"}
    TEST_STRATEGY = strategy
    print(f"✅ Đã đổi strategy thành {strategy}")
    return {"status": "ok", "strategy": TEST_STRATEGY}

async def get_real_cpu_load(instance_ip: str) -> float:
    query = f'1 - avg(rate(node_cpu_seconds_total{{mode="idle", instance="{instance_ip}:9100"}}[1m]))'
    try:
        res = await http_client.get(PROMETHEUS_URL, params={"query": query}, timeout=2.0)
        if res.status_code != 200:
            return 0.5
        data = res.json()
            
        if data and data.get("status") == "success":
            d = data.get("data")
            if d and d.get("result") and len(d["result"]) > 0:
                val = d["result"][0].get("value")
                if val and len(val) > 1:
                    return round(float(val[1]), 3)
    except Exception as e:
        print(f"⚠️Prometheus CPU Error ({instance_ip}): {e}")
    return 0.5 

async def write_log_safe(log_entry: dict):
    async with log_lock:
        async with aiofiles.open("experience_log.jsonl", mode="a", encoding="utf-8") as f:
            await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

async def update_latency_history(routed_to: str, total_latency: float):
    async with latency_lock:
        ALPHA = 0.3
        history_latency[routed_to] = (ALPHA * total_latency) + ((1 - ALPHA) * history_latency[routed_to])

@app.post("/ask")
async def ask_medical_question(request: MedicalQuery, background_tasks: BackgroundTasks):
    routing_start = time.time()
    nlp_state = analyze_query(request.query.strip())

    edge_cpu, cloud_cpu = await asyncio.gather(
        get_real_cpu_load("100.73.54.43"),
        get_real_cpu_load("100.110.167.50")
    )

    async with latency_lock:
        current_edge_lat = history_latency["edge"]
        current_cloud_lat = history_latency["cloud"]

    state_vector = [
        nlp_state["critical_count"],
        1.0 if nlp_state["is_complex"] else 0.0,
        min(current_edge_lat, 5.0),
        min(current_cloud_lat, 5.0),
        edge_cpu,
        cloud_cpu
    ]

    global round_robin_counter
    if TEST_STRATEGY == "RoundRobin":
        action = round_robin_counter % 2
        round_robin_counter += 1
    else:
        action = agent.get_action(np.array(state_vector))

        if current_cloud_lat > 5.0 and not nlp_state["is_complex"]:
            print(f"⚠️ [ADAPTIVE] Cloud bị nghẽn ({current_cloud_lat:.2f}s). Ép câu dễ về Edge!")
            action = 0
        elif nlp_state["is_complex"]:
            print("🛡️ [GUARDRAIL] Câu khó — đẩy lên Cloud 7B.")
            action = 1

    if action == 1:
        target_url, routed_to, model_name, num_predict = CLOUD_NODE_URL, "cloud", "qwen2:7b", 512
    else:
        target_url, routed_to, model_name, num_predict = EDGE_NODE_URL, "edge", "qwen2:1.5b-instruct-q8_0", 150

    REQUEST_COUNT.labels(routed_to=routed_to).inc()
    routing_latency_ms = (time.time() - routing_start) * 1000

    async def generate_response():
        full_response_text = ""
        yield json.dumps({"type": "meta", "action_taken": routed_to, "is_complex": nlp_state["is_complex"]}, ensure_ascii=False) + "\n"
        
        llm_start = time.time()
        try:
            payload = {"model": model_name, "prompt": request.query, "stream": True, "options": {"num_predict": num_predict, "temperature": 0.7}}
            
            async with http_client.stream("POST", target_url, json=payload, timeout=120.0) as response:
                async for line in response.aiter_lines():
                    if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                full_response_text += token
                                yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield json.dumps({"type": "error", "content": f"Connection error: {str(e)}"}, ensure_ascii=False) + "\n"

        llm_latency = time.time() - llm_start
        LATENCY_HIST.labels(routed_to=routed_to).observe(llm_latency)
        
        async with latency_lock:
            edge_lat_display = history_latency["edge"]
            cloud_lat_display = history_latency["cloud"]

        print("\n" + "="*50)
        print(f"Câu hỏi     : {request.query[:30]}...")
        print(f"Đặc trưng   : Phức tạp={nlp_state['is_complex']} | Rủi ro: {nlp_state['critical_count']}")
        print(f"Trễ định tuyến: {routing_latency_ms:.2f} ms")
        print(f"Trễ LLM        : {llm_latency:.3f} s")
        print(f"Trễ AI nhớ  : Edge {edge_lat_display:.3f}s | Cloud {cloud_lat_display:.3f}s")
        print(f"Định tuyến  : {routed_to.upper()}")
        print("="*50 + "\n")

        background_tasks.add_task(update_latency_history, routed_to, llm_latency)
        
        # ĐÃ SỬA LỖI Ở ĐÂY: Thêm lại response và state_vector để AI có thể học
        background_tasks.add_task(write_log_safe, {
            "timestamp": time.time(),
            "query": request.query,
            "response": full_response_text,   # <--- QUAN TRỌNG NHẤT
            "state_vector": state_vector,     # <--- QUAN TRỌNG NHẤT
            "action": int(action),
            "routed_to": routed_to,
            "latency": llm_latency,
            "routing_latency_ms": round(routing_latency_ms, 2),
            "is_complex": nlp_state["is_complex"]
        })

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")
