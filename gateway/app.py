import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# app.py — RL-based Medical LLM Gateway
#
# SỬA LỖI CHÍNH:
# 1. KHÔNG CÒN rule-based override — Agent quyết định 100%
# 2. State 8-dim bao gồm pending requests (state transition thực sự)
# 3. Log (state, action, reward, next_state, done=False) đúng chuẩn RL
# 4. Reward thuần outcome (latency + quality proxy), không hardcode policy

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
from gateway.feature_extractor import analyze_query
from rl.rl_agent import DQNAgent
from rl.environment import NetworkEnvironment
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter("gateway_requests_total", "Tổng số request đã nhận", ["routed_to"])
LATENCY_HIST = Histogram("gateway_request_latency_seconds", "Độ trễ thực tế của request", ["routed_to"])

# Network state tracking — KEY cho state transitions
history_latency = {
    "edge": 5.0,
    "cloud": 3.5
}
pending_requests = {
    "edge": 0,
    "cloud": 0
}

log_lock = asyncio.Lock()
latency_lock = asyncio.Lock()
pending_lock = asyncio.Lock()

http_client: httpx.AsyncClient = None
app = FastAPI(title="Medical LLM Gateway (RL-based)")

EDGE_NODE_URL = os.environ.get("EDGE_NODE_URL", "http://100.73.54.43:11434/api/generate")
CLOUD_NODE_URL = os.environ.get("CLOUD_NODE_URL", "http://100.110.167.50:11434/api/generate")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090/api/v1/query")

agent = DQNAgent(state_dim=8, action_dim=2)
agent.load_model(os.path.join(os.path.dirname(__file__), "..", "models", "dqn_model.pth"))


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
                        "prompt": "Hi",
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


async def build_state(nlp_state, edge_cpu, cloud_cpu):
    """Build 8-dim state vector, bao gồm pending request counts."""
    async with latency_lock:
        edge_lat = history_latency["edge"]
        cloud_lat = history_latency["cloud"]
    async with pending_lock:
        edge_pend = pending_requests["edge"]
        cloud_pend = pending_requests["cloud"]

    return np.array([
        nlp_state["critical_count"],
        1.0 if nlp_state["is_complex"] else 0.0,
        min(edge_lat, 5.0),
        min(cloud_lat, 5.0),
        edge_cpu,
        cloud_cpu,
        min(edge_pend / 10.0, 1.0),
        min(cloud_pend / 10.0, 1.0),
    ], dtype=np.float32)


async def write_log_safe(log_entry: dict):
    async with log_lock:
        async with aiofiles.open(os.path.join(os.path.dirname(__file__), "..", "data", "experience_log.jsonl"), mode="a", encoding="utf-8") as f:
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

    # Build state TRƯỚC khi chọn action
    state_vector = await build_state(nlp_state, edge_cpu, cloud_cpu)

    # AGENT QUYẾT ĐỊNH 100% — không có rule override
    action = agent.get_action(state_vector, explore=False)

    if action == 1:
        target_url, routed_to, model_name, num_predict = CLOUD_NODE_URL, "cloud", "qwen2:7b", 512
    else:
        target_url, routed_to, model_name, num_predict = EDGE_NODE_URL, "edge", "qwen2:1.5b-instruct-q8_0", 150

    # STATE TRANSITION: tăng pending count (ảnh hưởng state cho request sau)
    async with pending_lock:
        pending_requests[routed_to] += 1

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

        # STATE TRANSITION: giảm pending count sau khi response xong
        async with pending_lock:
            pending_requests[routed_to] = max(0, pending_requests[routed_to] - 1)

        # Build NEXT STATE sau khi action đã thực thi xong
        next_edge_cpu, next_cloud_cpu = await asyncio.gather(
            get_real_cpu_load("100.73.54.43"),
            get_real_cpu_load("100.110.167.50")
        )
        next_state_vector = await build_state(nlp_state, next_edge_cpu, next_cloud_cpu)

        # REWARD thuần outcome — không hardcode policy
        cost = 1.0 if action == 1 else 0.1
        quality_proxy = min(len(full_response_text) / 100.0, 10.0)  # Proxy cho chất lượng
        reward = NetworkEnvironment.compute_reward_from_log(llm_latency, quality_proxy, cost)

        print("\n" + "="*50)
        print(f"Câu hỏi     : {request.query[:30]}...")
        print(f"Đặc trưng   : Phức tạp={nlp_state['is_complex']} | Rủi ro: {nlp_state['critical_count']}")
        print(f"Trễ định tuyến: {routing_latency_ms:.2f} ms")
        print(f"Trễ LLM        : {llm_latency:.3f} s")
        print(f"Reward      : {reward:.3f}")
        print(f"Định tuyến  : {routed_to.upper()}")
        print("="*50 + "\n")

        background_tasks.add_task(update_latency_history, routed_to, llm_latency)

        # Log đầy đủ (state, action, reward, NEXT_STATE, done=False)
        background_tasks.add_task(write_log_safe, {
            "timestamp": time.time(),
            "query": request.query,
            "response": full_response_text,
            "state_vector": state_vector.tolist(),
            "next_state_vector": next_state_vector.tolist(),
            "action": int(action),
            "routed_to": routed_to,
            "reward": round(reward, 4),
            "latency": llm_latency,
            "quality_proxy": round(quality_proxy, 2),
            "routing_latency_ms": round(routing_latency_ms, 2),
            "is_complex": nlp_state["is_complex"],
            "done": False
        })

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")
