import json
import asyncio
import httpx
import re
import numpy as np
import time
import os
from rl_agent import DQNAgent

# FIX P0: Đọc API key từ biến môi trường, không hardcode
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Thiếu OPENAI_API_KEY. Chạy: export OPENAI_API_KEY=sk-...")

LOG_FILE = "experience_log.jsonl"
MODEL_FILE = "dqn_model.pth"

agent = DQNAgent(state_dim=6, action_dim=2)
agent.load_model(MODEL_FILE)

async def llm_judge(query: str, response: str) -> float:
    if response.startswith("Error:") or "Lỗi từ Ollama" in response or "quá tải" in response:
        return 1.0

    prompt = (
        "Chấm điểm câu trả lời y tế sau từ 1 đến 10 dựa trên độ chính xác, an toàn và hữu ích. "
        "Chỉ in ra 1 con số duy nhất, không giải thích thêm.\n"
        f"Câu hỏi: {query}\nTrả lời: {response}"
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    # FIX: Retry với exponential backoff để tránh 429
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
            if res.status_code == 200:
                text_result = res.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r'\d+(\.\d+)?', text_result)
                if match:
                    return min(max(float(match.group()), 1.0), 10.0)
            elif res.status_code == 429:
                wait = 2 ** attempt
                print(f"⚠️ Rate limit, chờ {wait}s...")
                await asyncio.sleep(wait)
                continue
            else:
                print(f"⚠️ [OpenAI Error] HTTP {res.status_code}: {res.text}")
                break
        except Exception as e:
            print(f"⚠️ [LLM Judge Exception] attempt {attempt+1}: {e}")
            await asyncio.sleep(1)

    return 5.0

def compute_reward(action: int, latency: float, is_complex: bool, normalized_score: float) -> float:
    """
    Hàm reward DUY NHẤT — dùng chung cho cả train_worker và fast_offline_train.
    Đặt ở đây để import, không copy-paste.
    """
    if action == 0:  # EDGE (SLA 15s)
        latency_penalty = latency * 0.5 if latency <= 15.0 else latency * 1.2
    else:            # CLOUD (SLA 5s)
        latency_penalty = latency * 1.0 if latency <= 5.0 else latency * 3.0

    cost_penalty = 15.0 if action == 1 else 0.0

    # FIX P1: Thống nhất với fast_offline_train — thưởng Edge đúng cho câu dễ
    decision_bonus = 0
    if is_complex and action == 1:
        decision_bonus = 30.0
    elif is_complex and action == 0:
        decision_bonus = -50.0
    elif not is_complex and action == 0:
        decision_bonus = 30.0
    # not is_complex + cloud: không thưởng, không phạt (ngoài cost_penalty)

    return (5.0 * normalized_score) - latency_penalty - cost_penalty + decision_bonus

async def process_logs_and_train():
    if not os.path.exists(LOG_FILE):
        print("⚠️ Không tìm thấy log.")
        return

    print("🚀 BẮT ĐẦU HUẤN LUYỆN (OPTIMIZED REWARD VỚI GPT-4o-mini)...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    valid_experiences = 0
    for idx, line in enumerate(lines):
        data = json.loads(line)
        state_vector = np.array(data["state_vector"])
        action = data["action"]
        latency = data["latency"]
        is_complex = data.get("is_complex", False)

        q_score = await llm_judge(data["query"], data["response"])
        # "normalized_score" thực chất là clip một chiều — giữ tên cũ để khỏi confuse log
        normalized_score = 10.0 if q_score >= 7.0 else q_score

        reward = compute_reward(action, latency, is_complex, normalized_score)

        print(f"[{idx+1}] {data['routed_to'].upper():<5} | GPT: {q_score:<4.1f} -> {normalized_score} | Trễ: {latency:.1f}s | Reward: {reward:.2f}")
        agent.remember(state_vector, action, reward, state_vector, done=True)
        valid_experiences += 1

        # FIX: Throttle để tránh rate limit (0.1s/call ~ 600 call/phút, dưới limit 4o-mini)
        await asyncio.sleep(0.1)

    if valid_experiences >= agent.batch_size:
        print(f"\n🧠 Đang cập nhật trọng số Neural Network...")
        for _ in range(15 * (valid_experiences // agent.batch_size)):
            agent.replay()
        agent.save_model(MODEL_FILE)

        backup_name = f"backup_log_{int(time.time())}.jsonl"
        os.rename(LOG_FILE, backup_name)
        print(f"✅ Đã huấn luyện xong và sao lưu log sang {backup_name}")
    else:
        print("⚠️ Chưa đủ data.")

if __name__ == "__main__":
    asyncio.run(process_logs_and_train())
