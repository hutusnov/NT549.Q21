"""
train_worker.py — Online Training từ production logs

SỬA LỖI CHÍNH:
1. Dùng next_state THỰC TẾ từ log (không phải next_state = state)
2. done=False (continuing task, agent học hậu quả dài hạn)
3. Reward thuần outcome (không có decision_bonus)
4. GPT judge cho quality score, nhưng reward không encode policy
"""

import json
import asyncio
import httpx
import re
import numpy as np
import time
import os
from rl_agent import DQNAgent
from environment import NetworkEnvironment

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Thiếu OPENAI_API_KEY. Chạy: export OPENAI_API_KEY=sk-...")

LOG_FILE = "experience_log.jsonl"
MODEL_FILE = "dqn_model.pth"

agent = DQNAgent(state_dim=8, action_dim=2)
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


async def process_logs_and_train():
    if not os.path.exists(LOG_FILE):
        print("⚠️ Không tìm thấy log.")
        return

    print("🚀 BẮT ĐẦU HUẤN LUYỆN (OUTCOME-BASED REWARD VỚI GPT-4o-mini)...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    valid_experiences = 0
    for idx, line in enumerate(lines):
        data = json.loads(line)
        state_vector = np.array(data["state_vector"])
        action = data["action"]
        latency = data["latency"]

        # SỬA #1: Dùng next_state THỰC TẾ từ log
        next_state_vector = np.array(
            data.get("next_state_vector", data["state_vector"])
        )

        # SỬA #2: done từ log (mặc định False = continuing task)
        done = data.get("done", False)

        # GPT judge cho quality score
        q_score = await llm_judge(data["query"], data["response"])

        # SỬA #3: Reward thuần outcome — KHÔNG CÓ decision_bonus
        cost = 1.0 if action == 1 else 0.1
        reward = NetworkEnvironment.compute_reward_from_log(latency, q_score, cost)

        print(
            f"[{idx+1}] {data['routed_to'].upper():<5} | "
            f"GPT: {q_score:<4.1f} | Trễ: {latency:.1f}s | "
            f"Reward: {reward:.2f} | done={done}"
        )

        # SỬA #4: next_state ≠ state, done ≠ True luôn
        agent.remember(state_vector, action, reward, next_state_vector, done=done)
        valid_experiences += 1

        await asyncio.sleep(0.1)

    if valid_experiences >= agent.batch_size:
        print(f"\n🧠 Đang cập nhật trọng số Neural Network ({valid_experiences} experiences)...")
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
