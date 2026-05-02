import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
generate_log.py — Sinh demo log qua NetworkEnvironment

SỬA LỖI: Dùng environment cho state transitions thực sự.
Agent explore (ε=0.5) để có diverse data, không chỉ expert policy.
"""

import json
import random
from rl.environment import NetworkEnvironment
from rl.rl_agent import DQNAgent

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "demo_experience.jsonl")
NUM_EPISODES = 40
EPISODE_LENGTH = 50

print(f"Đang sinh {NUM_EPISODES * EPISODE_LENGTH} dòng log qua RL Environment...")

env = NetworkEnvironment(episode_length=EPISODE_LENGTH)
agent = DQNAgent(state_dim=8, action_dim=2)
agent.epsilon = 0.5  # 50% explore, 50% exploit random policy

SIMPLE_QUERIES = [
    "Bệnh gút là gì?",
    "Triệu chứng của gút là gì?",
    "Chế độ ăn kiêng cho bệnh nhân gút?",
    "Gút có chữa khỏi hoàn toàn được không?",
    "Bệnh gút có lây không?",
]
COMPLEX_QUERIES = [
    "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận mãn tính thì nên điều trị như thế nào?",
    "Chống chỉ định của allopurinol khi bệnh nhân có tiền sử dị ứng và suy thận nặng?",
    "Bệnh nhân gút cấp tính có biến chứng dịch khớp nhiều, cần xét nghiệm gì trước phẫu thuật?",
    "Phác đồ điều trị gút cho bệnh nhân cao tuổi có suy thận, tiểu đường?",
]

records = []

for ep in range(NUM_EPISODES):
    state = env.reset()

    for step in range(EPISODE_LENGTH):
        action = agent.get_action(state, explore=True)
        next_state, reward, done, info = env.step(action)

        is_complex = bool(state[1])
        query = random.choice(COMPLEX_QUERIES if is_complex else SIMPLE_QUERIES)

        records.append({
            "timestamp": ep * EPISODE_LENGTH + step,
            "episode": ep,
            "query": query,
            "response": "Nội dung giả lập...",
            "state_vector": state.tolist(),
            "next_state_vector": next_state.tolist(),
            "action": int(action),
            "routed_to": "cloud" if action == 1 else "edge",
            "reward": round(float(reward), 4),
            "latency": round(info["latency"], 3),
            "quality": round(info["quality"], 2),
            "is_complex": is_complex,
            "done": done,
            "mock_score": round(info["quality"], 1),
        })

        state = next_state
        if done:
            break

with open(LOG_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

complex_count = sum(1 for r in records if r["is_complex"])
edge_count = sum(1 for r in records if r["action"] == 0)
print(f"✅ Hoàn tất! Đã lưu {len(records)} records vào {LOG_FILE}.")
print(f"   Episodes: {NUM_EPISODES} × {EPISODE_LENGTH} steps")
print(f"   Complex: {complex_count} ({complex_count/len(records)*100:.1f}%)")
print(f"   Edge: {edge_count} ({edge_count/len(records)*100:.1f}%)")
