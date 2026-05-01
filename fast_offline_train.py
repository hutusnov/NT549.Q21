import json
import numpy as np
import time
import os
from rl_agent import DQNAgent
from train_worker import compute_reward  # Single source of truth cho reward

LOG_FILE = "demo_experience.jsonl"
MODEL_FILE = "dqn_model_demo.pth"

agent = DQNAgent(state_dim=6, action_dim=2)
# Bỏ load_model để học từ đầu với fake data

def train_fast():
    if not os.path.exists(LOG_FILE):
        print("⚠️ Không tìm thấy file log.")
        return

    print("🚀 BẮT ĐẦU ÉP XUNG HUẤN LUYỆN (OFFLINE RL)...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    valid_experiences = 0
    for idx, line in enumerate(lines):
        data = json.loads(line)
        state_vector = np.array(data["state_vector"])
        action = data["action"]
        latency = data["latency"]
        is_complex = data.get("is_complex", False)

        q_score = data.get("mock_score", 5.0)
        normalized_score = 10.0 if q_score >= 7.0 else q_score

        reward = compute_reward(action, latency, is_complex, normalized_score)

        agent.remember(state_vector, action, reward, state_vector, done=True)
        valid_experiences += 1

    if valid_experiences >= agent.batch_size:
        print(f"Đang nhồi {valid_experiences} kinh nghiệm vào mạng Nơ-ron...")
        for _ in range(30 * (valid_experiences // agent.batch_size)):
            agent.replay()

        agent.save_model(MODEL_FILE)
        print(f"✅ Đã lưu bộ não vào {MODEL_FILE}")
        os.rename(LOG_FILE, f"used_demo_log_{int(time.time())}.jsonl")
    else:
        print("Lỗi: Không đủ data.")

if __name__ == "__main__":
    train_fast()
