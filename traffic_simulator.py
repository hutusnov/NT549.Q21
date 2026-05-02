"""
traffic_simulator.py — Sinh traffic với STATE TRANSITIONS thực sự

SỬA LỖI CHÍNH:
1. Dùng NetworkEnvironment → state transition (action ảnh hưởng state sau)
2. Mỗi record có cả state VÀ next_state
3. Agent explore (ε-greedy), không chỉ expert policy
4. Output episodes, không phải independent records

Cách dùng:
    python traffic_simulator.py --episodes 100 --length 50
    python traffic_simulator.py --episodes 200 --length 100 --scenario burst
"""

import json
import random
import argparse
import numpy as np
from environment import NetworkEnvironment
from rl_agent import DQNAgent


def generate_episodes(num_episodes: int, episode_length: int,
                      scenario: str, explore_rate: float = 0.3) -> list[dict]:
    """Sinh data qua environment với state transitions thực sự."""
    env = NetworkEnvironment(episode_length=episode_length)
    records = []

    # Dùng agent để sinh data đa dạng (nếu có model)
    agent = DQNAgent(state_dim=8, action_dim=2)
    agent.epsilon = explore_rate  # Explore nhiều để có diverse data

    SIMPLE_QUERIES = [
        "Bệnh gút là gì?",
        "Triệu chứng của gút là gì?",
        "Chế độ ăn kiêng cho bệnh nhân gút?",
        "Gút có chữa khỏi hoàn toàn được không?",
        "Bệnh gút có lây không?",
    ]
    COMPLEX_QUERIES = [
        "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận mãn tính giai đoạn 3, đang dùng corticoid thì nên điều trị như thế nào?",
        "Chống chỉ định của allopurinol trong trường hợp bệnh nhân có tiền sử dị ứng và suy thận nặng là gì?",
        "Bệnh nhân gút cấp tính có biến chứng dịch khớp nhiều, bạch cầu cấp tăng cao, cần xét nghiệm gì trước phẫu thuật?",
        "Phác đồ điều trị gút cho bệnh nhân cao tuổi có suy thận, tiểu đường, đang dùng nhiều thuốc huyết áp?",
    ]

    for ep in range(num_episodes):
        state = env.reset()

        # Scenario: điều chỉnh environment
        if scenario == "burst":
            env.edge_cpu = random.uniform(0.6, 0.9)
            env.cloud_cpu = random.uniform(0.4, 0.7)
            env.edge_pending = random.randint(3, 8)
            state = env._get_state()
        elif scenario == "degraded":
            env.edge_lat = 5.0
            env.cloud_lat = random.uniform(3.0, 5.0)
            state = env._get_state()

        for step in range(episode_length):
            # Agent chọn action (có exploration)
            action = agent.get_action(state, explore=True)

            next_state, reward, done, info = env.step(action)

            is_complex = bool(state[1])
            query = random.choice(COMPLEX_QUERIES if is_complex else SIMPLE_QUERIES)

            records.append({
                "timestamp": ep * episode_length + step,
                "episode": ep,
                "step": step,
                "query": query,
                "response": "Nội dung giả lập...",
                "state_vector": state.tolist(),
                "next_state_vector": next_state.tolist(),
                "action": int(action),
                "routed_to": "cloud" if action == 1 else "edge",
                "reward": round(float(reward), 4),
                "latency": round(info["latency"], 3),
                "quality": round(info["quality"], 2),
                "cost": info["cost"],
                "is_complex": is_complex,
                "done": done,
                "scenario": scenario,
            })

            state = next_state
            if done:
                break

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--scenario", choices=["normal", "burst", "degraded"],
                        default="normal")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_file = args.out or f"traffic_{args.scenario}.jsonl"
    records = generate_episodes(args.episodes, args.length, args.scenario)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    complex_count = sum(1 for r in records if r["is_complex"])
    edge_count = sum(1 for r in records if r["action"] == 0)
    print(f"✅ Đã sinh {len(records)} records ({args.episodes} episodes × {args.length} steps)")
    print(f"   Scenario: {args.scenario}")
    print(f"   Câu phức tạp: {complex_count} ({complex_count/len(records)*100:.1f}%)")
    print(f"   Edge routing: {edge_count} ({edge_count/len(records)*100:.1f}%)")
    print(f"   Đã lưu vào: {out_file}")
