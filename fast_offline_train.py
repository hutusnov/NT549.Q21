"""
fast_offline_train.py — Offline RL Training qua Environment Simulator

SỬA LỖI CHÍNH:
1. Dùng NetworkEnvironment có state transition THỰC SỰ
2. Agent TỰ EXPLORE (ε-greedy), không bắt chước expert
3. Multi-step episodes (done=False trong episode, True ở cuối)
4. Reward thuần outcome từ environment, không hardcode policy
"""

import numpy as np
import time
from rl_agent import DQNAgent
from environment import NetworkEnvironment

MODEL_FILE = "dqn_model.pth"
NUM_EPISODES = 500
EPISODE_LENGTH = 50

agent = DQNAgent(state_dim=8, action_dim=2)
# Không load model cũ — học từ đầu với architecture mới
agent.epsilon = 1.0  # Bắt đầu với full exploration


def train_with_environment():
    env = NetworkEnvironment(episode_length=EPISODE_LENGTH)

    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN RL THỰC SỰ")
    print(f"   Episodes: {NUM_EPISODES} × {EPISODE_LENGTH} steps")
    print(f"   Agent tự explore và học từ hậu quả\n")

    all_rewards = []
    all_edge_ratios = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        edge_count = 0

        for step in range(EPISODE_LENGTH):
            # Agent TỰ CHỌN action bằng ε-greedy (explore)
            action = agent.get_action(state, explore=True)

            # Environment trả về NEXT STATE thực sự
            # (state transition: node bận hơn sau khi route)
            next_state, reward, done, info = env.step(action)

            # Lưu experience với next_state ≠ state
            agent.remember(state, action, reward, next_state, done)

            # Học từ experience
            agent.replay()

            state = next_state
            episode_reward += reward
            if action == 0:
                edge_count += 1

            if done:
                break

        all_rewards.append(episode_reward)
        edge_ratio = edge_count / EPISODE_LENGTH * 100
        all_edge_ratios.append(edge_ratio)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(all_rewards[-50:])
            avg_edge = np.mean(all_edge_ratios[-50:])
            print(
                f"Episode {ep+1:>4}/{NUM_EPISODES} | "
                f"Avg Reward: {avg_r:>7.2f} | "
                f"Edge%: {avg_edge:>5.1f}% | "
                f"ε: {agent.epsilon:.3f}"
            )

    agent.save_model(MODEL_FILE)
    print(f"\n✅ Đã lưu model vào {MODEL_FILE}")

    # In thống kê cuối
    print("\n📊 Kết quả training:")
    print(f"   Reward đầu (50 ep):  {np.mean(all_rewards[:50]):.2f}")
    print(f"   Reward cuối (50 ep): {np.mean(all_rewards[-50:]):.2f}")
    print(f"   Edge% đầu:  {np.mean(all_edge_ratios[:50]):.1f}%")
    print(f"   Edge% cuối: {np.mean(all_edge_ratios[-50:]):.1f}%")


if __name__ == "__main__":
    train_with_environment()
