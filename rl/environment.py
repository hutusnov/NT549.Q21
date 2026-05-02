import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
environment.py - Network Routing Environment cho RL

BẢN CHẤT RL: Action ảnh hưởng state tương lai.
- Route đến Edge → Edge bận hơn → latency/CPU tăng cho request SAU
- Agent phải học CÂN BẰNG TẢI qua nhiều step, không chỉ tối ưu từng request

State vector (8-dim):
  [critical_count, is_complex, edge_lat, cloud_lat,
   edge_cpu, cloud_cpu, edge_pending_norm, cloud_pending_norm]
"""

import numpy as np
import random


class NetworkEnvironment:
    """
    Simulates a 2-node network (Edge 1.5B + Cloud 7B).
    Mỗi step = 1 request đến. Episode = N requests liên tiếp.
    """

    def __init__(self, episode_length=50):
        self.episode_length = episode_length
        self.step_count = 0
        self.reset()

    def reset(self):
        """Reset về trạng thái ban đầu cho episode mới."""
        self.step_count = 0
        self.edge_cpu = random.uniform(0.1, 0.3)
        self.cloud_cpu = random.uniform(0.1, 0.2)
        self.edge_lat = random.uniform(1.0, 3.0)
        self.cloud_lat = random.uniform(1.0, 2.5)
        self.edge_pending = 0
        self.cloud_pending = 0
        self._generate_query()
        return self._get_state()

    def _generate_query(self):
        """Sinh query ngẫu nhiên cho step tiếp theo."""
        self.is_complex = random.random() < 0.3
        self.critical_count = random.randint(1, 3) if self.is_complex else 0

    def _get_state(self):
        return np.array([
            self.critical_count,
            1.0 if self.is_complex else 0.0,
            min(self.edge_lat, 5.0),
            min(self.cloud_lat, 5.0),
            min(self.edge_cpu, 1.0),
            min(self.cloud_cpu, 1.0),
            min(self.edge_pending / 10.0, 1.0),
            min(self.cloud_pending / 10.0, 1.0),
        ], dtype=np.float32)

    def step(self, action: int):
        """
        Thực hiện action, trả về (next_state, reward, done, info).

        STATE TRANSITION: action route đến node → node bận hơn →
        latency tăng cho request tiếp theo. Đây là điểm khác biệt
        cốt lõi so với contextual bandit.
        """
        self.step_count += 1

        # --- Tính outcome của action ---
        if action == 0:  # Edge (1.5B) - Yếu, chạy qua VPN
            # Base latency của Edge rất cao (16 - 20s)
            base_lat = 18.0 + self.edge_pending * 3.0
            cpu_factor = 1.0 + self.edge_cpu
            complexity_factor = 1.2 if self.is_complex else 1.0
            latency = base_lat * cpu_factor * complexity_factor
            latency += random.uniform(-2.0, 2.0)
            latency = max(5.0, latency)

            # Chất lượng: Câu dễ = 9.0 (rất tốt), Câu khó = 4.0 (rất tệ)
            quality = random.uniform(3.0, 5.0) if self.is_complex else random.uniform(8.5, 9.5)
            cost = 0.0  # Chạy ở Edge miễn phí
        else:  # Cloud (7B) - Nhanh, mạnh nhưng tốn tiền
            # Base latency của Cloud thấp hơn (4 - 6s)
            base_lat = 5.0 + self.cloud_pending * 1.0
            cpu_factor = 1.0 + self.cloud_cpu * 0.5
            latency = base_lat * cpu_factor
            latency += random.uniform(-1.0, 1.0)
            latency = max(2.0, latency)

            # Chất lượng luôn tốt
            quality = random.uniform(8.5, 10.0)
            cost = 1.0  # Chi phí Cloud hợp lý hơn để Agent dám dùng

        # --- STATE TRANSITION: action ảnh hưởng state tương lai ---
        if action == 0:
            self.edge_pending += 1
            self.edge_cpu = min(0.99, self.edge_cpu + 0.05 + self.edge_pending * 0.02)
            self.edge_lat = 0.3 * latency + 0.7 * self.edge_lat
        else:
            self.cloud_pending += 1
            self.cloud_cpu = min(0.95, self.cloud_cpu + 0.03 + self.cloud_pending * 0.01)
            self.cloud_lat = 0.3 * latency + 0.7 * self.cloud_lat

        # Một số request cũ hoàn thành → giảm tải
        if self.edge_pending > 0 and random.random() < 0.6:
            self.edge_pending -= 1
            self.edge_cpu = max(0.1, self.edge_cpu - 0.03)
        if self.cloud_pending > 0 and random.random() < 0.7:
            self.cloud_pending -= 1
            self.cloud_cpu = max(0.1, self.cloud_cpu - 0.02)

        # CPU tự hồi phục dần
        self.edge_cpu = max(0.1, self.edge_cpu * 0.98)
        self.cloud_cpu = max(0.1, self.cloud_cpu * 0.98)

        # --- REWARD: thuần outcome, KHÔNG encode policy ---
        reward = self._compute_reward(latency, quality, cost)

        # Query tiếp theo đến
        self._generate_query()
        next_state = self._get_state()
        done = self.step_count >= self.episode_length

        info = {
            "latency": latency,
            "quality": quality,
            "cost": cost,
            "is_complex": self.is_complex,
            "edge_cpu": self.edge_cpu,
            "cloud_cpu": self.cloud_cpu,
        }

        return next_state, reward, done, info

    @staticmethod
    def _compute_reward(latency, quality, cost):
        """
        Reward thuần outcome. KHÔNG có decision_bonus.
        Agent phải TỰ HỌC rằng complex → Cloud cho quality cao hơn.
        """
        quality_reward = (quality / 10.0) * 5.0
        # Trễ chuẩn được nới lỏng lên 25.0s do Edge chạy cực yếu qua VPN
        latency_penalty = (latency / 25.0) ** 1.5
        # Phạt chi phí vừa đủ để Agent cân nhắc
        cost_penalty = cost * 2.0

        return quality_reward - latency_penalty - cost_penalty

    @staticmethod
    def compute_reward_from_log(latency, quality, cost):
        """Public API cho train_worker dùng."""
        return NetworkEnvironment._compute_reward(latency, quality, cost)
