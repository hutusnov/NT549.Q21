"""
benchmark_runner.py — Benchmark qua NetworkEnvironment

SỬA LỖI: Dùng environment cho simulation thay vì heuristic latency.
So sánh tất cả baselines + DQN Agent trên cùng environment.

Cách dùng:
    python benchmark_runner.py --model dqn_model.pth --episodes 100
"""
import json
import argparse
import csv
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

from baselines import get_all_baselines
from rl_agent import DQNAgent
from environment import NetworkEnvironment

# SLA thresholds
EDGE_SLA_SECONDS = 15.0
CLOUD_SLA_SECONDS = 5.0


@dataclass
class StrategyResult:
    name: str
    latencies: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    sla_violations: List[bool] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    routing_log: List[Dict] = field(default_factory=list)

    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    mean_latency: float = 0.0
    total_cost: float = 0.0
    sla_violation_rate: float = 0.0
    mean_quality: float = 0.0
    mean_reward: float = 0.0
    edge_ratio: float = 0.0
    cloud_ratio: float = 0.0

    def compute_metrics(self):
        if not self.latencies:
            return
        arr = np.array(self.latencies)
        self.p50_latency = float(np.percentile(arr, 50))
        self.p95_latency = float(np.percentile(arr, 95))
        self.p99_latency = float(np.percentile(arr, 99))
        self.mean_latency = float(np.mean(arr))
        self.total_cost = float(np.sum(self.costs))
        self.sla_violation_rate = float(np.mean(self.sla_violations)) * 100
        self.mean_quality = float(np.mean(self.quality_scores))
        self.mean_reward = float(np.mean(self.rewards))

        actions = [r["action"] for r in self.routing_log]
        self.edge_ratio = actions.count(0) / len(actions) * 100
        self.cloud_ratio = actions.count(1) / len(actions) * 100


def check_sla_violation(action: int, latency: float) -> bool:
    sla = EDGE_SLA_SECONDS if action == 0 else CLOUD_SLA_SECONDS
    return latency > sla


def run_benchmark(model_path: str, num_episodes: int = 100,
                  episode_length: int = 50) -> List[StrategyResult]:
    strategies = get_all_baselines()

    # Thêm DQN Agent
    if model_path and os.path.exists(model_path):
        dqn_agent = DQNAgent(state_dim=8, action_dim=2)
        dqn_agent.load_model(model_path)

        class DQNWrapper:
            def __init__(self, agent):
                self.agent = agent
                self.name = "DQN_Agent"

            def get_action(self, state_vector, context=None):
                return self.agent.get_action(np.array(state_vector), explore=False)

        strategies.append(DQNWrapper(dqn_agent))
    else:
        print("⚠️  Không tìm thấy model DQN, chỉ chạy baselines.")

    results = {s.name: StrategyResult(name=s.name) for s in strategies}

    # Chạy từng strategy qua CÙNG MỘT environment (fair comparison)
    for ep in range(num_episodes):
        # Seed giống nhau cho mọi strategy trong cùng episode
        seed = ep * 1000

        for strategy in strategies:
            np.random.seed(seed)
            import random
            random.seed(seed)

            env = NetworkEnvironment(episode_length=episode_length)
            state = env.reset()

            for step in range(episode_length):
                action = strategy.get_action(state)
                next_state, reward, done, info = env.step(action)

                latency = info["latency"]
                quality = info["quality"]
                cost = info["cost"]
                violated = check_sla_violation(action, latency)

                r = results[strategy.name]
                r.latencies.append(latency)
                r.costs.append(cost)
                r.sla_violations.append(violated)
                r.quality_scores.append(quality)
                r.rewards.append(reward)
                r.routing_log.append({
                    "action": action,
                    "latency": latency,
                    "is_complex": info.get("is_complex", False),
                    "violated": violated,
                    "state_vector": state.tolist(),
                    "timestamp": ep * episode_length + step,
                    "routed_to": "cloud" if action == 1 else "edge",
                    "edge_cpu": info["edge_cpu"],
                })

                state = next_state
                if done:
                    break

    for r in results.values():
        r.compute_metrics()

    return list(results.values())


def print_table(results: List[StrategyResult]):
    COL = 14
    headers = ["Strategy", "p50(s)", "p95(s)", "p99(s)", "Mean(s)",
               "Cost", "SLA_Viol%", "Quality", "Reward", "Edge%"]
    sep = "-" * (COL * len(headers))
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (via NetworkEnvironment)")
    print("=" * 70)
    print("".join(h.ljust(COL) for h in headers))
    print(sep)

    for r in sorted(results, key=lambda x: -x.mean_reward):
        row = [
            r.name,
            f"{r.p50_latency:.2f}",
            f"{r.p95_latency:.2f}",
            f"{r.p99_latency:.2f}",
            f"{r.mean_latency:.2f}",
            f"{r.total_cost:.0f}",
            f"{r.sla_violation_rate:.1f}%",
            f"{r.mean_quality:.2f}",
            f"{r.mean_reward:.2f}",
            f"{r.edge_ratio:.1f}%",
        ]
        print("".join(v.ljust(COL) for v in row))
    print(sep + "\n")


def save_csv(results: List[StrategyResult], out_path: str = "benchmark_results.csv"):
    fieldnames = ["strategy", "p50_latency", "p95_latency", "p99_latency",
                  "mean_latency", "total_cost", "sla_violation_rate",
                  "mean_quality", "mean_reward", "edge_ratio", "cloud_ratio"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "strategy": r.name,
                "p50_latency": round(r.p50_latency, 4),
                "p95_latency": round(r.p95_latency, 4),
                "p99_latency": round(r.p99_latency, 4),
                "mean_latency": round(r.mean_latency, 4),
                "total_cost": round(r.total_cost, 2),
                "sla_violation_rate": round(r.sla_violation_rate, 2),
                "mean_quality": round(r.mean_quality, 3),
                "mean_reward": round(r.mean_reward, 3),
                "edge_ratio": round(r.edge_ratio, 2),
                "cloud_ratio": round(r.cloud_ratio, 2),
            })
    print(f"💾 Đã lưu kết quả vào {out_path}")


def save_routing_detail(results: List[StrategyResult], out_dir: str = "benchmark_detail"):
    """Lưu routing log chi tiết của từng strategy để vẽ biểu đồ sau."""
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        path = os.path.join(out_dir, f"{r.name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for entry in r.routing_log:
                f.write(json.dumps(entry) + "\n")
    print(f"📁 Chi tiết routing đã lưu vào thư mục '{out_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark routing strategies")
    parser.add_argument("--model", default="dqn_model.pth",
                        help="File model DQN (.pth)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Số episodes để benchmark")
    parser.add_argument("--length", type=int, default=50,
                        help="Số steps mỗi episode")
    parser.add_argument("--out", default="benchmark_results.csv",
                        help="File CSV đầu ra")
    args = parser.parse_args()

    results = run_benchmark(args.model, args.episodes, args.length)
    print_table(results)
    save_csv(results, args.out)
    save_routing_detail(results)
