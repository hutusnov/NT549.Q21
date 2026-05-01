"""
benchmark_runner.py
Chạy tất cả baseline + DQN Agent trên cùng một tập dữ liệu log,
tính toán các metrics chuẩn và xuất kết quả ra CSV + in bảng so sánh.

Cách dùng:
    python benchmark_runner.py --log demo_experience.jsonl --model dqn_model_demo.pth
"""
import json
import argparse
import csv
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from baselines import get_all_baselines
from rl_agent import DQNAgent

# ─── Cấu hình SLA ─────────────────────────────────────────────────────────────
EDGE_SLA_SECONDS  = 15.0   # SLA cho Edge node
CLOUD_SLA_SECONDS =  5.0   # SLA cho Cloud node

# Chi phí giả định (đơn vị tùy ý, dùng để tính Cost ratio)
CLOUD_COST_PER_REQUEST = 1.0
EDGE_COST_PER_REQUEST  = 0.1

# ─── Dataclass chứa kết quả của một chiến lược ────────────────────────────────
@dataclass
class StrategyResult:
    name: str
    latencies: List[float] = field(default_factory=list)
    costs: List[float]     = field(default_factory=list)
    sla_violations: List[bool] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    routing_log: List[Dict]    = field(default_factory=list)  # chi tiết từng request

    # Metrics tổng hợp (tính sau)
    p50_latency:   float = 0.0
    p95_latency:   float = 0.0
    p99_latency:   float = 0.0
    mean_latency:  float = 0.0
    total_cost:    float = 0.0
    sla_violation_rate: float = 0.0
    mean_quality:  float = 0.0
    edge_ratio:    float = 0.0   # % request về Edge
    cloud_ratio:   float = 0.0

    def compute_metrics(self):
        if not self.latencies:
            return
        arr = np.array(self.latencies)
        self.p50_latency  = float(np.percentile(arr, 50))
        self.p95_latency  = float(np.percentile(arr, 95))
        self.p99_latency  = float(np.percentile(arr, 99))
        self.mean_latency = float(np.mean(arr))
        self.total_cost   = float(np.sum(self.costs))
        self.sla_violation_rate = float(np.mean(self.sla_violations)) * 100  # %
        self.mean_quality = float(np.mean(self.quality_scores)) if self.quality_scores else 0.0

        actions = [r["action"] for r in self.routing_log]
        self.edge_ratio  = actions.count(0) / len(actions) * 100
        self.cloud_ratio = actions.count(1) / len(actions) * 100


def simulate_latency(action: int, base_latency: float,
                     edge_cpu: float, cloud_cpu: float) -> float:
    """
    Giả lập latency thực tế dựa trên action và CPU load.
    Dùng khi log không có latency thực (ví dụ fake logs từ generate_fake_logs).
    Nếu log đã có 'latency' thì dùng thẳng.
    """
    if action == 0:  # Edge
        return base_latency * (1.0 + edge_cpu * 0.5)
    else:            # Cloud
        return base_latency * (1.0 + cloud_cpu * 0.3)


def check_sla_violation(action: int, latency: float) -> bool:
    sla = EDGE_SLA_SECONDS if action == 0 else CLOUD_SLA_SECONDS
    return latency > sla


def compute_cost(action: int) -> float:
    return CLOUD_COST_PER_REQUEST if action == 1 else EDGE_COST_PER_REQUEST


def run_benchmark(log_path: str, model_path: str | None = None) -> List[StrategyResult]:
    # Load data
    with open(log_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"📂 Loaded {len(records)} records từ {log_path}")

    # Khởi tạo tất cả strategy
    strategies = get_all_baselines()

    # Thêm DQN Agent nếu có model
    dqn_agent = None
    if model_path and os.path.exists(model_path):
        dqn_agent = DQNAgent(state_dim=6, action_dim=2)
        dqn_agent.load_model(model_path)
        # Wrap agent để cùng interface với baseline
        class DQNWrapper:
            def __init__(self, agent):
                self.agent = agent
                self.name = "DQN_Agent"
            def get_action(self, state_vector, context=None):
                return self.agent.get_action(np.array(state_vector))
        strategies.append(DQNWrapper(dqn_agent))
    else:
        print("⚠️  Không tìm thấy model DQN, chỉ chạy baselines.")

    results = {s.name: StrategyResult(name=s.name) for s in strategies}

    # ─── Chạy từng record qua từng strategy ────────────────────────────────────
    for rec in records:
        state_vector = rec["state_vector"]
        edge_cpu  = state_vector[4]
        cloud_cpu = state_vector[5]
        is_complex = rec.get("is_complex", False)
        mock_score = rec.get("mock_score", rec.get("latency", 5.0))  # fallback

        # Latency thực tế từ log (nếu có) hoặc tính lại
        actual_latency_logged = rec.get("latency", None)

        for strategy in strategies:
            action = strategy.get_action(state_vector)

            # Latency: nếu đây là log thực tế và action khớp với action gốc thì dùng log
            # Nếu không (strategy chọn khác) thì simulate
            if actual_latency_logged is not None and action == rec.get("action"):
                latency = actual_latency_logged
            else:
                # Simulate dựa trên loại node và CPU
                base = 8.0 if action == 0 else 5.0
                latency = simulate_latency(action, base, edge_cpu, cloud_cpu)
                # Câu phức tạp ở Edge thì chậm hơn nhiều
                if is_complex and action == 0:
                    latency *= 4.0

            cost     = compute_cost(action)
            violated = check_sla_violation(action, latency)

            # Quality: câu phức tạp ở Edge bị giảm chất lượng
            if is_complex and action == 0:
                quality = mock_score * 0.5 if isinstance(mock_score, float) else 4.0
            else:
                quality = mock_score if isinstance(mock_score, float) else 8.0

            r = results[strategy.name]
            r.latencies.append(latency)
            r.costs.append(cost)
            r.sla_violations.append(violated)
            r.quality_scores.append(quality)
            r.routing_log.append({
                "action": action,
                "latency": latency,
                "is_complex": is_complex,
                "violated": violated
            })

    # ─── Tính metrics tổng hợp ─────────────────────────────────────────────────
    for r in results.values():
        r.compute_metrics()

    return list(results.values())


def print_table(results: List[StrategyResult]):
    COL = 14
    headers = ["Strategy", "p50(s)", "p95(s)", "p99(s)", "Mean(s)",
               "Cost", "SLA_Viol%", "Quality", "Edge%", "Cloud%"]
    sep = "-" * (COL * len(headers))
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print("".join(h.ljust(COL) for h in headers))
    print(sep)

    # Sắp xếp theo p95 latency
    for r in sorted(results, key=lambda x: x.p95_latency):
        row = [
            r.name,
            f"{r.p50_latency:.2f}",
            f"{r.p95_latency:.2f}",
            f"{r.p99_latency:.2f}",
            f"{r.mean_latency:.2f}",
            f"{r.total_cost:.0f}",
            f"{r.sla_violation_rate:.1f}%",
            f"{r.mean_quality:.2f}",
            f"{r.edge_ratio:.1f}%",
            f"{r.cloud_ratio:.1f}%",
        ]
        print("".join(v.ljust(COL) for v in row))
    print(sep + "\n")


def save_csv(results: List[StrategyResult], out_path: str = "benchmark_results.csv"):
    fieldnames = ["strategy", "p50_latency", "p95_latency", "p99_latency",
                  "mean_latency", "total_cost", "sla_violation_rate",
                  "mean_quality", "edge_ratio", "cloud_ratio"]
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
    parser.add_argument("--log",   default="demo_experience.jsonl",
                        help="File log JSONL đầu vào")
    parser.add_argument("--model", default="dqn_model_demo.pth",
                        help="File model DQN (.pth)")
    parser.add_argument("--out",   default="benchmark_results.csv",
                        help="File CSV đầu ra")
    args = parser.parse_args()

    results = run_benchmark(args.log, args.model)
    print_table(results)
    save_csv(results, args.out)
    save_routing_detail(results)
