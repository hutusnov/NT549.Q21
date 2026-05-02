import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# analyze_results.py
import argparse
import csv
import json
import os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # Không cần GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS = {
    "RoundRobin":   "#888780",
    "AlwaysCloud":  "#E24B4A",
    "AlwaysEdge":   "#BA7517",
    "LatencyBased": "#378ADD",
    "CpuAware":     "#1D9E75",
    "DQN_Agent":    "#7F77DD",
}
DEFAULT_COLOR = "#555"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_detail(detail_dir: str) -> dict[str, list[dict]]:
    """Trả về {strategy_name: [{"latency": x, "timestamp": y, "routed_to": z, "edge_cpu": c}, ...]}"""
    out = {}
    if not os.path.isdir(detail_dir):
        return out
    for fname in os.listdir(detail_dir):
        if not fname.endswith(".jsonl"):
            continue
        name = fname[:-6]  # bỏ .jsonl
        records = []
        with open(os.path.join(detail_dir, fname), encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                
                # Trích xuất edge_cpu từ state_vector (Vị trí index 4)
                state_vec = entry.get("state_vector", [0, 0, 0, 0, 0, 0])
                edge_cpu = state_vec[4] if len(state_vec) > 4 else 0.0

                records.append({
                    "latency": entry["latency"],
                    "timestamp": entry.get("timestamp", 0),
                    "routed_to": entry.get("routed_to", "edge"),
                    "edge_cpu": edge_cpu
                })
        out[name] = records
    return out


def bar_chart(ax, names: list[str], values: list[float],
              title: str, ylabel: str, fmt: str = "{:.2f}"):
    colors = [COLORS.get(n, DEFAULT_COLOR) for n in names]
    bars = ax.bar(names, values, color=colors, width=0.55, edgecolor="none")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                fmt.format(val),
                ha="center", va="bottom", fontsize=9)


def cdf_chart(ax, detail: dict[str, list[dict]], title: str):
    for name, records in sorted(detail.items()):
        lats = [r["latency"] for r in records] 
        if not lats: continue
        arr = np.sort(lats)
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        color = COLORS.get(name, DEFAULT_COLOR)
        lw = 2.5 if name == "DQN_Agent" else 1.5
        ls = "-" if name == "DQN_Agent" else "--"
        ax.plot(arr, cdf, label=name, color=color, linewidth=lw, linestyle=ls)

    # Đường SLA
    ax.axvline(x=5.0,  color="#E24B4A", linewidth=1, linestyle=":", alpha=0.8)
    ax.axvline(x=15.0, color="#BA7517", linewidth=1, linestyle=":", alpha=0.8)
    ax.text(5.2,  0.05, "Cloud SLA\n5s",  fontsize=8, color="#E24B4A")
    ax.text(15.2, 0.05, "Edge SLA\n15s", fontsize=8, color="#BA7517")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, loc="lower right")


def time_series_chart(detail: dict[str, list[dict]], out_dir: str):
    """Vẽ biểu đồ sự thích ứng của DQN khi có Burst Traffic (Hạng mục P1)"""
    if "DQN_Agent" not in detail:
        return
        
    records = detail["DQN_Agent"]
    if not records:
        return
        
    start_t = min(r["timestamp"] for r in records)
    
    # Gom nhóm dữ liệu theo từng block 30 giây để thấy rõ Sốc tải
    window_size = 30
    bins = defaultdict(lambda: {"cloud": 0, "edge": 0, "cpu_sum": 0.0, "count": 0})
    
    for r in records:
        w_idx = int((r["timestamp"] - start_t) // window_size)
        bins[w_idx]["cloud" if r["routed_to"] == "cloud" else "edge"] += 1
        bins[w_idx]["cpu_sum"] += r["edge_cpu"]
        bins[w_idx]["count"] += 1
        
    sorted_windows = sorted(bins.keys())
    
    time_axis = []
    cloud_ratios = []
    edge_cpus = []
    req_counts = []
    
    for w in sorted_windows:
        b = bins[w]
        total = b["count"]
        time_axis.append(w * window_size) # Chuyển về giây
        cloud_ratios.append((b["cloud"] / total * 100) if total > 0 else 0)
        edge_cpus.append((b["cpu_sum"] / total * 100) if total > 0 else 0)
        req_counts.append(total)
        
    # Bắt đầu vẽ
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # 1. Trục Y Trái: Vẽ CPU Edge và Tỷ lệ Cloud
    ax1.plot(time_axis, edge_cpus, color="#BA7517", linestyle='--', linewidth=2.5, label='Edge CPU Load (%)')
    ax1.plot(time_axis, cloud_ratios, color="#E24B4A", linewidth=3, label='% Định tuyến lên Cloud')
    
    ax1.set_title("Time-Series: Dueling DQN tự động xoay trục định tuyến khi Sốc tải (Burst)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Thời gian mô phỏng (Giây)", fontsize=11)
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Trục Y Phải: Vẽ Lưu lượng Request (Burst)
    ax2 = ax1.twinx()
    ax2.fill_between(time_axis, 0, req_counts, color="#378ADD", alpha=0.15, label='Lưu lượng (Req/30s)')
    ax2.set_ylabel("Số lượng Request", fontsize=11)
    
    max_req = max(req_counts) if req_counts else 10
    ax2.set_ylim(0, max_req * 1.5) # Để khoảng trống phía trên cho đẹp

    # Gộp legend của 2 trục
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "p1_routing_time_series.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"✅ Đã lưu biểu đồ Time-series (P1) tại: {path}")
    plt.close(fig)


def make_report(csv_path: str, detail_dir: str, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_csv(csv_path)
    detail = load_detail(detail_dir)

    names  = [r["strategy"]           for r in rows]
    p95    = [float(r["p95_latency"]) for r in rows]
    cost   = [float(r["total_cost"])  for r in rows]
    sla    = [float(r["sla_violation_rate"]) for r in rows]
    qual   = [float(r["mean_quality"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Benchmark: Routing Strategy Comparison", fontsize=14, fontweight="bold", y=1.01)

    bar_chart(axes[0][0], names, p95,  "p95 Latency (giây)", "Latency (s)")
    bar_chart(axes[0][1], names, cost, "Tổng Chi phí (đơn vị)", "Cost")
    bar_chart(axes[1][0], names, sla,  "SLA Violation Rate", "Violation (%)", fmt="{:.1f}%")
    bar_chart(axes[1][1], names, qual, "Điểm Chất lượng Trung bình", "Quality Score (1-10)")

    plt.tight_layout()
    path1 = os.path.join(out_dir, "benchmark_bars.png")
    fig.savefig(path1, bbox_inches="tight")
    print(f"✅ Đã lưu {path1}")
    plt.close(fig)

    # CDF & Time-series riêng
    if detail:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        cdf_chart(ax2, detail, "CDF Latency — tất cả chiến lược")
        plt.tight_layout()
        path2 = os.path.join(out_dir, "latency_cdf.png")
        fig2.savefig(path2, bbox_inches="tight")
        print(f"✅ Đã lưu {path2}")
        plt.close(fig2)
        
        # Gọi hàm vẽ Time-series (giải quyết P1)
        time_series_chart(detail, out_dir)
    else:
        print("⚠️  Không tìm thấy benchmark_detail/, bỏ qua CDF & Time-series chart.")

    # In bảng tóm tắt
    print("\n📊 Tóm tắt kết quả:")
    print(f"{'Strategy':<16} {'p95(s)':>8} {'Cost':>8} {'SLA%':>8} {'Quality':>9}")
    print("-" * 55)
    for r in sorted(rows, key=lambda x: float(x["p95_latency"])):
        print(f"{r['strategy']:<16} {float(r['p95_latency']):>8.2f} "
              f"{float(r['total_cost']):>8.0f} "
              f"{float(r['sla_violation_rate']):>7.1f}% "
              f"{float(r['mean_quality']):>9.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default=os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_results.csv"))
    parser.add_argument("--detail", default=os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_detail"))
    parser.add_argument("--out",    default="figures")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Không tìm thấy {args.csv}. Chạy benchmark_runner.py trước.")
    else:
        make_report(args.csv, args.detail, args.out)
