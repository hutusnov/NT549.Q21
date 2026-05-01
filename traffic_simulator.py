"""
traffic_simulator.py
Sinh traffic giả lập có:
  - Burst load (đột biến request lúc giờ cao điểm)
  - Time-of-day variation (sáng nhẹ, trưa vừa, chiều đông)
  - Token count thực tế dùng tiktoken
  - Các kịch bản test: normal, burst, network_degraded

Output: JSONL tương thích với benchmark_runner.py và train_worker.py

Cách dùng:
    python traffic_simulator.py --scenario normal   --records 1000
    python traffic_simulator.py --scenario burst    --records 500
    python traffic_simulator.py --scenario degraded --records 500
"""
import json
import random
import time
import argparse
import math

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3  # ước tính thô

# Tập câu hỏi y tế mẫu để tính token count thực
SIMPLE_QUERIES = [
    "Bệnh gút là gì?",
    "Triệu chứng của gút là gì?",
    "Chế độ ăn kiêng cho bệnh nhân gút?",
    "Gút có chữa khỏi hoàn toàn được không?",
    "Bệnh gút có lây không?",
]

COMPLEX_QUERIES = [
    "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận mãn tính giai đoạn 3, đang dùng corticoid thì nên điều trị như thế nào?",
    "Chống chỉ định của allopurinol trong trường hợp bệnh nhân có tiền sử dị ứng và suy thận nặng là gì? Có thể thay thế bằng thuốc nào?",
    "Bệnh nhân gút cấp tính có biến chứng dịch khớp nhiều, bạch cầu cấp tăng cao, cần làm xquang và xét nghiệm gì thêm trước phẫu thuật?",
    "Phác đồ điều trị gút cho bệnh nhân cao tuổi có suy thận, tiểu đường, đang dùng nhiều thuốc huyết áp khác nhau?",
]


def time_of_day_load_factor(hour: float) -> float:
    """Hàm sin giả lập load theo giờ trong ngày (0-23)."""
    # Cao điểm lúc 9h sáng và 15h chiều
    morning = math.exp(-((hour - 9) ** 2) / 8)
    afternoon = math.exp(-((hour - 15) ** 2) / 6)
    return 0.3 + 0.7 * max(morning, afternoon)


def generate_record(idx: int, hour: float, scenario: str,
                    edge_cpu_base: float, cloud_cpu_base: float) -> dict:
    load_factor = time_of_day_load_factor(hour)

    # Tỷ lệ câu khó tăng theo giờ cao điểm
    complex_prob = 0.2 + 0.2 * load_factor
    is_complex = random.random() < complex_prob
    critical_count = random.randint(1, 3) if is_complex else 0

    query = random.choice(COMPLEX_QUERIES if is_complex else SIMPLE_QUERIES)
    token_count = count_tokens(query)

    # CPU bị ảnh hưởng bởi scenario
    if scenario == "burst":
        edge_cpu = min(0.99, edge_cpu_base + load_factor * 0.5 + random.uniform(0, 0.2))
        cloud_cpu = min(0.85, cloud_cpu_base + load_factor * 0.3)
    elif scenario == "degraded":
        # Network degraded: CPU bình thường nhưng latency tăng do mạng
        edge_cpu = edge_cpu_base + random.uniform(-0.1, 0.15)
        cloud_cpu = cloud_cpu_base + random.uniform(-0.1, 0.10)
        edge_cpu = min(0.95, max(0.1, edge_cpu))
        cloud_cpu = min(0.90, max(0.1, cloud_cpu))
    else:
        edge_cpu = min(0.95, max(0.1, edge_cpu_base + random.uniform(-0.1, 0.1)))
        cloud_cpu = min(0.90, max(0.1, cloud_cpu_base + random.uniform(-0.1, 0.1)))

    safe_edge_lat = min(random.uniform(1.0, 20.0), 5.0)
    safe_cloud_lat = min(random.uniform(1.0, 12.0), 5.0)

    # Nếu degraded thì latency lịch sử bị kéo lên
    if scenario == "degraded":
        safe_edge_lat = 5.0  # Edge link bị nghẽn, luôn ở mức tối đa
        safe_cloud_lat = min(random.uniform(2.0, 8.0), 5.0)

    state_vector = [
        critical_count,
        1.0 if is_complex else 0.0,
        safe_edge_lat,
        safe_cloud_lat,
        edge_cpu,
        cloud_cpu
    ]

    # Simulate action từ expert policy (dùng để tạo training data tốt)
    if is_complex or edge_cpu > 0.88:
        action = 1
    else:
        action = 0 if random.random() < 0.85 else 1

    routed_to = "cloud" if action == 1 else "edge"

    # Simulate latency kết quả
    if action == 0:  # Edge
        base_lat = 8.0 + token_count * 0.05
        latency = base_lat * (1.0 + edge_cpu) * (2.5 if is_complex else 1.0)
        if scenario == "degraded":
            latency *= random.uniform(1.5, 3.0)  # Network penalty
        mock_score = random.uniform(3.0, 5.5) if is_complex else random.uniform(7.0, 9.0)
    else:  # Cloud
        base_lat = 4.0 + token_count * 0.02
        latency = base_lat * (1.0 + cloud_cpu * 0.4)
        if scenario == "burst":
            latency *= random.uniform(1.0, 2.5)  # Cloud cũng bị quá tải khi burst
        mock_score = random.uniform(8.5, 10.0)

    return {
        "timestamp": time.time() - random.randint(0, 86400),
        "query": query,
        "response": "Nội dung giả lập...",
        "state_vector": state_vector,
        "action": action,
        "routed_to": routed_to,
        "latency": round(latency, 3),
        "is_complex": is_complex,
        "token_count": token_count,
        "scenario": scenario,
        "hour": round(hour, 1),
        "mock_score": round(mock_score, 1)
    }


def generate_traffic(scenario: str, num_records: int) -> list[dict]:
    records = []

    # Baseline CPU cho từng scenario
    if scenario == "burst":
        edge_cpu_base, cloud_cpu_base = 0.7, 0.5
    elif scenario == "degraded":
        edge_cpu_base, cloud_cpu_base = 0.4, 0.35
    else:
        edge_cpu_base, cloud_cpu_base = 0.4, 0.3

    for i in range(num_records):
        # Giả lập thời gian: phân bố không đều, nhiều request lúc 8-17h
        if random.random() < 0.7:
            hour = random.uniform(8, 18)
        else:
            hour = random.uniform(0, 24)

        # Burst: một số burst ngắn trong ngày
        if scenario == "burst" and random.random() < 0.15:
            for _ in range(random.randint(3, 8)):
                records.append(generate_record(len(records), hour,
                                               scenario, 0.92, 0.75))

        records.append(generate_record(i, hour, scenario,
                                       edge_cpu_base, cloud_cpu_base))

    return records[:num_records]  # đảm bảo đúng số lượng


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["normal", "burst", "degraded"],
                        default="normal")
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_file = args.out or f"traffic_{args.scenario}.jsonl"
    records = generate_traffic(args.scenario, args.records)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    complex_count = sum(1 for r in records if r["is_complex"])
    print(f"✅ Đã sinh {len(records)} records (scenario={args.scenario})")
    print(f"   Câu phức tạp: {complex_count} ({complex_count/len(records)*100:.1f}%)")
    print(f"   Đã lưu vào: {out_file}")
