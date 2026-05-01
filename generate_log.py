import json
import random
import time

LOG_FILE = "demo_experience.jsonl"
NUM_RECORDS = 2000

print(f"Đang sinh {NUM_RECORDS} dòng log giả lập dữ liệu mạng...")

records = []
for i in range(NUM_RECORDS):
    is_complex = random.random() < 0.3
    critical_count = random.randint(1, 3) if is_complex else 0

    edge_cpu = random.uniform(0.1, 0.95)
    # FIX: Cloud cũng có thể bị tải cao, không bias thấp hơn Edge
    cloud_cpu = random.uniform(0.1, 0.90)

    # FIX: random.uniform(1.0, 25.0) thay vì (5.0, 25.0) để safe_lat có thể < 5.0
    # min(..., 5.0) giữ đúng như app3 nhưng bây giờ có variance thực sự
    safe_edge_lat = min(random.uniform(1.0, 25.0), 5.0)
    safe_cloud_lat = min(random.uniform(1.0, 15.0), 5.0)

    state_vector = [
        critical_count,
        1.0 if is_complex else 0.0,
        safe_edge_lat,
        safe_cloud_lat,
        edge_cpu,
        cloud_cpu
    ]

    # 80% expert, 20% ngẫu nhiên
    if random.random() < 0.8:
        if is_complex:
            action = 1
        elif edge_cpu > 0.85:
            action = 1
        else:
            action = 0
    else:
        action = random.choice([0, 1])

    routed_to = "cloud" if action == 1 else "edge"

    if action == 0:
        if is_complex:
            latency = random.uniform(30.0, 60.0)
            mock_score = random.uniform(3.0, 5.0)
        else:
            latency = random.uniform(6.0, 14.0) * (1.0 + edge_cpu)
            mock_score = random.uniform(7.5, 9.0)
    else:
        latency = random.uniform(4.0, 8.0) * (1.0 + cloud_cpu)
        mock_score = random.uniform(8.5, 10.0)

    record = {
        "timestamp": time.time() - random.randint(0, 86400),
        "query": f"Fake {'Complex' if is_complex else 'Simple'} Query #{i}",
        "response": "Nội dung giả lập...",
        "state_vector": state_vector,
        "action": action,
        "routed_to": routed_to,
        "latency": round(latency, 3),
        "is_complex": is_complex,
        "mock_score": round(mock_score, 1)
    }
    records.append(record)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Hoàn tất! Đã lưu vào {LOG_FILE}.")
