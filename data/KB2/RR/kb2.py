import subprocess
import time
import requests
import json
import os
import argparse
import numpy as np
import sys

# CẤU HÌNH
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_FILE = "locustfile.py"  # Thay bằng tên file locust chứa bộ 60 câu hỏi của bạn
RESULTS_FILE = "kb2_resource_metrics.json"

def get_prom_metric(query, start_t, end_t, step="5s"):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        'query': query,
        'start': start_t,
        'end': end_t,
        'step': step
    }
    try:
        response = requests.get(url, params=params)
        # Nếu không phải 200 (OK), in ra nội dung lỗi để kiểm tra
        if response.status_code != 200:
            print(f"⚠️ Prometheus trả về lỗi HTTP {response.status_code}: {response.text}")
            return 0.0
            
        data = response.json()
        if data['status'] == 'success' and data['data']['result']:
            values = [float(v[1]) for v in data['data']['result'][0]['values']]
            return np.mean(values)
        else:
            print(f"⚠️ Query thành công nhưng không có dữ liệu cho khoảng thời gian này.")
    except Exception as e:
        print(f"⚠️ Lỗi xử lý dữ liệu: {e}")
    return 0.0

def run_benchmark(strategy_name, users=20, rate=5, duration="2m"):
    print(f"🚀 Bắt đầu Benchmark cho chiến lược: [{strategy_name.upper()}]")
    print(f"🔥 Đang bắn Locust ngầm: {users} users, {rate} req/s, thời gian: {duration}...")
    
    start_time = int(time.time())
    
    # Chạy Locust ở chế độ headless (không cần giao diện web)
    cmd = [
        sys.executable, "-m", "locust", "-f", LOCUST_FILE, 
        "--headless", 
        "-u", "50",     # Tăng lên 50 user ập vào cùng lúc
        "-r", "10",     # 10 request / giây
        "--run-time", "1m", # Chạy 1 phút cho nhanh
        "--host", "http://localhost:8000"
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end_time = int(time.time())
    
    print("✅ Đã bắn tải xong! Đang trích xuất dữ liệu từ Prometheus...")
    
    # Nghỉ 5 giây để Prometheus kịp scrape data cuối cùng
    time.sleep(5) 
    
    # Truy vấn CPU (Tùy thuộc bạn dùng thư viện nào, đây là PromQL chuẩn của python client)
    # Nếu bạn dùng node_exporter, sửa thành: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)
    cpu_query = '(1 - avg(rate(node_cpu_seconds_total{mode="idle", instance="100.73.54.43:9100"}[1m]))) * 100'
    ram_query = 'process_resident_memory_bytes / 1024 / 1024' # Đổi ra MB
    
    avg_cpu = get_prom_metric(cpu_query, start_time, end_time)
    avg_ram = get_prom_metric(ram_query, start_time, end_time)
    
    print(f"📊 KẾT QUẢ [{strategy_name}]: CPU Trung bình = {avg_cpu:.1f}% | RAM Trung bình = {avg_ram:.1f} MB")
    
    # Lưu kết quả
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
            
    results[strategy_name] = {
        "cpu_percent": avg_cpu,
        "ram_mb": avg_ram
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Đã lưu vào {RESULTS_FILE}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, help="Tên chiến lược đang test (vd: RoundRobin, DQN)")
    args = parser.parse_args()
    
    run_benchmark(args.strategy)
