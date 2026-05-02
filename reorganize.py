import os
import shutil
import re

print("Tổ chức lại thư mục project...")

# 1. Tạo các thư mục mới
dirs = ["gateway", "rl", "training", "evaluation", "tests", "models", "data", "docs"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    if d not in ["models", "data", "docs"]:
        with open(os.path.join(d, "__init__.py"), "w") as f:
            f.write("# Init module\n")

# 2. Phân loại file
file_map = {
    "gateway": ["app.py", "feature_extractor.py", "frontend.py"],
    "rl": ["rl_agent.py", "environment.py", "baselines.py"],
    "training": ["fast_offline_train.py", "train_worker.py", "traffic_simulator.py", "generate_log.py"],
    "evaluation": ["benchmark_runner.py", "analyze_results.py", "view_brain.py", "plot_kb2.py", "plot_kb3.py", "kb2.py"],
    "tests": ["mock_infrastructure.py", "test_request.py", "test_suite.py", "locustfile.py"],
    "models": ["dqn_model.pth", "dqn_model_main.pth", "dqn_model_old_6dim.pth"],
    "data": ["experience_log.jsonl", "demo_experience.jsonl", "benchmark_results.csv"],
    "docs": ["README.md"]
}

# Di chuyển benchmark_detail
if os.path.exists("benchmark_detail"):
    shutil.move("benchmark_detail", os.path.join("data", "benchmark_detail"))

# Di chuyển file
for folder, files in file_map.items():
    for file in files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(folder, file))

if os.path.exists("run_mock_test.ps1"):
    shutil.move("run_mock_test.ps1", os.path.join("tests", "run_mock_test.ps1"))

# 3. Sửa nội dung code (Path + Import)
def process_file(filepath):
    if not filepath.endswith(".py"): return
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Thêm sys.path.append để import chéo hoạt động
    sys_path_inject = """import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
    if "import sys" not in content[:200]:
        content = sys_path_inject + "\n" + content

    # Fix Imports
    content = re.sub(r'from rl_agent import', r'from rl.rl_agent import', content)
    content = re.sub(r'from environment import', r'from rl.environment import', content)
    content = re.sub(r'from baselines import', r'from rl.baselines import', content)
    content = re.sub(r'from feature_extractor import', r'from gateway.feature_extractor import', content)
    
    # Fix Paths (Dùng os.path.join(os.path.dirname(__file__), "..", "thư_mục", "file"))
    content = content.replace('"dqn_model.pth"', 'os.path.join(os.path.dirname(__file__), "..", "models", "dqn_model.pth")')
    content = content.replace('"experience_log.jsonl"', 'os.path.join(os.path.dirname(__file__), "..", "data", "experience_log.jsonl")')
    content = content.replace('"demo_experience.jsonl"', 'os.path.join(os.path.dirname(__file__), "..", "data", "demo_experience.jsonl")')
    content = content.replace('"benchmark_results.csv"', 'os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_results.csv")')
    content = content.replace('"benchmark_detail"', 'os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_detail")')

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# Quét tất cả file py trong các folder để sửa
for folder in ["gateway", "rl", "training", "evaluation", "tests"]:
    for f in os.listdir(folder):
        if f.endswith(".py"):
            process_file(os.path.join(folder, f))

print("✅ Đã tổ chức xong! Các file đã được phân vào thư mục và tự động cập nhật đường dẫn (import & models).")
