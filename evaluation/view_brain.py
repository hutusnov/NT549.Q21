import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# 1. Đọc file .pth vào bộ nhớ
print("Đang mở hộp sọ AI...")
try:
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "..", "models", "dqn_model.pth"), map_location=torch.device('cpu'), weights_only=True)

    # Hỗ trợ cả format mới (checkpoint dict) và cũ (raw state_dict)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        brain_weights = checkpoint['model']
        print(f"   Epsilon: {checkpoint.get('epsilon', 'N/A')}")
        print(f"   Replay count: {checkpoint.get('replay_count', 'N/A')}")
    else:
        brain_weights = checkpoint

    print("\n✅ THÀNH CÔNG! Dưới đây là cấu trúc các nơ-ron bên trong:\n")
    print("-" * 50)
    print(f"{'TÊN LỚP NƠ-RON (LAYER)':<25} | {'KÍCH THƯỚC MA TRẬN (SHAPE)'}")
    print("-" * 50)

    # 2. Duyệt qua từng lớp nơ-ron và in ra kích thước của nó
    for layer_name, weight_matrix in brain_weights.items():
        print(f"{layer_name:<25} | {list(weight_matrix.shape)}")

    print("-" * 50)

    # 3. (Tùy chọn) In thử nội dung của lớp đầu tiên để xem các con số
    first_layer = list(brain_weights.keys())[0]
    print(f"\n👉 Trích xuất thử một góc nhỏ của '{first_layer}':")
    print(brain_weights[first_layer][0][:5]) # In 5 con số đầu tiên

except Exception as e:
    print(f"❌ Lỗi không mở được file: {e}")
