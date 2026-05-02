import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
import asyncio
import json
import time

URL = "http://127.0.0.1:8000/ask"

QUERIES = {
    "SIMPLE": [
        "Bệnh gút là gì?",
        "Triệu chứng nhẹ của gút?",
        "Ăn gì khi bị gút?",
        "Gút có lây không?",
        "Làm sao để phòng bệnh gút?"
    ],
    "COMPLEX": [
        "Bệnh nhân gút có hạt tôphi bị loét, kèm suy thận mãn tính giai đoạn 3 thì nên điều trị như thế nào?",
        "Chống chỉ định của allopurinol trong trường hợp suy thận nặng là gì?",
        "Phẫu thuật cắt bỏ hạt tophi có rủi ro gì cho người già bị tiểu đường?",
        "Bệnh nhân có biến chứng dịch khớp nhiều, bạch cầu tăng cao sau khi dùng corticoid lâu ngày?",
        "Phác đồ kết hợp thuốc kháng viêm và giảm acid uric cho người suy gan?"
    ]
}

async def send_request(client, category, query, request_id):
    start_time = time.time()
    try:
        async with client.stream("POST", URL, json={"query": query}, timeout=60.0) as resp:
            if resp.status_code != 200:
                print(f"❌ [REQ {request_id}] Lỗi {resp.status_code}")
                return
            
            # Lấy iterator duy nhất cho stream
            lines = resp.aiter_lines()
            
            # Đọc dòng đầu tiên (metadata)
            try:
                meta_line = await lines.__anext__()
                meta = json.loads(meta_line)
                action = meta.get("action_taken", "unknown")
                is_complex = meta.get("is_complex", False)
            except StopAsyncIteration:
                print(f"❌ [REQ {request_id}] Phản hồi trống")
                return
            
            # Đọc tiếp các dòng còn lại từ CÙNG iterator 'lines'
            full_text = ""
            async for line in lines:
                if line:
                    try:
                        token_data = json.loads(line)
                        full_text += token_data.get("content", "")
                    except: pass
            
            duration = time.time() - start_time
            print(f"📌 [REQ {request_id}] [{category}] -> {action.upper()} (Phức tạp={is_complex}) | Thời gian: {duration:.2f}s")
            # print(f"   Trả lời: {full_text[:50]}...")
            
    except Exception as e:
        print(f"❌ [REQ {request_id}] Lỗi: {e}")

async def run_suite():
    async with httpx.AsyncClient() as client:
        print("\n=== BẮT ĐẦU TEST SUITE ===\n")
        
        print("--- PHASE 1: CÂU HỎI ĐƠN GIẢN (Mong đợi: EDGE) ---")
        for i, q in enumerate(QUERIES["SIMPLE"]):
            await send_request(client, "SIMPLE", q, i+1)
            await asyncio.sleep(0.5)
            
        print("\n--- PHASE 2: CÂU HỎI PHỨC TẠP (Mong đợi: CLOUD) ---")
        for i, q in enumerate(QUERIES["COMPLEX"]):
            await send_request(client, "COMPLEX", q, i+6)
            await asyncio.sleep(0.5)

        print("\n--- PHASE 3: TEST KHẢ NĂNG CHỊU TẢI & BẺ LÁI (Gửi dồn dập) ---")
        tasks = []
        for i in range(10):
            # Gửi ngẫu nhiên các câu hỏi dồn dập
            q = QUERIES["SIMPLE"][i % 5]
            tasks.append(send_request(client, "STRESS", q, i+11))
        
        await asyncio.gather(*tasks)
        
        print("\n=== HOÀN TẤT TEST SUITE ===\n")

if __name__ == "__main__":
    asyncio.run(run_suite())
