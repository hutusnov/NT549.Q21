import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="Medical AI Gateway", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 Hệ thống trợ lý Y tế (DQN Edge-Cloud)")

# --- CẤU HÌNH IP LINH HOẠT TỪ BÊN TRONG GIAO DIỆN ---
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    # Thay bằng IP của Gateway để hội đồng test qua điện thoại/LAN
    GATEWAY_URL = st.text_input("Gateway API Endpoint", value="http://127.0.0.1:8000/ask")
    st.markdown("---")
    st.info("Trạng thái: Sẵn sàng định tuyến 🚀")

# ==========================================

# ==========================================
col1, col2 = st.columns([3, 2]) 

# ==========================================
# CỘT TRÁI (RỘNG HƠN): KHU VỰC TƯƠNG TÁC
# ==========================================
with col1:
    st.subheader("Hỏi AI")
    query = st.text_area("**Bạn đang gặp vấn đề gì về sức khỏe?**", height=120, placeholder="Ví dụ: Tôi bị đau tức ngực trái và vã mồ hôi...")
    
    if st.button("Gửi yêu cầu", type="primary"):
        if query:
            # Tạo ô trống để chữ chạy ra từ từ (Streaming)
            response_box = st.empty()
            full_text = ""
            
            try:
                start_time = time.time()
                # Bật chế độ stream=True
                with requests.post(GATEWAY_URL, json={"query": query}, stream=True, timeout=125.0) as r:
                    if r.status_code != 200:
                        st.error(f"Lỗi hệ thống: HTTP {r.status_code}")
                    else:
                        for line in r.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                
                                # Nhận Meta Data định tuyến
                                if data["type"] == "meta":
                                    st.session_state.meta = data
                                
                                # Nhận từng chữ và gõ ra màn hình
                                elif data["type"] == "token":
                                    full_text += data["content"]
                                    response_box.info("👨‍⚕️ **Phản hồi:**\n\n" + full_text + " ▌") # Con trỏ nhấp nháy
                                
                                elif data["type"] == "error":
                                    st.error(data["content"])
                                    
                # Bỏ con trỏ nhấp nháy khi viết xong, làm nổi bật ô kết quả
                response_box.info("👨‍⚕️ **Phản hồi:**\n\n" + full_text)
                
                # Lưu total time để hiển thị bên cột 2
                if "meta" in st.session_state:
                    st.session_state.meta["latency_sec"] = round(time.time() - start_time, 2)
                    st.session_state.meta["response"] = full_text

            except Exception as e:
                st.error(f"Lỗi kết nối Gateway: {e}")
        else:
            st.warning("Vui lòng nhập câu hỏi!")

# ==========================================
# CỘT PHẢI (HẸP HƠN): KHU VỰC PHÂN TÍCH
# ==========================================
with col2:
    st.subheader("Phân tích từ hệ thống")
    
    if 'meta' in st.session_state:
        res = st.session_state.meta
        
        # Các chỉ số vẫn dàn đều ngang nhau trong cột nhỏ này
        m_col1, m_col2, m_col3 = st.columns(3)
        
        # UI chỉ báo định tuyến
        label = "CLOUD" if res['action_taken'] == "cloud" else "EDGE"
        m_col1.metric("Xử lý tại", label)
        m_col2.metric("Thời gian", f"{res.get('latency_sec', '...')}s")
        m_col3.metric("Rủi ro", "Cao" if res['is_complex'] else "Thấp")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # =======================================================
        # LOGIC HIỂN THỊ ĐÚNG GUARDRAIL VÀ ADAPTIVE ROUTING
        # =======================================================
        if res['action_taken'] == "cloud":
            if res['is_complex']:
                st.error("🚨 **Guardrail Y tế kích hoạt:** Nhận diện triệu chứng phức tạp, hệ thống ưu tiên đẩy lên Cloud (7B) để đảm bảo độ chính xác y khoa tuyệt đối.")
            else:
                st.warning("⚖️ **Adaptive Routing (Tự thích nghi):** Câu hỏi tư vấn cơ bản nhưng Edge đang quá tải. Hệ thống tự động san tải lên Cloud để không làm gián đoạn trải nghiệm.")
        else:
            if res['is_complex']:
                st.warning("🛡️ **Adaptive Routing (Chống đứt gãy):** Câu hỏi phức tạp nhưng kết nối lên Cloud đang nghẽn nặng. Hệ thống ép xử lý khẩn cấp tại Edge để duy trì dịch vụ.")
            else:
                st.success("⚡ **Tối ưu hiệu năng:** Câu hỏi mang tính chất tư vấn cơ bản. Hệ thống xử lý tại Edge (1.5B) để phản hồi tức thì và tiết kiệm chi phí.")
        
    else:
        st.info("Chưa có dữ liệu. Vui lòng nhập câu hỏi để trải nghiệm khả năng phân luồng của Dueling DQN.")
