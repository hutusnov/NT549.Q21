# NT549.Q21
Đồ án môn Học máy tăng cường cho các hệ thống mạng

## 1. Chủ đề
Tối ưu hóa giảm tải suy luận (Inference Offloading) cho dịch vụ LLM trong lĩnh vực y tế trên kiến trúc Edge–Cloud bằng học tăng cường.

## 2. Nội dung dự kiến
Xây dựng kiến trúc Edge–Cloud thực nghiệm: Triển khai một hệ thống gồm ba node, bao gồm node Edge với tài nguyên hạn chế để xử lý suy luận bằng mô hình ngôn ngữ nhỏ/nén, node Cloud có cấu hình mạnh để chạy mô hình đầy đủ, và một node Gateway chịu trách nhiệm tiếp nhận và điều phối các yêu cầu.
Triển khai và phục vụ mô hình ngôn ngữ: Các mô hình ngôn ngữ phục vụ truy vấn y tế được container hóa bằng Docker và triển khai thông qua Ollama, đồng thời xây dựng các API để các node có thể giao tiếp và xử lý yêu cầu một cách thống nhất.
Thiết kế môi trường và thuật toán học tăng cường: Mô hình hóa bài toán offloading dưới dạng MDP, trong đó trạng thái bao gồm độ trễ mạng, mức sử dụng GPU tại Edge và đặc trưng độ phức tạp của câu hỏi. Tác nhân học tăng cường (Dueling DQN) được huấn luyện để đưa ra quyết định xử lý yêu cầu tại Edge hay chuyển lên Cloud sao cho tối ưu hiệu năng tổng thể.
Giả lập điều kiện mạng động: Sử dụng các công cụ điều khiển mạng trên Linux như tc và netem để tạo ra các kịch bản mạng khác nhau (độ trễ cao, nghẽn mạng, mất gói), qua đó đánh giá và huấn luyện khả năng thích ứng của tác nhân RL trong môi trường gần với thực tế.

## 3. Kết quả dự kiến
Demo hoạt động thực tế của hệ thống: Hệ thống có khả năng tự động phân luồng truy vấn y tế, trong đó các câu hỏi đơn giản được xử lý nhanh tại Edge nhằm giảm độ trễ, trong khi các truy vấn phức tạp hơn được chuyển lên Cloud để đảm bảo chất lượng trả lời.
Khả năng thích ứng theo thời gian thực: Tác nhân RL học được cách điều chỉnh chiến lược offloading khi điều kiện mạng hoặc tải hệ thống thay đổi, chẳng hạn ưu tiên xử lý tại Edge khi kết nối Cloud gặp độ trễ hoặc không ổn định.
Cải thiện hiệu năng so với các phương pháp truyền thống: Phương pháp đề xuất đạt được sự cân bằng tốt hơn giữa độ trễ và chất lượng suy luận so với các chiến lược định tuyến cố định hoặc ngẫu nhiên.
Phân tích định lượng rõ ràng: Kết quả được trình bày thông qua các biểu đồ và bảng so sánh về thời gian phản hồi, chi phí tài nguyên và mức độ đáp ứng SLA trong nhiều kịch bản thử nghiệm khác nhau."
