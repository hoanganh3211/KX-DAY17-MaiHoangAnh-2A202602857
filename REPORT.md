# Báo cáo Phân tích Kết quả & Đánh giá Trade-off Hệ thống Memory (Day 17)

## 1. Kết quả Benchmark Thực nghiệm

Dưới đây là bảng số liệu thu được khi chạy thử nghiệm trên cùng điều kiện giữa **Baseline Agent** (thuần short-term memory) và **Advanced Agent** (3 tầng memory: short-term + `User.md` persistent + compact memory):

### 1.1. Standard Benchmark (`data/conversations.json` - 10 hội thoại thường)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 1,963 | 15,119 | 0.0% | 40.0% | 0 | 0 |
| **Advanced Agent** | 2,302 | 24,064 | **100.0%** | **100.0%** | 290 | 0 |

### 1.2. Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 336 | 22,262 | 0.0% | 40.0% | 0 | 0 |
| **Advanced Agent** | 747 | **10,701** (giảm > 51.9%) | **100.0%** | **100.0%** | 228 | **7** |

---

## 2. Phân tích Chuyên sâu & Đánh giá Trade-off

### 2.1. Vì sao Advanced Agent có Recall vượt trội so với Baseline Agent?
- **Baseline Agent**: Chỉ lưu trữ tin nhắn tạm thời trong cùng một `thread_id`. Khi sang thread mới (mô phỏng phiên làm việc mới của người dùng), toàn bộ ngữ cảnh cũ bị xóa bỏ hoàn toàn. Do đó, khả năng trả lời các câu hỏi kiểm tra chéo phiên (`cross-session recall`) rơi về `0.0%`.
- **Advanced Agent**: Sử dụng tầng **Persistent Memory** với file cấu trúc `state/profiles/<user>/User.md`. Bất cứ khi nào người dùng cung cấp fact (tên, nơi ở, nghề nghiệp, đồ uống, style trả lời), hệ thống trích xuất và cập nhật trực tiếp vào hồ sơ cá nhân. Khi sang thread mới, agent đọc `User.md` để tiêm vào prompt hệ thống, giúp recall đạt mức tuyệt đối `100.0%`.

### 2.2. Vì sao Advanced Agent tốn token hơn ở hội thoại ngắn?
- Trong các hội thoại ngắn (Standard Benchmark), `Prompt tokens processed` của Advanced cao hơn Baseline (24,064 vs 15,119).
- **Lý do**: Với mỗi lượt chat, Advanced Agent luôn phải nạp thêm ngữ cảnh từ `User.md` (hồ sơ người dùng) vào prompt để giữ được tính cá nhân hóa và khả năng recall. Trong các phiên hội thoại ngắn dưới ngưỡng compact, chi phí mang vác `User.md` này là một overhead bổ sung.
- **Trade-off**: Chấp nhận tốn thêm một lượng nhỏ token prompt ở mỗi lượt để đổi lấy khả năng ghi nhớ dài hạn và tính nhất quán xuyên suốt phiên.

### 2.3. Vì sao Compact Memory giúp Advanced Agent chiếm ưu thế tuyệt đối ở hội thoại dài?
- Trong bài kiểm tra stress test 16 lượt dài:
  - **Baseline Agent** giữ nguyên toàn bộ lịch sử hội thoại không nén. Lượng token tích lũy qua mỗi lượt tăng theo hàm bậc hai ($O(N^2)$), dẫn đến `Prompt tokens processed` bùng nổ lên **22,262 tokens**.
  - **Advanced Agent** kích hoạt **Compact Memory** tổng cộng **7 lần** ngay khi dung lượng hội thoại vượt ngưỡng `compact_threshold_tokens`. Toàn bộ các tin nhắn cũ được nén thành đoạn tóm tắt súc tích, chỉ giữ lại các tin nhắn gần nhất.
  - **Kết quả**: Lượng prompt context của Advanced Agent giảm từ 22,262 xuống chỉ còn **10,701 tokens** (tiết kiệm hơn **51.9%** chi phí xử lý ngữ cảnh), đồng thời vẫn bảo toàn 100% facts quan trọng nhờ `User.md`.

### 2.4. Phân tích Tốc độ Tăng trưởng File Memory & Các Rủi ro Đi Kèm
- **Tốc độ tăng trưởng**: Trong benchmark, file `User.md` tăng từ 0 lên 290 bytes (Standard) và 228 bytes (Stress). Nhờ cấu trúc fact dạng key-value chuẩn hóa, kích thước file tăng tuyến tính rất chậm ($O(K)$ với $K$ là số lượng thực thể thuộc tính).
- **Các rủi ro tiềm ẩn trong môi trường thực tế**:
  1. **Lưu sai fact do nhiễu hoặc câu nói đùa**: Người dùng nói đùa ("hay là đổi nghề sang product manager") nếu không có bộ lọc có thể ghi đè nghề nghiệp thật.
  2. **Xung đột fact khi người dùng đính chính**: Người dùng chuyển nơi ở từ Đà Nẵng sang Huế; nếu hệ thống chỉ append mà không update/replace thì profile sẽ chứa 2 thông tin trái ngược nhau.
  3. **File phình to không giới hạn (Unbounded Growth)**: Nếu lưu cả thông tin vụn vặt, prompt context sẽ ngày càng nặng và làm loãng khả năng chú ý (attention dilution) của LLM.

---

## 3. Các Tính năng Mở rộng (Bonus) Đã Triển khai

Để giải quyết các rủi ro trên, hệ thống đã được bổ sung 4 cơ chế nâng cao:

### 3.1. Phân loại Khai báo vs Đặt câu hỏi (Question vs Declaration Discrimination)
- **Vấn đề giải quyết**: Tránh lưu nhầm khi người dùng đặt câu hỏi tra cứu (ví dụ: *"Bạn có biết DũngCT thích uống gì không?"*) thành một fact mới.
- **Cách triển khai**: Nhận diện các mẫu câu hỏi nghi vấn (`?`, `ở đâu`, `gì`, `nhắc lại`, `phải không`). Nếu tin nhắn không đi kèm động từ xác nhận (`mình là`, `tên là`, `đang ở`, `đính chính`), hệ thống gán độ tin cậy `0.0` và tuyệt đối không ghi vào `User.md`.

### 3.2. Ngưỡng Độ tin cậy (Confidence Thresholding)
- **Vấn đề giải quyết**: Lọc bỏ các thông tin thoáng qua, đùa cợt hoặc không chắc chắn.
- **Cách triển khai**:
  - Fact trực tiếp (`"mình tên là DũngCT"`, `"đang ở Huế"`): confidence = `0.95 - 0.98`.
  - Thông tin đùa cợt (`"câu đùa chuyển sang product manager"`): confidence = `0.10` $\rightarrow$ Bị loại bỏ.
  - Thông tin công tác tạm thời (`"Hà Nội chỉ là nơi vừa bay ra họp 2 ngày"`): confidence = `0.15` $\rightarrow$ Bị loại bỏ.
  - Ngưỡng ghi `min_confidence = 0.70`.

### 3.3. Xử lý Xung đột & Cập nhật Thông tin (Conflict Handling & Fact Upserting)
- **Vấn đề giải quyết**: Khi người dùng thay đổi thông tin (đổi nơi ở, đổi công việc), fact cũ phải được thay thế thay vì giữ song song cả hai.
- **Cách triển khai**: Hàm `upsert_facts()` phân tích khóa fact (`location`, `profession`) và ghi đè giá trị mới, đồng thời hỗ trợ gộp các sở thích kỹ thuật (`interests`: Python + AI).

### 3.4. Cơ chế Giảm trừ Ký ức (Memory Decay)
- **Vấn đề giải quyết**: Dọn dẹp các thông tin tạm thời không còn được nhắc lại trong thời gian dài để tránh phình file memory.
- **Đánh giá rủi ro của Bonus**:
  - Tăng độ phức tạp tính toán khi trích xuất thực thể.
  - Nếu đặt ngưỡng confidence quá cao có thể bỏ sót sở thích ngầm của người dùng (False Negatives).
