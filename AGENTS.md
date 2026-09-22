# AGENTS.md — Quy tắc phát triển Legal AI

## Mục tiêu và phạm vi

- Dùng `kientruc.md` làm kiến trúc mục tiêu, không giả định các module/thư mục trong sơ đồ đã được triển khai.
- Repo hiện còn là prototype gồm các script cấp gốc (`data.py`, `embedding.py`, `retrieval.py`, `chatbot.py`, `chatbot1.py`) và dữ liệu dưới `DATA/`. Khi sửa, bám cấu trúc đang có; chỉ tạo hoặc di chuyển module khi yêu cầu cụ thể cần tới.
- Không rewrite toàn bộ project, không thay framework hay schema diện rộng trong một lần sửa. Ưu tiên thay đổi nhỏ, độc lập, dễ review và có thể kiểm tra.
- Trước khi sửa, đọc file liên quan, xác định luồng gọi và giữ tương thích với dữ liệu hiện có nếu không có yêu cầu đổi schema.

## Kiến trúc và công nghệ đã chọn

- **Vector database:** Qdrant là vector DB chính. Không bổ sung vector store khác nếu chưa có yêu cầu rõ ràng.
- **Embedding:** BGE-M3 (`BAAI/bge-m3`). Giữ metadata pháp lý cùng vector; không tách thứ tự vector khỏi bản ghi nguồn.
- **LLM:** Gemini. API key lấy từ biến môi trường; không ghi secret vào mã nguồn, log, fixture hay tài liệu được commit.
- **MongoDB:** chỉ dùng cho dữ liệu ứng dụng và metadata khi cần (ví dụ hội thoại, trạng thái xử lý, metadata văn bản/quan hệ). Không dùng MongoDB thay Qdrant cho vector search.
- **Rule Engine:** các phép tính pháp lý và xác định kết quả theo điều kiện pháp luật phải được thực hiện bởi rule/code xác định, có thể giải thích và kiểm tra. LLM chỉ diễn giải đầu vào/đầu ra, không tự tính hoặc tự quyết định kết quả pháp lý.
- **Legal Validator:** kiểm tra dữ kiện, điều kiện, hiệu lực, xung đột và căn cứ trước khi kết luận; khi không đủ dữ liệu phải nêu thiếu sót thay vì suy đoán.
- Giữ module theo trách nhiệm trong `kientruc.md`: ingestion, embeddings, retrieval, rag, router, legal tools, rules, validator, llm, database, api, frontend. Chỉ tạo phần cần cho thay đổi hiện tại.

## Citation và nội dung pháp lý

- LLM tuyệt đối không được tự tạo, đoán hoặc sửa citation.
- Citation trả cho người dùng phải lấy từ các legal chunks thực sự được retrieval trả về và dùng làm context.
- Mỗi chunk cần giữ định danh tài liệu và locator đủ cụ thể (văn bản/số hiệu, Điều, Khoản, Điểm hoặc mục tương ứng, đường dẫn nguồn khi phù hợp).
- Chỉ đưa citation vào câu trả lời nếu chunk có nguồn/locator tương ứng. Không có nguồn phù hợp thì nói rõ chưa tìm thấy căn cứ trong dữ liệu truy xuất.
- Giữ nguyên nội dung trích dẫn pháp luật; không để LLM diễn đạt lại bên trong phần được đánh dấu trích dẫn.
- Phân biệt văn bản, cơ quan ban hành, cấp hiệu lực, tình trạng hiệu lực theo thời điểm và phạm vi áp dụng. Không suy ra hiệu lực chỉ từ tên file hay điểm tìm kiếm.
- Toàn bộ văn bản hiện có được chủ dự án xác nhận còn hiệu lực tại thời điểm xác nhận. Thiết kế metadata vẫn phải hỗ trợ lịch sử hiệu lực và quan hệ sửa đổi/thay thế một phần hoặc toàn phần.

## Quy tắc code

- Viết Python rõ ràng, module hóa theo trách nhiệm; ưu tiên hàm nhỏ có input/output xác định và `pathlib` cho đường dẫn.
- Tránh đường dẫn tuyệt đối phụ thuộc máy cá nhân. Cấu hình đường dẫn, model, collection, database và API qua cấu hình/biến môi trường phù hợp.
- Không tạo abstraction, dependency, framework hoặc cấu hình tổng quát nếu thay đổi hiện tại không cần.
- Xử lý lỗi có ngữ cảnh; không nuốt lỗi hoặc bỏ qua tài liệu/chunk âm thầm.
- Dữ liệu tiếng Việt lưu/đọc UTF-8. Không làm sạch theo cách xóa số điều, chú thích, ngày tháng hoặc nội dung sửa đổi/bãi bỏ.
- Không ghi `.env`, API key, dữ liệu người dùng hoặc thông tin nhạy cảm vào Git.
- Thay đổi schema phải nhỏ, có đường chuyển đổi tương thích khi cần và cập nhật nơi đọc/ghi liên quan trong cùng phạm vi thay đổi.

## Kiểm tra bắt buộc sau mỗi thay đổi

- Sau **mỗi thay đổi mã nguồn**, chạy kiểm tra phù hợp trước khi chuyển sang việc khác: ít nhất compile/syntax check cho file Python vừa sửa (`python -m py_compile <file>`); nếu có test liên quan thì chạy test tương ứng.
- Nếu sửa nhiều file phụ thuộc nhau, kiểm tra cú pháp tất cả file đã sửa và chạy test nhỏ nhất bao phủ luồng thay đổi.
- Nếu thay đổi tài liệu/cấu hình không thực thi, kiểm tra nội dung và đường dẫn/tham chiếu; không cần chạy kiểm tra Python không liên quan.
- Không tự thêm hoặc chạy toàn bộ bộ test ngoài phạm vi nếu tốn kém/đòi dịch vụ bên ngoài; chạy kiểm tra cục bộ khả thi và báo rõ kiểm tra nào không chạy được.
- Không báo thay đổi đã kiểm chứng nếu lệnh kiểm tra chưa chạy hoặc thất bại. Nêu lệnh và kết quả trong phần bàn giao.

## Quy trình làm việc

1. Xác định yêu cầu và file/luồng bị ảnh hưởng; kiểm tra trạng thái Git để không ghi đè công việc có sẵn.
2. Thực hiện một thay đổi nhỏ, đủ để đáp ứng yêu cầu.
3. Chạy kiểm tra tương ứng ngay sau thay đổi.
4. Sửa lỗi phát hiện được và chạy lại kiểm tra.
5. Báo cáo file đã đổi, hành vi ảnh hưởng, kiểm tra đã chạy và giới hạn còn lại.

## Giới hạn chỉnh sửa

- Mỗi file được chỉnh sửa phải có dưới 300 dòng sau thay đổi.
- Chỉ sửa đúng hàm hoặc file được giao trong yêu cầu.
- Không tự ý viết lại logic, cấu trúc hoặc hành vi ở các module khác ngoài phạm vi được giao.
- Nếu cần thay đổi ngoài phạm vi để hoàn thành yêu cầu, phải nêu rõ lý do và xin xác nhận trước.

## Bảo vệ dữ liệu và vận hành

- Không xóa/ghi đè dữ liệu gốc hoặc embedding đã sinh nếu chưa được yêu cầu rõ ràng.
- Không gửi dữ liệu pháp lý, nội dung người dùng hoặc secret tới dịch vụ ngoài nếu luồng hiện tại chưa được cấu hình/cho phép.
- Với Qdrant/MongoDB/Gemini, giữ kết nối và thao tác bên ngoài ở adapter/module riêng; không làm cho import module hoặc chạy syntax check tự kết nối dịch vụ.
- Thao tác ingest hoặc re-index phải có thể chạy lặp lại an toàn, dùng ID ổn định và tránh nhân đôi dữ liệu.
