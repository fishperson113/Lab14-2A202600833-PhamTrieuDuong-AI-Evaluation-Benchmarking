# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 64
- **Tỉ lệ Pass/Fail:** 57/7
- **Điểm trung bình Multi-Judge:** 3.358 / 5.0
- **Retrieval Metrics:**
  - Hit Rate: 56.25%
  - MRR: 0.5547
- **Chi phí:** $0.037633 cho 64 cases ($0.000588/eval)

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Tỉ lệ | Nguyên nhân dự kiến |
|----------|----------|-------|---------------------|
| Hallucination | 0 | 0% | Agent trả lời "không biết" khi không có context — đúng behavior |
| Incomplete Answer | 5 | 71.4% | Agent dùng template cứng, chỉ concatenate snippet thay vì trích xuất thông tin cụ thể |
| Low Accuracy | 5 | 71.4% | Retrieval tìm đúng tài liệu nhưng agent response không trả lời chính xác câu hỏi |
| Tone/Safety Issue | 0 | 0% | Tất cả responses đều professional và an toàn |
| Retrieval Miss | 2 | 28.6% | Hit Rate = 0 — retriever không tìm được tài liệu do thiếu keyword mapping |

*Note: Một case có thể thuộc nhiều nhóm lỗi (ví dụ: Incomplete + Low Accuracy)*

### Chi tiết:

**Lưu ý quan trọng về Retrieval Quality:**
- Khi Hit Rate = 1, điểm Faithfulness trung bình: 0.7264
- Khi Hit Rate = 0, điểm Faithfulness trung bình: 0.6411
- Chênh lệch: +0.0853
- Điều này cho thấy: Retrieval chất lượng cao giúp cải thiện faithfulness, nhưng mức cải thiện chưa lớn do agent response bị giới hạn bởi template cứng

---

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: "How many API requests can I make per hour?" — Score 2.67/5

1. **Symptom:** Agent trả lời chung chung dạng "Dựa trên tài liệu hệ thống, how many api requests can i make per hour?... API Integration Guide: REST API endpoints..." mà không đưa ra con số 1000 requests/hour.
2. **Why 1:** Agent response là template cứng (`"Dựa trên tài liệu hệ thống, {question}... {snippet}"`) không trích xuất được thông tin định lượng từ context.
3. **Why 2:** Không có LLM generation stage thực sự — chỉ concatenate câu hỏi + snippet đầu tiên.
4. **Why 3:** Agent implementation (`MainAgent.query()`) không dùng LLM để paraphrase hay tổng hợp câu trả lời từ context.
5. **Root Cause:** Agent thiếu generation stage thực sự. Pipeline chỉ gồm retrieval + template ghép chuỗi, không có LLM call để sinh câu trả lời chính xác dựa trên context.

### Case #2: "How can I enable 2FA in configuration?" — Score 2.67/5

1. **Symptom:** Agent trả lời "Xin lỗi, tôi không tìm thấy thông tin liên quan" mặc dù tài liệu `doc_config_1` có chứa thông tin về `enable_2fa`.
2. **Why 1:** Retriever không tìm thấy document nào → retrieved_ids = [].
3. **Why 2:** Keyword "enable" không có trong KEYWORD_INDEX; từ "2fa" khớp với `doc_security_1` nhưng context không chứa config.
4. **Why 3:** Index chỉ build thủ công với limited keywords, không coverage cho cụm "enable 2FA" hay "two-factor" trong ngữ cảnh config.
5. **Root Cause:** Keyword Index thiếu entries cho các biến thể ngôn ngữ ("enable 2FA", "two-factor authentication config") và không có cơ chế synonym expansion hoặc text embedding để match ngữ nghĩa.

### Case #3: "How much does it cost?" — Score 2.83/5

1. **Symptom:** Agent trả lời "không tìm thấy thông tin" dù knowledge base có tài liệu về pricing (`doc_billing_1`).
2. **Why 1:** Từ "cost" không match với bất kỳ keyword nào trong KEYWORD_INDEX.
3. **Why 2:** INDEX có "price", "billing", "plan" nhưng không có "cost" — thiếu synonym mapping.
4. **Why 3:** Retrieval pipeline không có bước mở rộng từ khóa (stemming, lemmatization, synonym expansion).
5. **Root Cause:** Thiếu Synonym Dictionary trong Retrieval stage. Câu hỏi chứa "cost" nhưng index chỉ có "price" — đây là lỗi kinh điển của keyword-based retrieval không có semantic understanding.

---

## 4. Kế hoạch cải tiến (Action Plan)

### Short-term (trong lab)
- [x] **Thêm Synonym Expansion** vào KEYWORD_INDEX: "cost"→doc_billing_1, "enable"→doc_config_1+doc_security_1, "browser"→doc_troubleshoot_1, "email"→doc_account_1, v.v.
- [ ] **Cải thiện Agent Generation**: Thay template cứng bằng LLM call thực sự để paraphrase câu trả lời từ context
- [ ] **Thêm Fallback**: Khi Hit Rate = 0, trả về "Tôi không biết" thay vì cố gắng trả lời
- [ ] **Cải thiện Keyword Index**: Mở rộng coverage cho 10+ keywords còn thiếu

### Medium-term (sau lab)
- [ ] **Semantic Retrieval**: Thay keyword match bằng embedding-based vector search
- [ ] **Reranking Stage**: Thêm Cross-encoder reranker sau retrieval
- [ ] **Hybrid Search**: Kết hợp keyword (BM25) + vector search
- [ ] **No-Answer Classifier**: Model riêng để detect khi nào không nên trả lời

### Cost Optimization
- [ ] **Model Cascade**: Easy cases → gpt-4o-mini, Hard cases → gpt-4o (giảm ~40% cost)
- [ ] **Judge Caching**: Cache kết quả judge cho các câu hỏi tương tự
- [ ] **Dynamic Judge Selection**: Easy cases chỉ dùng 1 judge thay vì 2

---

## 5. Kết luận

- Hệ thống hiện tại có điểm yếu chính ở **Agent Generation Stage**: response là template cứng, không trích xuất được thông tin chính xác từ context dẫn đến 5/7 failures.
- **Retrieval Stage** cũng là vấn đề lớn: 28/64 cases (43.75%) có Hit Rate = 0 do keyword index thiếu coverage.
- **Faithfulness và Relevancy** bị ảnh hưởng bởi chất lượng Retrieval (Δ faithfulness = +0.0853 khi có hit).
- **Multi-Judge consensus** hoạt động tốt với Agreement Rate = 0.96 (gần như hoàn hảo).
- Đề xuất tập trung cải thiện Generation Stage trước (thay template = LLM thật), sau đó mới tối ưu Retrieval.
- Chi phí hiện tại $0.000588/eval có thể giảm ~30% bằng model cascade + caching.
