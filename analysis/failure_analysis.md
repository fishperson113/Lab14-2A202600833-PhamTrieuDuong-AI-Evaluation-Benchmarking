# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** {total_cases}
- **Tỉ lệ Pass/Fail:** {pass_count}/{fail_count}
- **Điểm trung bình Multi-Judge:** {avg_score} / 5.0
- **Retrieval Metrics:**
  - Hit Rate: {hit_rate}%
  - MRR: {mrr}
- **Chi phí:** ${total_cost} cho {total_cases} cases (${cost_per_eval}/eval)

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Tỉ lệ | Nguyên nhân dự kiến |
|----------|----------|-------|---------------------|
| Hallucination | {hallucination_count} | {hallucination_pct}% | Retriever lấy sai context hoặc context không đủ |
| Incomplete Answer | {incomplete_count} | {incomplete_pct}% | Agent không khai thác hết thông tin trong context |
| Low Accuracy | {accuracy_count} | {accuracy_pct}% | Knowledge base không có thông tin chính xác |
| Tone/Safety Issue | {tone_count} | {tone_pct}% | Prompt không có instruction về tone |
| Retrieval Miss | {retrieval_miss_count} | {retrieval_miss_pct}% | Hit Rate = 0 — retriever không tìm được tài liệu |

### Chi tiết:

**Lưu ý quan trọng về Retrieval Quality:**
- Khi Hit Rate = 1, điểm Faithfulness trung bình: {faithfulness_when_hit}
- Khi Hit Rate = 0, điểm Faithfulness trung bình: {faithfulness_when_miss}
- Chênh lệch: {faithfulness_delta}
- Điều này cho thấy: {faithfulness_conclusion}

---

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: [Mô tả ngắn — ví dụ: "Agent trả lời sai về chính sách bảo mật"]

1. **Symptom:** Agent trả lời sai thông tin về chính sách bảo mật dữ liệu.
2. **Why 1:** LLM không thấy thông tin chính xác trong context được cung cấp.
3. **Why 2:** Vector DB không tìm thấy tài liệu liên quan nhất (doc_data_1).
4. **Why 3:** Chunking strategy đã cắt tài liệu thành các chunk quá nhỏ, làm mất ngữ cảnh quan trọng.
5. **Why 4:** Keyword indexing không cover được các từ đồng nghĩa ("privacy" vs "protection").
6. **Root Cause:** Chiến lược Chunking không phù hợp và thiếu Synonym Expansion trong Retrieval pipeline.

### Case #2: [Mô tả ngắn — ví dụ: "Câu trả lời thiếu thông tin về pricing"]

1. **Symptom:** Agent chỉ trả lời một phần thông tin về subscription plans.
2. **Why 1:** LLM không được cung cấp đầy đủ context về các gói dịch vụ.
3. **Why 2:** Retriever chỉ tìm được 1 trong 3 tài liệu về billing.
4. **Why 3:** Câu hỏi chứa từ "price" nhưng index chỉ có "billing", "subscription".
5. **Why 4:** Thiếu mapping từ đồng nghĩa trong keyword index.
6. **Root Cause:** Thiếu từ điển đồng nghĩa (Synonym Dictionary) trong Retrieval stage.

### Case #3: [Mô tả ngắn — ví dụ: "Agent hallucinate khi không có context"]

1. **Symptom:** Agent tự bịa ra câu trả lời khi không tìm thấy tài liệu liên quan.
2. **Why 1:** LLM không được instruction rõ ràng về việc "chỉ trả lời dựa trên context".
3. **Why 2:** System prompt thiếu guardrails cho Out-of-Context scenarios.
4. **Why 3:** Không có cơ chế detect khi retrieval trả về 0 kết quả.
5. **Why 4:** Thiếu bước "No-Answer Detection" trong pipeline.
6. **Root Cause:** Thiếu guardrails trong Prompt Engineering và thiếu Fallback Strategy.

---

## 4. Kế hoạch cải tiến (Action Plan)

### Short-term (trong lab)
- [ ] **Thêm Synonym Expansion** vào Retrieval stage (VD: "price" ↔ "billing", "subscription")
- [ ] **Cập nhật System Prompt**: "Chỉ trả lời dựa trên context. Nếu không có context, hãy nói 'Tôi không tìm thấy thông tin'."
- [ ] **Thêm Fallback**: Nếu Hit Rate = 0, trả về "Tôi không biết" thay vì hallucinate
- [ ] **Cải thiện Keyword Index**: Mở rộng keyword mapping coverage

### Medium-term (sau lab)
- [ ] **Semantic Chunking**: Thay vì cắt cố định, chunk theo semantic units
- [ ] **Reranking Stage**: Thêm Cross-encoder reranker sau retrieval
- [ ] **No-Answer Classifier**: Model riêng để detect khi nào không nên trả lời
- [ ] **Hybrid Search**: Kết hợp keyword (BM25) + vector search

### Cost Optimization
- [ ] **Model Cascade**: Easy cases → gpt-4o-mini, Hard cases → gpt-4o
- [ ] **Judge Caching**: Cache kết quả judge cho các câu hỏi tương tự
- [ ] **Dynamic Judge Selection**: Easy cases chỉ dùng 1 judge thay vì 2

---

## 5. Kết luận

- Hệ thống hiện tại có điểm yếu chính ở **Retrieval Stage**: thiếu synonym handling và chunking chưa tối ưu.
- **Faithfulness và Relevancy** bị ảnh hưởng trực tiếp bởi chất lượng Retrieval (Δ = {faithfulness_delta}).
- **Multi-Judge consensus** hoạt động với Cohen's Kappa = {kappa} ({kappa_interp}).
- Đề xuất tập trung cải thiện Retrieval trước, sau đó mới tối ưu Generation.
- Chi phí hiện tại ${cost_per_eval}/eval có thể giảm ~30% bằng model cascade + caching.
