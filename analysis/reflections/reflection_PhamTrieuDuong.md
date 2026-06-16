# Individual Reflection — Phạm Triều Dương

## Vai trò trong nhóm
- Phát triển toàn bộ hệ thống: Synthetic Data Generator, Multi-Judge Consensus Engine, Benchmark Runner, Regression Release Gate, và Failure Analysis.

## Đóng góp cụ thể
- **SDG (Synthetic Data Generator)**: Thiết kế dataset 64 cases (50 standard + 14 hard/adversarial) với mapping `expected_retrieval_ids` cho từng document, bao gồm các Red Teaming cases (prompt injection, out-of-context, ambiguous).
- **Multi-Judge Consensus Engine**: Triển khai `LLMJudge` với 2 models qua OpenAI-compatible gateway (gemini-3.1-flash-lite, gemini-3-flash), xử lý JSON parsing edge cases, retry logic, conflict detection và weighted consensus.
- **Retrieval Evaluation**: Tính Hit Rate, MRR, phân tích correlation giữa retrieval quality và answer quality.
- **Benchmark Runner**: Async pipeline chạy song song với batch processing, tracking token usage, latency, và cost.
- **Regression Gate**: So sánh V1 vs V2 với auto gate (ROLLBACK/APPROVE) dựa trên các ngưỡng chất lượng.
- **Position Bias & Cohen's Kappa**: Implement `check_position_bias()` và `cohens_kappa()` để đánh giá độ tin cậy của judge.

## Giải thích kỹ thuật

### MRR (Mean Reciprocal Rank)
MRR đo lường khả năng xếp hạng của Retrieval system. Nếu tài liệu đúng xuất hiện ở vị trí thứ k, điểm reciprocal rank = 1/k. MRR là trung bình của các reciprocal rank trên toàn bộ dataset. Ví dụ: doc mong muốn xuất hiện ở vị trí thứ 2 → RR = 1/2 = 0.5. Hệ thống đạt MRR = 0.5547, nghĩa là trung bình tài liệu đúng xuất hiện ở vị trí ~1.8.

### Cohen's Kappa
Hệ số đo lường độ đồng thuận giữa 2 judge, loại bỏ yếu tố ngẫu nhiên.
κ = (Po - Pe) / (1 - Pe), với Po là observed agreement, Pe là expected agreement by chance.
κ > 0.6 cho thấy 2 judge có độ đồng thuận đáng kể. Với agreement_rate = 0.96, κ rất cao.

### Position Bias
Hiện tượng judge model ưu tiên response được trình bày đầu tiên, bất kể chất lượng.
Phát hiện bằng cách đổi chỗ 2 response và so sánh kết quả. Nếu điểm khác biệt > 0.5, judge bị bias.

## Trade-off giữa Chi phí và Chất lượng
- Dùng gemini-3.1-flash-lite + gemini-3-flash qua gateway: $0.000588/case, chất lượng ổn định
- Dùng GPT-4o cho judge: ~$0.005/case, độ chính xác cao hơn nhưng tốn kém hơn 8-10x
- Giải pháp tối ưu: Model cascade — easy cases dùng model rẻ, hard cases dùng model đắt, có thể giảm ~30% cost

## Khó khăn gặp phải & Cách giải quyết
- **Hit Rate thấp (56%)**: Keyword index thiếu nhiều mapping ("sign", "email", "browser", "cost"...). Giải pháp: mở rộng KEYWORD_INDEX với 15+ entries mới.
- **Template response không chính xác**: Agent dùng template cứng thay vì LLM generation → 5/7 failures. Cần thay bằng LLM call thực sự.
- **JSON parse error từ judge model**: Retry logic + regex fallback + strip markdown fences để đảm bảo parse được điểm số.

## Kết luận
- Hoàn thành pipeline benchmark đầy đủ: SDG → Retrieval Eval → Multi-Judge → Regression → Reports.
- Điểm mạnh: Multi-Judge hoạt động ổn định (agreement_rate = 0.96), async performance tốt (~100s cho 128 cases).
- Điểm yếu: Agent generation còn là template cứng, retrieval cần thêm synonym expansion.
- Bài học: Trong RAG systems, cả Retrieval và Generation đều quan trọng — cải thiện một stage thôi chưa đủ.
