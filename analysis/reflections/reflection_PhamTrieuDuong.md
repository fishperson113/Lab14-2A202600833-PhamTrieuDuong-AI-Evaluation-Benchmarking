# Individual Reflection — Phạm Triều Dương

## Vai trò trong nhóm
- [Ví dụ: Phát triển Multi-Judge Consensus Engine & Benchmark Runner]

## Đóng góp cụ thể
- [Mô tả các module đã implement, kèm link Git commit nếu có]
- Ví dụ: Triển khai LLMJudge với OpenAI GPT-4o + Anthropic Claude
- Ví dụ: Implement Cohen's Kappa và Position Bias detection

## Giải thích kỹ thuật

### MRR (Mean Reciprocal Rank)
MRR đo lường khả năng xếp hạng của Retrieval system. Nếu tài liệu đúng xuất hiện ở vị trí thứ k, điểm reciprocal rank = 1/k. MRR là trung bình của các reciprocal rank trên toàn bộ dataset.

### Cohen's Kappa
Hệ số đo lường độ đồng thuận giữa 2 judge, loại bỏ yếu tố ngẫu nhiên.
κ = (Po - Pe) / (1 - Pe), với Po là observed agreement, Pe là expected agreement by chance.
κ > 0.6 cho thấy 2 judge có độ đồng thuận đáng kể.

### Position Bias
Hiện tượng judge model ưu tiên response được trình bày đầu tiên, bất kể chất lượng.
Phát hiện bằng cách đổi chỗ 2 response và so sánh kết quả.

## Trade-off giữa Chi phí và Chất lượng
- Dùng GPT-4o cho judge: $X.XX/case, độ chính xác cao
- Dùng GPT-4o-mini cho judge: $Y.YY/case, chất lượng thấp hơn X%
- Giải pháp tối ưu: Model cascade — easy cases dùng model rẻ, hard cases dùng model đắt

## Khó khăn gặp phải & Cách giải quyết
- [Ví dụ: Xử lý API rate limit bằng batch processing]
- [Ví dụ: Xử lý JSON parse error từ Claude bằng cách strip markdown fences]

## Kết luận
- [Bài học rút ra, điều làm được và chưa làm được]
