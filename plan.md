# Lab 14: AI Evaluation Factory — Plan triển khai

## 1. Bức tranh toàn cảnh (Big Picture)

Ta đang xây dựng một **hệ thống benchmark AI Agent tự động**. Luồng hoạt động:

```
Golden Dataset (data/golden_set.jsonl) 
    → BenchmarkRunner (chạy Agent trên từng test case)
        → RetrievalEvaluator (tính Hit Rate, MRR)
        → LLMJudge (nhiều model chấm điểm)
    → So sánh V1 vs V2 (Regression)
    → Quyết định Release / Rollback
    → Xuất reports + phân tích lỗi
```

**Hiện tại code là stubs/placeholders — cần implement thật.**

---

## 2. Cấu trúc thư mục & file cần sửa

| File | Vai trò | Hiện tại | Cần làm |
|------|---------|----------|---------|
| `data/synthetic_gen.py` | Tạo Golden Dataset | Trả 1 câu mẫu | Sinh 50+ câu hỏi từ tài liệu, kèm `expected_retrieval_ids` |
| `data/golden_set.jsonl` | Dataset đầu vào | Không có (tạo bằng `synthetic_gen.py`) | Phải có 50+ dòng |
| `agent/main_agent.py` | Agent cần benchmark | Mock trả lời cứng | Thay = Agent thật hoặc giả lập thực tế |
| `engine/retrieval_eval.py` | Đo Retrieval chất lượng | `evaluate_batch` là placeholder | Tính Hit Rate & MRR trên dataset |
| `engine/llm_judge.py` | Multi-Judge chấm điểm | Hardcode score 4 & 3 | Gọi 2 model thật (GPT + Claude) |
| `engine/runner.py` | Async orchestrator | Gọi đúng pattern | Cần log token usage, latency |
| `main.py` | Entry point | Stub classes | Tích hợp real components |
| `reports/summary.json` | Kết quả tổng quan | Tự sinh | Format đúng check_lab.py |
| `reports/benchmark_results.json` | Chi tiết từng case | Tự sinh | Đầy đủ từng case |
| `analysis/failure_analysis.md` | 5 Whys | Template rỗng | Điền thật |

---

## 3. Phân công công việc (Team 4-6 người)

### 👥 Nhóm Data (1-2 người) — Dataset & Retrieval

**Nhiệm vụ:**
1. **Sinh Golden Dataset 50+ cases** (`data/synthetic_gen.py`)
   - Dùng API (OpenAI/Claude) để tạo QA pairs từ tài liệu gốc
   - Mỗi case cần: `question`, `expected_answer`, `context`, `expected_retrieval_ids` (mảng ID)
   - Thêm **10-15 Red Teaming cases**: prompt injection, out-of-context, conflicting info, ambiguous (tham khảo `data/HARD_CASES_GUIDE.md`)
   - Phân bố độ khó: 30% easy, 40% medium, 30% hard

2. **Implement RetrievalEvaluator** (`engine/retrieval_eval.py`)
   - `calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)` → float
   - `calculate_mrr(expected_ids, retrieved_ids)` → float
   - `evaluate_batch(dataset)` → `{"avg_hit_rate": ..., "avg_mrr": ...}`
   - **Chứng minh**: notebook/markdown giải thích correlation giữa Retrieval Quality và Answer Quality

**File chạm tới:** `data/synthetic_gen.py`, `data/golden_set.jsonl`, `engine/retrieval_eval.py`, `analysis/failure_analysis.md`

**Điểm rubric:** Dataset & SDG (10đ) + Retrieval Evaluation (10đ) + cá nhân (15đ)

---

### 👥 Nhóm AI/Backend (1-2 người) — Multi-Judge & Eval Engine

**Nhiệm vụ:**
1. **Implement LLMJudge** (`engine/llm_judge.py`)
   - Tích hợp **2 model thật**: GPT-4o (OpenAI) + Claude-3.5-Sonnet (Anthropic)
   - Rubric chi tiết: Accuracy (1-5), Professionalism (1-5), Safety (1-5)
   - `evaluate_multi_judge(q, a, gt)` → `{"final_score", "agreement_rate", "individual_scores", "reasoning"}`
   - Consensus logic: nếu 2 model chênh > 1 điểm → auto detect conflict → gọi judge thứ 3 (hoặc lấy trung bình + cảnh báo)
   - `check_position_bias(a, b)`: đổi chỗ response, kiểm tra judge có thiên vị vị trí không

2. **Nâng cấp BenchmarkRunner** (`engine/runner.py`)
   - Log `tokens_used`, `latency`, `cost` cho mỗi case
   - Cost tracking: tính `$` dựa trên token count + model pricing

3. **Tích hợp vào main.py**
   - Thay thế các stub class `ExpertEvaluator`, `MultiModelJudge` bằng implementation thật

**File chạm tới:** `engine/llm_judge.py`, `engine/runner.py`, `main.py`, `reports/summary.json`, `reports/benchmark_results.json`

**Điểm rubric:** Multi-Judge consensus (15đ) + Performance Async (10đ) + cá nhân (15đ)

---

### 👥 Nhóm DevOps/Analyst (1-2 người) — Regression, Báo cáo, Phân tích

**Nhiệm vụ:**
1. **Regression Testing** (`main.py`)
   - Chạy benchmark cho 2 version Agent: V1 (base) vs V2 (optimized)
   - Tính delta: `Δ = metric_v2 - metric_v1`
   - Release Gate logic:
     ```python
     if avg_score_drop > 0.5 or hit_rate_drop > 0.1:
         decision = "ROLLBACK"
     elif cost_per_eval > threshold:
         decision = "ROLLBACK" 
     else:
         decision = "APPROVE"
     ```

2. **Performance & Cost report**
   - Benchmark pipeline **< 2 phút cho 50 cases** (dùng asyncio + batch)
   - Xuất cost report: `$0.0X / eval`, tổng `$X.XX`
   - Đề xuất giảm 30% cost: caching, dùng model rẻ hơn cho easy cases

3. **Failure Analysis** (`analysis/failure_analysis.md`)
   - Phân cụm lỗi: Hallucination / Incomplete / Tone Mismatch / Wrong Retrieval
   - **5 Whys** cho 3 case tệ nhất (đi sâu vào: chunking → retrieval → prompt → generation)

4. **Individual Reports** (`analysis/reflections/reflection_[Tên_SV].md`)
   - Mỗi thành viên tự viết reflection

**File chạm tới:** `main.py` (regression gate logic), `analysis/failure_analysis.md`, `reports/summary.json`

**Điểm rubric:** Regression Testing (10đ) + Performance Async (10đ - phần cost) + Failure Analysis (5đ) + cá nhân (10đ)

---

## 4. Lộ trình thực hiện (4 tiếng)

### Giai đoạn 1 (45p) — Setup & Dataset
- [ ] Nhóm Data: clone repo, `pip install -r requirements.txt`, setup `.env` với API keys
- [ ] Nhóm Data: implement `synthetic_gen.py` — sinh 50+ QA pairs từ tài liệu
- [ ] Nhóm Data: thêm 10-15 Red Teaming cases (adversarial + edge cases)
- [ ] Nhóm Data: chạy `python data/synthetic_gen.py` → tạo `data/golden_set.jsonl`
- [ ] Cả nhóm: thống nhất format dataset (field names)

### Giai đoạn 2 (90p) — Core Engine
- [ ] Nhóm AI/Backend: implement `LLMJudge.evaluate_multi_judge()` gọi 2 model thật
- [ ] Nhóm AI/Backend: implement consensus logic (xử lý conflict score)
- [ ] Nhóm AI/Backend: implement `BenchmarkRunner` với cost + token tracking
- [ ] Nhóm Data: implement `RetrievalEvaluator.evaluate_batch()` 
- [ ] Nhóm Data: implement `MainAgent.query()` thật hoặc semi-real
- [ ] Nhóm DevOps: implement regression gate logic trong `main.py`

### Giai đoạn 3 (60p) — Chạy & Phân tích
- [ ] Chạy `python main.py` → tạo `reports/summary.json` + `reports/benchmark_results.json`
- [ ] Nhóm DevOps: phân tích kết quả, viết failure analysis
- [ ] Nhóm Data: chứng minh correlation retrieval ↔ answer quality
- [ ] Nhóm AI: chạy position bias test
- [ ] Cả nhóm: viết Individual Reports

### Giai đoạn 4 (45p) — Tối ưu & Hoàn thiện
- [ ] Tối ưu Agent dựa trên kết quả failure analysis
- [ ] Chạy lại benchmark cho V2
- [ ] Verify regression gate hoạt động
- [ ] Chạy `python check_lab.py` — zero lỗi
- [ ] Push lên GitHub/GitLab

---

## 5. Lưu ý "ăn điểm"

### Retrieval (15%)
- Không chỉ tính Hit Rate/MRR — phải có phân tích correlation với answer quality
- Chỉ ra chunk nào gây hallucination cụ thể

### Multi-Judge (20%)
- PHẢI dùng 2 model khác nhau (GPT + Claude), không được fake
- Cohen's Kappa là điểm cộng lớn (thể hiện technical depth)
- Position bias test = điểm vàng

### Cost Optimization (15%)
- Cost per eval phải có số liệu thật
- Đề xuất: dùng model rẻ cho easy cases, cache cho câu hỏi trùng

### 5 Whys (20%)
- Phân tích sâu đến tận gốc: chunking strategy → retrieval → prompt
- Không dừng ở "LLM trả lời sai"

---

## 6. Checklist nộp bài

- [ ] `data/golden_set.jsonl` — 50+ cases
- [ ] `reports/summary.json` — đúng format (metrics: avg_score, hit_rate, agreement_rate)
- [ ] `reports/benchmark_results.json` — chi tiết từng case
- [ ] `analysis/failure_analysis.md` — đã điền 5 Whys
- [ ] `analysis/reflections/reflection_*.md` — mỗi thành viên
- [ ] Chạy `python check_lab.py` → không lỗi
- [ ] File `.env` không bị push lên git
