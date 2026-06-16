from typing import List, Dict


class RetrievalEvaluator:
    """
    Đánh giá chất lượng Retrieval của RAG Agent.
    Metrics: Hit Rate, MRR (Mean Reciprocal Rank).
    """

    @staticmethod
    def calculate_hit_rate(expected_ids: List[str], retrieved_ids: List[str],
                           top_k: int = 3) -> float:
        """
        Hit Rate: ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không?
        """
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    @staticmethod
    def calculate_mrr(expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank): 1 / position của expected_id đầu tiên tìm thấy.
        Nếu không tìm thấy, trả về 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict],
                             results: List[Dict]) -> Dict:
        """
        Đánh giá retrieval quality cho toàn bộ dataset.

        Args:
            dataset: golden_set entries (mỗi entry có 'expected_retrieval_ids')
            results: benchmark results (mỗi result có 'ragas.retrieval')

        Returns:
            Dict với avg_hit_rate, avg_mrr, và phân tích correlation
        """
        hit_rates = []
        mrrs = []
        faithfulness_scores = []
        relevancy_scores = []

        for case, res in zip(dataset, results):
            expected_ids = case.get("expected_retrieval_ids", [])
            agent_response = res.get("response", {})
            retrieved_ids = agent_response.get("retrieved_ids", [])

            hr = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            hit_rates.append(hr)
            mrrs.append(mrr)

            # Lấy faithfulness và relevancy từ kết quả judge để phân tích correlation
            ragas = res.get("ragas", {})
            faithfulness_scores.append(ragas.get("faithfulness", 0))
            relevancy_scores.append(ragas.get("relevancy", 0))

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

        # Phân tích correlation giữa Retrieval và Answer Quality
        correlation = self._analyze_correlation(
            hit_rates, faithfulness_scores, relevancy_scores
        )

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "hit_rates": hit_rates,
            "mrrs": mrrs,
            "correlation_with_faithfulness": correlation["faithfulness"],
            "correlation_with_relevancy": correlation["relevancy"],
            "total_cases": len(dataset),
        }

    @staticmethod
    def _analyze_correlation(hit_rates: List[float],
                              faithfulness: List[float],
                              relevancy: List[float]) -> Dict:
        """
        Phân tích correlation giữa Retrieval và Answer Quality.
        Hit Rate cao → Faithfulness/Relevancy cao?
        """
        if len(hit_rates) < 2:
            return {
                "faithfulness": "insufficient_data",
                "relevancy": "insufficient_data",
            }

        # So sánh điểm trung bình khi Hit Rate = 1 vs Hit Rate = 0
        hits_1_f = [f for hr, f in zip(hit_rates, faithfulness) if hr == 1]
        hits_0_f = [f for hr, f in zip(hit_rates, faithfulness) if hr == 0]
        hits_1_r = [r for hr, r in zip(hit_rates, relevancy) if hr == 1]
        hits_0_r = [r for hr, r in zip(hit_rates, relevancy) if hr == 0]

        avg_f_when_hit = sum(hits_1_f) / len(hits_1_f) if hits_1_f else 0
        avg_f_when_miss = sum(hits_0_f) / len(hits_0_f) if hits_0_f else 0
        avg_r_when_hit = sum(hits_1_r) / len(hits_1_r) if hits_1_r else 0
        avg_r_when_miss = sum(hits_0_r) / len(hits_0_r) if hits_0_r else 0

        return {
            "faithfulness": {
                "avg_when_hit": round(avg_f_when_hit, 4),
                "avg_when_miss": round(avg_f_when_miss, 4),
                "delta": round(avg_f_when_hit - avg_f_when_miss, 4),
                "interpretation": (
                    "Retrieval chất lượng cao giúp cải thiện faithfulness"
                    if avg_f_when_hit > avg_f_when_miss
                    else "Faithfulness không bị ảnh hưởng nhiều bởi retrieval"
                ),
            },
            "relevancy": {
                "avg_when_hit": round(avg_r_when_hit, 4),
                "avg_when_miss": round(avg_r_when_miss, 4),
                "delta": round(avg_r_when_hit - avg_r_when_miss, 4),
                "interpretation": (
                    "Retrieval chất lượng cao giúp cải thiện relevancy"
                    if avg_r_when_hit > avg_r_when_miss
                    else "Relevancy không bị ảnh hưởng nhiều bởi retrieval"
                ),
            },
        }

    @staticmethod
    def explain_retrieval_impact(correlation: Dict) -> str:
        """Giải thích mối liên hệ giữa Retrieval và Answer Quality."""
        lines = []
        lines.append("## Phân tích tác động của Retrieval đến Answer Quality\n")

        corr_f = correlation.get("correlation_with_faithfulness", {})
        corr_r = correlation.get("correlation_with_relevancy", {})

        if isinstance(corr_f, dict):
            lines.append(
                f"- **Faithfulness**: Khi Hit Rate = 1 → "
                f"{corr_f.get('avg_when_hit', 0):.2f}, "
                f"Khi Hit Rate = 0 → {corr_f.get('avg_when_miss', 0):.2f}"
                f" (Δ = {corr_f.get('delta', 0):+.2f})"
            )
            lines.append(f"  → {corr_f.get('interpretation', '')}")

        if isinstance(corr_r, dict):
            lines.append(
                f"- **Relevancy**: Khi Hit Rate = 1 → "
                f"{corr_r.get('avg_when_hit', 0):.2f}, "
                f"Khi Hit Rate = 0 → {corr_r.get('avg_when_miss', 0):.2f}"
                f" (Δ = {corr_r.get('delta', 0):+.2f})"
            )
            lines.append(f"  → {corr_r.get('interpretation', '')}")

        return "\n".join(lines)
