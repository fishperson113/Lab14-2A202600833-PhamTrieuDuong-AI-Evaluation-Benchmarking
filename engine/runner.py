import asyncio
import time
from typing import List, Dict

from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import MODEL_PRICING


class BenchmarkRunner:
    """
    Orchestrator chạy benchmark cho Agent:
    1. Gọi Agent → response
    2. Đánh giá Retrieval (Hit Rate, MRR)
    3. Multi-Judge consensus
    4. Tính cost & token usage
    """

    def __init__(self, agent, evaluator: RetrievalEvaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict, case_idx: int = 0) -> Dict:
        """Chạy benchmark cho 1 test case."""
        start = time.perf_counter()

        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start

        # 2. Retrieval evaluation
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        hit_rate = self.evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.evaluator.calculate_mrr(expected_ids, retrieved_ids)

        # 3. Multi-judge
        judge_result = await self.judge.evaluate(
            test_case["question"],
            response["answer"],
            test_case.get("expected_answer", ""),
        )

        # 4. Tính cost cho Agent (simulated pricing)
        meta = response.get("metadata", {})
        agent_model = meta.get("model", "gpt-4o-mini")
        pricing = MODEL_PRICING.get(agent_model, {"input": 0, "output": 0})
        in_tok = meta.get("input_tokens", 0)
        out_tok = meta.get("output_tokens", 0)
        agent_cost = (in_tok / 1_000_000 * pricing["input"]) + (
            out_tok / 1_000_000 * pricing["output"]
        )

        # 5. Faithfulness & Relevancy từ judge scores
        #    (proxy: accuracy → faithfulness, average accuracy+tone → relevancy)
        detailed = judge_result.get("detailed_scores", {})
        acc_vals = []
        tone_vals = []
        safe_vals = []
        for _, sc in detailed.items():
            acc_vals.append(sc.get("accuracy", 3))
            tone_vals.append(sc.get("tone", 3))
            safe_vals.append(sc.get("safety", 3))

        avg_acc = (sum(acc_vals) / len(acc_vals)) if acc_vals else 3
        avg_tone = (sum(tone_vals) / len(tone_vals)) if tone_vals else 3
        avg_safe = (sum(safe_vals) / len(safe_vals)) if safe_vals else 3

        faithfulness = round((avg_acc + avg_safe) / 10, 4)   # 0-1 scale
        relevancy = round((avg_acc + avg_tone) / 10, 4)      # 0-1 scale

        total_cost = agent_cost + judge_result.get("total_cost", 0)

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": round(latency, 3),
            "response": {
                "retrieved_ids": retrieved_ids,
                "contexts": response.get("contexts", []),
                "total_tokens": meta.get("total_tokens", 0),
            },
            "token_usage": {
                "input": in_tok,
                "output": out_tok,
                "total": in_tok + out_tok,
            },
            "agent_cost": round(agent_cost, 6),
            "judge_cost": round(judge_result.get("total_cost", 0), 6),
            "total_cost": round(total_cost, 6),
            "ragas": {
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "retrieval": {"hit_rate": hit_rate, "mrr": mrr},
            },
            "judge": {
                "final_score": judge_result["final_score"],
                "agreement_rate": judge_result["agreement_rate"],
                "individual_scores": judge_result["individual_scores"],
                "detailed_scores": detailed,
                "reasoning": judge_result.get("resolution", ""),
                "conflict": judge_result.get("conflict", False),
            },
            "status": "pass" if judge_result["final_score"] >= 3 else "fail",
            "idx": case_idx,
        }

    async def run_all(self, dataset: List[Dict],
                      batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song với batch_size để tránh rate limit.
        """
        results = []
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [
                self.run_single_test(case, idx)
                for idx, case in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"  [PROGRESS] {min(i + batch_size, total)}/{total} cases")
        return results
