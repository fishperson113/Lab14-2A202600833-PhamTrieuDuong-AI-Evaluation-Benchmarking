import asyncio
import json
import os
import time

from dotenv import load_dotenv

from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import MultiJudgeConsensus
from engine.runner import BenchmarkRunner

load_dotenv()

# ---------------------------------------------------------------------------
# Cost thresholds
# ---------------------------------------------------------------------------
RELEASE_THRESHOLDS = {
    "min_avg_score": 3.0,
    "max_score_drop": 0.5,     # so với V1
    "max_cost_per_eval": 0.05,  # $0.05
    "min_hit_rate": 0.6,
    "max_latency": 5.0,         # seconds
}


async def benchmark_version(version: str, dataset: list,
                             runner: BenchmarkRunner) -> tuple:
    """Chạy benchmark cho 1 version Agent, trả về (results, summary)."""
    print(f"\n{'='*50}")
    print(f"  Benchmarking {version}...")
    print(f"{'='*50}")

    start_t = time.perf_counter()
    results = await runner.run_all(dataset, batch_size=5)
    elapsed = time.perf_counter() - start_t

    total = len(results)
    if total == 0:
        return results, None

    # Aggregate metrics
    summary = {
        "metadata": {
            "version": version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
        },
        "metrics": {
            "avg_score": round(
                sum(r["judge"]["final_score"] for r in results) / total, 4
            ),
            "hit_rate": round(
                sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
                4,
            ),
            "mrr": round(
                sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total, 4
            ),
            "agreement_rate": round(
                sum(r["judge"]["agreement_rate"] for r in results) / total, 4
            ),
            "avg_faithfulness": round(
                sum(r["ragas"]["faithfulness"] for r in results) / total, 4
            ),
            "avg_relevancy": round(
                sum(r["ragas"]["relevancy"] for r in results) / total, 4
            ),
            "avg_latency": round(
                sum(r["latency"] for r in results) / total, 4
            ),
            "pass_rate": round(
                sum(1 for r in results if r["status"] == "pass") / total, 4
            ),
            "total_cost": round(sum(r["total_cost"] for r in results), 6),
            "cost_per_eval": round(
                sum(r["total_cost"] for r in results) / total, 6
            ),
            "total_tokens": sum(
                r["token_usage"]["total"] for r in results
            ),
        },
    }
    return results, summary


def compute_regression(v1_summary: dict, v2_summary: dict) -> dict:
    """So sánh V2 với V1 → delta metrics."""
    m1 = v1_summary["metrics"]
    m2 = v2_summary["metrics"]

    delta = {
        "v1_version": v1_summary["metadata"]["version"],
        "v2_version": v2_summary["metadata"]["version"],
        "delta_avg_score": round(m2["avg_score"] - m1["avg_score"], 4),
        "delta_hit_rate": round(m2["hit_rate"] - m1["hit_rate"], 4),
        "delta_mrr": round(m2["mrr"] - m1["mrr"], 4),
        "delta_latency": round(m2["avg_latency"] - m1["avg_latency"], 4),
        "delta_cost_per_eval": round(
            m2["cost_per_eval"] - m1["cost_per_eval"], 6
        ),
        "delta_pass_rate": round(m2["pass_rate"] - m1["pass_rate"], 4),
        "delta_faithfulness": round(
            m2["avg_faithfulness"] - m1["avg_faithfulness"], 4
        ),
        "delta_relevancy": round(m2["avg_relevancy"] - m1["avg_relevancy"], 4),
    }

    # Release gate
    failures = []
    if delta["delta_avg_score"] < -RELEASE_THRESHOLDS["max_score_drop"]:
        failures.append(
            f"Score drop {delta['delta_avg_score']:.2f} exceeds "
            f"threshold {-RELEASE_THRESHOLDS['max_score_drop']}"
        )
    if m2["avg_score"] < RELEASE_THRESHOLDS["min_avg_score"]:
        failures.append(
            f"V2 score {m2['avg_score']:.2f} below minimum "
            f"{RELEASE_THRESHOLDS['min_avg_score']}"
        )
    if m2["hit_rate"] < RELEASE_THRESHOLDS["min_hit_rate"]:
        failures.append(
            f"V2 hit_rate {m2['hit_rate']:.2f} below minimum "
            f"{RELEASE_THRESHOLDS['min_hit_rate']}"
        )
    if m2["cost_per_eval"] > RELEASE_THRESHOLDS["max_cost_per_eval"]:
        failures.append(
            f"V2 cost_per_eval ${m2['cost_per_eval']:.4f} exceeds "
            f"${RELEASE_THRESHOLDS['max_cost_per_eval']}"
        )
    if m2["avg_latency"] > RELEASE_THRESHOLDS["max_latency"]:
        failures.append(
            f"V2 avg_latency {m2['avg_latency']:.2f}s exceeds "
            f"{RELEASE_THRESHOLDS['max_latency']}s"
        )

    decision = "ROLLBACK" if failures else "APPROVE"
    delta["decision"] = decision
    delta["failures"] = failures
    delta["reasoning"] = (
        f"Blocked by {len(failures)} gate(s): {'; '.join(failures)}"
        if failures
        else "All quality/cost/latency thresholds passed."
    )
    return delta


def generate_cost_report(v1_summary: dict, v2_summary: dict, regression: dict):
    """Báo cáo chi phí chi tiết và đề xuất tối ưu."""
    m1 = v1_summary["metrics"]
    m2 = v2_summary["metrics"]
    total = v1_summary["metadata"]["total"]

    lines = [
        "## Báo cáo Chi phí & Hiệu năng\n",
        f"**Dataset:** {total} test cases",
        f"",
        f"### V1 ({v1_summary['metadata']['version']})",
        f"- Total cost: ${m1['total_cost']:.4f}",
        f"- Cost per eval: ${m1['cost_per_eval']:.6f}",
        f"- Avg latency: {m1['avg_latency']:.2f}s",
        f"- Total tokens: {m1['total_tokens']}",
        f"- Elapsed: {v1_summary['metadata']['elapsed_seconds']:.1f}s",
        f"",
        f"### V2 ({v2_summary['metadata']['version']})",
        f"- Total cost: ${m2['total_cost']:.4f}",
        f"- Cost per eval: ${m2['cost_per_eval']:.6f}",
        f"- Avg latency: {m2['avg_latency']:.2f}s",
        f"- Total tokens: {m2['total_tokens']}",
        f"- Elapsed: {v2_summary['metadata']['elapsed_seconds']:.1f}s",
        f"",
        f"### Delta",
        f"- Cost delta: ${regression['delta_cost_per_eval']:+.6f}/eval",
        f"- Latency delta: {regression['delta_latency']:+.2f}s",
        f"",
        f"### Cost Optimization Recommendations",
    ]

    recs = []
    if m2["cost_per_eval"] > 0.005:
        recs.append(
            "- **Model cascade**: Dùng model rẻ (gpt-4o-mini) cho easy cases, "
            "model đắt cho hard cases → tiết kiệm ~40%"
        )
    if regr := regression.get("delta_cost_per_eval", 0) > 0:
        recs.append(
            "- **Caching kết quả judge**: Cache các câu hỏi trùng lặp → "
            "giảm API calls ~15%"
        )
    recs.append(
        "- **Batch processing**: Gộp nhiều câu hỏi vào 1 prompt → "
        "giảm input/output token overhead"
    )
    recs.append(
        "- **Dynamic judge selection**: Easy cases chỉ cần 1 judge → "
        "giảm 50% judge cost"
    )
    lines.extend(recs)
    return "\n".join(lines)


async def main():
    print("AI Evaluation Factory — Benchmark Pipeline")
    print("=" * 50)

    # 1. Kiểm tra dataset
    if not os.path.exists("data/golden_set.jsonl"):
        print("[ERROR] Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if len(dataset) < 5:
        print(f"[WARN] Dataset chỉ có {len(dataset)} cases. Nên tạo ít nhất 50.")
    print(f"[DATA] Loaded {len(dataset)} test cases\n")

    # 2. Khởi tạo components
    evaluator = RetrievalEvaluator()
    judge_consensus = MultiJudgeConsensus()
    print(f"[CONFIG] Multi-Judge với {judge_consensus.judge_count} models: "
          f"{[j.model_short() for j in judge_consensus.judges]}")

    # 3. Benchmark V1
    agent_v1 = MainAgent(version="V1")
    runner_v1 = BenchmarkRunner(agent_v1, evaluator, judge_consensus)
    v1_results, v1_summary = await benchmark_version("Agent_V1_Base", dataset, runner_v1)

    # 4. Benchmark V2
    agent_v2 = MainAgent(version="V2")
    runner_v2 = BenchmarkRunner(agent_v2, evaluator, judge_consensus)
    v2_results, v2_summary = await benchmark_version("Agent_V2_Optimized", dataset, runner_v2)

    if not v1_summary or not v2_summary:
        print("[ERROR] Benchmark failed. Kiểm tra API keys và dataset.")
        return

    # 5. Regression analysis
    print("\n📊 --- REGRESSION ANALYSIS ---")
    regression = compute_regression(v1_summary, v2_summary)
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {regression['delta_avg_score']:+.4f}")
    print(f"V1 Hit Rate: {v1_summary['metrics']['hit_rate']:.2%}")
    print(f"V2 Hit Rate: {v2_summary['metrics']['hit_rate']:.2%}")
    print(f"V1 MRR: {v1_summary['metrics']['mrr']:.4f}")
    print(f"V2 MRR: {v2_summary['metrics']['mrr']:.4f}")
    print(f"Cost per eval V1: ${v1_summary['metrics']['cost_per_eval']:.6f}")
    print(f"Cost per eval V2: ${v2_summary['metrics']['cost_per_eval']:.6f}")
    print(f"\n=> QUYẾT ĐỊNH: {regression['decision']}")
    if regression.get("failures"):
        for f in regression["failures"]:
            print(f"  [FAIL] {f}")

    # 6. Cohen's Kappa (inter-rater reliability)
    print("\n📊 --- INTER-RATER RELIABILITY (COHEN'S KAPPA) ---")
    # Collect scores from V2 results for Cohen's Kappa
    scores_by_judge = {}
    for r in v2_results:
        indiv = r["judge"].get("individual_scores", {})
        for model_name, score in indiv.items():
            scores_by_judge.setdefault(model_name, []).append(score)

    model_names = list(scores_by_judge.keys())
    if len(model_names) >= 2:
        kappa = MultiJudgeConsensus.cohens_kappa(
            scores_by_judge[model_names[0]],
            scores_by_judge[model_names[1]],
        )
        print(f"Cohen's Kappa ({model_names[0]} vs {model_names[1]}): "
              f"{kappa['kappa']:.4f} — {kappa['interpretation']}")
    else:
        print("[WARN] Không đủ judge models để tính Cohen's Kappa.")

    # 7. Retrieval correlation analysis
    print("\n📊 --- RETRIEVAL ↔ ANSWER QUALITY CORRELATION ---")
    ret_corr = await evaluator.evaluate_batch(dataset, v2_results)
    print(f"Avg Hit Rate: {ret_corr['avg_hit_rate']:.2%}")
    print(f"Avg MRR: {ret_corr['avg_mrr']:.4f}")
    print(evaluator.explain_retrieval_impact(ret_corr))

    # 8. Cost report
    print("\n💰 --- COST REPORT ---")
    cost_report = generate_cost_report(v1_summary, v2_summary, regression)
    print(cost_report)

    # 9. Lưu reports
    os.makedirs("reports", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    # Gộp thông tin regression vào V2 summary
    v2_summary["regression"] = regression
    v2_summary["retrieval_analysis"] = {
        "avg_hit_rate": ret_corr["avg_hit_rate"],
        "avg_mrr": ret_corr["avg_mrr"],
        "correlation_with_faithfulness": ret_corr.get(
            "correlation_with_faithfulness", {}
        ),
        "correlation_with_relevancy": ret_corr.get(
            "correlation_with_relevancy", {}
        ),
    }
    v2_summary["cost_report"] = cost_report

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Reports saved to reports/")

    # 10. Tóm tắt cuối cùng
    print(f"\n{'='*50}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*50}")
    print(f"  Version: V2_Optimized")
    print(f"  Total cases: {v2_summary['metadata']['total']}")
    print(f"  Avg score: {v2_summary['metrics']['avg_score']}/5")
    print(f"  Hit Rate: {v2_summary['metrics']['hit_rate']:.2%}")
    print(f"  Pass Rate: {v2_summary['metrics']['pass_rate']:.2%}")
    print(f"  Total cost: ${v2_summary['metrics']['total_cost']:.4f}")
    print(f"  Decision: {regression['decision']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
