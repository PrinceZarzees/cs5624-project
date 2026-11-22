"""Cheap execution proxy: 1 validation problem per task per candidate."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphharness.agent import run_agent
from graphharness.data_loader import load_split
from graphharness.evaluate_answer import score_response
from graphharness.models import OpenAIModel
from graphharness.optimizer import sample_records
from graphharness.rubric_proxy import collect_candidates, pairwise_accuracy, spearman

candidates = collect_candidates(["experiments/qwen3_4b_opt_a/optimizer", "experiments/qwen3_4b/optimizer"])
val = load_split("val")
subset = sample_records(val, 1, random.Random(5624))
model = OpenAIModel("qwen3-4b-q4", base_url="http://127.0.0.1:8003/v1", api_key="EMPTY", max_tokens=512, timeout_s=300)
rows = []
for i, c in enumerate(candidates, 1):
    correct = parsed = 0
    for record in subset:
        result = run_agent(c["harness"], model, record, max_steps=8, save_messages=False)
        score = score_response(record, result.response or None)
        correct += bool(score["correct"]); parsed += bool(score["parsed"])
    rows.append({"id": c["id"], "val_accuracy": c["val_accuracy"],
                 "cheap_accuracy": correct / len(subset), "cheap_parsed": parsed / len(subset)})
    print(i, len(candidates), c["id"], rows[-1], flush=True)
x = [r["cheap_accuracy"] for r in rows]; y = [r["val_accuracy"] for r in rows]
best = max(rows, key=lambda r: r["val_accuracy"])
ranked = sorted(rows, key=lambda r: r["cheap_accuracy"], reverse=True)
payload = {"n": len(rows), "subset_per_task": 1, "subset_n": len(subset),
           "spearman": spearman(x, y), "pairwise": pairwise_accuracy(x, y),
           "top3_recall": best["id"] in [r["id"] for r in ranked[:3]],
           "best_actual": best, "top3": ranked[:3], "rows": rows}
out = Path("experiments/review_rubric_proxy/cheap_proxy_results.json")
out.write_text(json.dumps(payload, indent=2))
print(json.dumps({k: payload[k] for k in ["spearman", "pairwise", "top3_recall"]}))
