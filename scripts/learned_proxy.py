"""Leave-one-out learned static proxy from harness text features."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from graphharness.agent import Harness
from graphharness.rubric_proxy import collect_candidates, spearman, pairwise_accuracy

cands = collect_candidates(["experiments/qwen3_4b_opt_a/optimizer", "experiments/qwen3_4b/optimizer"])
keywords = ["verify", "check", "print", "networkx", "nx.", "diameter", "clique",
            "component", "timeout", "node id", "answer", "exact", "heuristic"]
rows=[]
for c in cands:
    h=c["harness"]; text=(h.system_prompt+"\n"+"\n".join(h.rules)).lower()
    feats=[len(h.system_message()), len(h.rules),
           sum(1 for k in keywords if k in text),
           int("print(" in text), int("diameter" in text), int("clique" in text),
           int("component" in text), int("verify" in text or "check" in text),
           int(h.allow_tools), int(h.graph_view=="edge_list"),
           int(bool(re.search(r"\bnode\s+\d+\b", text)))]
    rows.append((c["id"], c["val_accuracy"], feats))
X=np.array([r[2] for r in rows],dtype=float); y=np.array([r[1] for r in rows])
mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Xs=(X-mu)/sd
lam=1.0
pred=[]
for i in range(len(y)):
    mask=np.arange(len(y))!=i
    A=Xs[mask].T@Xs[mask]+lam*np.eye(Xs.shape[1]); b=Xs[mask].T@y[mask]
    w=np.linalg.solve(A,b); pred.append(float(Xs[i]@w))
payload={"n":len(y),"spearman_loo":spearman(pred,y.tolist()),
         "pairwise_loo":pairwise_accuracy(pred,y.tolist()),
         "top3":sorted([{"id":r[0],"val_accuracy":r[1],"loo":p} for r,p in zip(rows,pred)],
                       key=lambda x:x["loo"],reverse=True)[:3],
         "rows":[{"id":r[0],"val_accuracy":r[1],"loo":p} for r,p in zip(rows,pred)]}
Path("experiments/review_rubric_proxy/learned_proxy_results.json").write_text(json.dumps(payload,indent=2))
print(json.dumps({k:payload[k] for k in ["spearman_loo","pairwise_loo"]},indent=2))
