#!/usr/bin/env python3
"""Check a new split is disjoint from the original six splits."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graphharness.data_loader import load_jsonl

def key(rec):
    edges = sorted(tuple(sorted(map(str, e[:2]))) for e in rec["graph"]["edges"])
    blob = json.dumps([rec["task"], rec["graph"]["nodes"], edges, rec.get("params", {})],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()

def main(new_path, old_dir="data"):
    new = load_jsonl(new_path)
    old_keys = {}
    for split in ["train","val","test_small","test_medium","test_large","test_stress"]:
        p = Path(old_dir) / f"{split}.jsonl"
        if p.exists():
            for r in load_jsonl(p):
                old_keys[key(r)] = r["id"]
    overlap = [(r["id"], old_keys[key(r)]) for r in new if key(r) in old_keys]
    print(f"new={len(new)} old={len(old_keys)} overlap={len(overlap)}")
    for a,b in overlap[:10]: print("OVERLAP",a,b)
    return 1 if overlap else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv)>1 else "data_final/test_small_fresh.jsonl"))
