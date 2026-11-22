import json
from pathlib import Path
from hda.agent import RunConfig, ScriptedClient, run

ROOT = Path(__file__).resolve().parent.parent
REF = (ROOT / "reference" / "triangle_count.cpp").read_text()
WRONG = REF.replace('cout<<c<<"\\n";', 'cout<<c+1<<"\\n";')

def test_loop_with_scripted_client(tmp_path):
    replies = [
        f"Candidates: edge count, triangle count, components.\n<statement>count edges</statement>\n```cpp\n{WRONG}```\n<action>test</action>",
        "<statement>count triangles</statement>\n<action>submit</action>",       # submits WRONG (last cpp block)
        f"<statement>Count triangles in the graph.</statement>\n```cpp\n{REF}```\n<action>test</action>",
        "<action>submit</action>",
    ]
    cfg = RunConfig(task="triangle_count", condition="hidden", k=2, n_iter=2, steps=3, data=str(ROOT / "data/pilot"),
                    out=str(tmp_path / "run"), test_splits=["test_large"])
    final = run(cfg, ScriptedClient(replies), judge_client=ScriptedClient(["EQUIVALENT. same."]))
    assert final["val_curve"][1] == 1.0 and final["best_it"] == 2
    assert final["statements"] == ["count triangles", "Count triangles in the graph."]
    assert final["test"]["test_large"]["accuracy"] == 1.0
    assert final["statement_judge"] == "EQUIVALENT"
    kinds = [json.loads(l)["kind"] for l in open(tmp_path / "run" / "log.jsonl")]
    assert {"prompt", "reply", "tool", "submission", "iteration", "test", "judge"} <= set(kinds)

def test_hardcoded_submission_rejected(tmp_path):
    hard = '#include <cstdio>\nint main(){ puts("12345"); puts("67890"); }'
    cfg = RunConfig(task="mst_weight", k=3, n_iter=1, steps=1, data=str(ROOT / "data/pilot"), out=str(tmp_path / "r"), test_splits=[])
    final = run(cfg, ScriptedClient([f"```cpp\n{hard}```\n<action>submit</action>"]))
    assert final["val_curve"] == [0.0]

def test_stress_tool(tmp_path):
    gen = '#include <cstdio>\nint main(){ int n=100000; printf("%d %d 0 0\\n", n, n-1); for(int i=0;i+1<n;i++) printf("%d %d\\n", i, i+1); printf("0\\n"); }'
    cc = (ROOT / "reference" / "connected_components.cpp").read_text()
    replies = [f"```cpp\n{cc}```\n```gen\n{gen}```\n<action>stress</action>", "<action>submit</action>"]
    cfg = RunConfig(task="connected_components", k=1, n_iter=1, steps=2, data=str(ROOT / "data/pilot"), out=str(tmp_path / "s"), test_splits=[])
    run(cfg, ScriptedClient(replies))
    tools = [json.loads(l) for l in open(tmp_path / "s" / "log.jsonl") if '"tool"' in l]
    assert tools[0]["result"]["output"].strip() == "1" and tools[0]["gen_bytes"] > 100000

def test_hidden_prompt_omits_statement(tmp_path):
    from hda.tasks import TASKS
    replies = ["<statement>a path metric</statement>\n<action>submit</action>"]
    cfg = RunConfig(task="shortest_path", condition="hidden", k=2, n_iter=1, steps=1,
                    data=str(ROOT / "data/pilot"), out=str(tmp_path / "h"), test_splits=[])
    run(cfg, ScriptedClient(replies))
    prompts = [json.loads(l) for l in open(tmp_path / "h" / "log.jsonl") if '"prompt"' in l]
    blob = prompts[0]["system"] + prompts[0]["user"]
    assert TASKS["shortest_path"].statement not in blob
    assert "HIDDEN" in prompts[0]["user"]
