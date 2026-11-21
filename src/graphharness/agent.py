"""The agent harness: prompts, tool protocol, baselines and the interaction loop.

A harness is intentionally just data (system prompt, rules, graph rendering and
limits).  That makes it cheap to save, edit by an optimizer and ablate rule by
rule.  The model talks to the sandbox through either JSON tool calls or
fenced Python blocks, and finishes with ``<answer>[...]</answer>``.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .encoding import STYLES
from .jsonutil import loads_relaxed
from .models import ChatResult, Model
from .csandbox import CSandbox
from .sandbox import PythonSandbox, SandboxLimits
from .tasks import get_task

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+.-]+)?[ \t]*\r?\n(?P<code>.*?)```",
    re.DOTALL,
)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]", re.DOTALL)

TOOL_PROTOCOL_C = """\
## C tool and answer format

You may write C code and compile it with `gcc`. NetworkX is not available. The
working directory contains `graph_io.h`, `graph_io.c`, `graph.txt` and
`params.txt`.

`graph.txt` format: first line `n m`, then `m` lines `u v weight`. Node ids are
integers; `Graph` from `graph_io.h` exposes `n`, `m`, `max_id`, `eu`, `ev`,
`ew`, plus `load_graph()` and `get_param("source")`.

Reply with a ```c fenced block or a JSON call. Minimal example:

```c
#include "graph_io.h"
int main() {
    Graph g = load_graph();
    printf("nodes=%d edges=%d\n", g.n, g.m);
    free_graph(&g);
    return 0;
}
```

{"tool": "run_c", "code": "#include \"graph_io.h\"\nint main(){...}"}

When ready, give the final answer in the task-requested format, e.g.
<answer>[1, 2, 3]</answer>.
"""

TOOL_PROTOCOL = """\
## Tools and answer format

You can run Python in a sandbox. NetworkX is installed. Every code run is a
fresh process in a working directory with these names already defined:

- `G`: a `networkx.Graph` containing the problem graph
- `params`: a dict with task-specific inputs such as `source` and `target`
- `nodes`: the list of node ids

Files you write in the working directory persist between runs. Print a small
amount of output; large outputs are truncated. A bare final expression is
echoed automatically, but calling `print(...)` is safer.

Use a tool by replying with a fenced Python block, for example:

```python
print(G.number_of_nodes())
```

A JSON tool call is also accepted, but it must be valid JSON:

{"tool": "run_python", "code": "print(G.number_of_nodes())"}

When you are ready, finish with the answer in the format the task asks for.
For node-list tasks the default is:

<answer>[1, 2, 3]</answer>

or, equivalently:

{"tool": "answer", "answer": [1, 2, 3]}

For a number, yes/no, label, mapping, or two lists, put that value inside the
tags instead, for example <answer>42</answer> or <answer>yes</answer>.
Use `[]` for a negative node-list answer. Do not invent node ids.
"""


@dataclass
class Harness:
    """A frozen, serializable agent configuration."""

    name: str
    system_prompt: str
    rules: list[str] = field(default_factory=list)
    graph_view: str = "summary"
    max_steps: int = 8
    tool_timeout_s: float = 30.0
    memory_mb: int = 2048
    max_output_bytes: int = 16_000
    temperature: float = 0.0
    allow_tools: bool = True
    tool_backend: str = "python"
    tool_preamble: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.graph_view not in {"summary", *STYLES}:
            raise ValueError(f"unknown graph_view {self.graph_view!r}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.allow_tools and self.graph_view == "summary":
            raise ValueError("allow_tools=False needs a text graph view (edge_list or adjacency_list)")
        if self.tool_backend not in {"python", "c"}:
            raise ValueError(f"unknown tool_backend {self.tool_backend!r}")

    def system_message(self) -> str:
        parts = [self.system_prompt.strip()]
        rules = [r.strip() for r in self.rules if str(r).strip()]
        if rules:
            parts.append("Rules:\n" + "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1)))
        if self.tool_preamble:
            parts.append("Helper code already loaded in the sandbox:\n```python\n"
                         + self.tool_preamble.strip() + "\n```")
        if self.allow_tools:
            parts.append((TOOL_PROTOCOL_C if self.tool_backend == "c" else TOOL_PROTOCOL).strip())
        else:
            parts.append(
                "## No tools\nYou cannot run code or call tools for this task. "
                "Read the graph text carefully and reason directly. Give the "
                "final answer in the requested format."
            )
        return "\n\n".join(p for p in parts if p)

    def user_message(self, record: dict) -> str:
        graph_text = render_graph(record, self.graph_view)
        question = record["question"].strip()
        if self.tool_backend == "c":
            hint = ("Use the C tool (`run_c`). The graph is in `graph.txt`; "
                    "include `graph_io.h` and compile with gcc.")
        elif record.get("graphs"):
            hint = ("The two graphs are available as `G1` and `G2` (and also `G` = G1) "
                    "in the sandbox. Use `run_python` to inspect or compute the answer.")
        elif self.graph_view == "summary":
            hint = ("The full edge list is not shown. Use the `run_python` tool to inspect "
                    "`G` or compute the answer.")
        else:
            hint = "Use the `run_python` tool to check or compute the answer."
        return (f"## Graph\n{graph_text}\n\n## Task\n{question}\n\n{hint}\n\n"
                f"{answer_instruction(record)}")

    def sandbox_limits(self) -> SandboxLimits:
        return SandboxLimits(
            timeout_s=self.tool_timeout_s,
            cpu_seconds=max(1, int(round(self.tool_timeout_s))),
            memory_mb=self.memory_mb,
            max_output_bytes=self.max_output_bytes,
        )

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "rules": list(self.rules),
            "graph_view": self.graph_view,
            "max_steps": self.max_steps,
            "tool_timeout_s": self.tool_timeout_s,
            "memory_mb": self.memory_mb,
            "max_output_bytes": self.max_output_bytes,
            "temperature": self.temperature,
            "allow_tools": self.allow_tools,
            "tool_backend": self.tool_backend,
            "tool_preamble": self.tool_preamble,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Harness":
        allowed = {
            "name", "system_prompt", "rules", "graph_view", "max_steps",
            "tool_timeout_s", "memory_mb", "max_output_bytes", "temperature",
            "allow_tools", "tool_backend", "tool_preamble", "notes",
        }
        kwargs = {k: v for k, v in data.items() if k in allowed}
        kwargs.setdefault("name", "harness")
        kwargs.setdefault("system_prompt", "")
        kwargs["rules"] = [str(r) for r in kwargs.get("rules") or []]
        return cls(**kwargs)

    def copy(self, **changes: Any) -> "Harness":
        data = self.to_dict()
        data.pop("version", None)
        data.update(changes)
        return Harness.from_dict(data)

    def without_rule(self, index: int) -> "Harness":
        if not 0 <= index < len(self.rules):
            raise IndexError(index)
        rules = [r for i, r in enumerate(self.rules) if i != index]
        return self.copy(rules=rules, name=f"{self.name}-no-rule-{index}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Harness":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


BASELINE_A = Harness(
    name="baseline_a",
    system_prompt=(
        "You are a graph reasoning assistant. Solve the problem accurately. "
        "You may use Python with NetworkX in the sandbox. Use the tools when "
        "they help, then give the final answer in the required format."
    ),
    rules=[],
    graph_view="summary",
    notes="Minimal zero-shot harness: protocol plus raw tools, no strategy.",
)

_BASELINE_B_PROMPT = (
    "You are an expert graph algorithms engineer. You solve graph problems by "
    "writing and running code, never by guessing from a node's numeric label."
)

_BASELINE_B_RULES = [
    "Use the NetworkX `G` object directly; never infer edges or distances from node ids.",
    "Prefer exact library routines: nx.shortest_path, nx.connected_components, "
    "nx.find_cycle, nx.is_connected and nx.diameter.",
    "For maximum clique, run exact search only when the graph is small; on large "
    "graphs use a heuristic and verify that every returned pair is adjacent.",
    "Check feasibility before answering: path endpoints and edges, one representative "
    "per connected component, cycle closure, clique pairwise adjacency, or diameter distance.",
    "Keep sandbox output short: print only the result, its length/size, and a small check.",
    "If code times out or errors, change the algorithm instead of rerunning it unchanged.",
    "Put only the final node list inside <answer></answer>; use [] for a negative answer.",
]

BASELINE_B = Harness(
    name="baseline_b",
    system_prompt=_BASELINE_B_PROMPT,
    rules=_BASELINE_B_RULES,
    graph_view="summary",
    notes="Human expert harness with explicit graph-algorithm best practices.",
)

# Baseline B v2 was written after inspecting the first Qwen3-4B pilot failures.
# The original baseline_b is kept unchanged so those pilot results stay
# reproducible.  The additions are general task recipes, not dataset facts.
_BASELINE_B_V2_RULES = _BASELINE_B_RULES + [
    "For connected components, print `list(nx.connected_components(G))` and "
    "return one node from each component.",
    "For diameter, find a pair of nodes whose shortest-path distance equals the "
    "diameter (for example with nx.all_pairs_shortest_path_length), then return "
    "nx.shortest_path between that pair. Do not index the node list by distance.",
    "Always call print(...) on the value you need. The sandbox also echoes a "
    "bare final expression, but explicit output is safer.",
    "For maximum clique on small graphs, print `nx.max_weight_clique(G, weight=None)` "
    "and return the clique; on large graphs use a verified heuristic.",
    "Once a successful run prints a feasible result, stop and give that result in "
    "<answer> tags; do not rerun identical code.",
    "Never filter, sort, or index nodes by numeric value; node ids are arbitrary "
    "and may be larger than the graph size.",
    "Materialize NetworkX generators with list(...) or dict(...) before iterating "
    "them more than once; exhausted generators silently produce empty results.",
]

BASELINE_B_V2 = Harness(
    name="baseline_b_v2",
    system_prompt=_BASELINE_B_PROMPT,
    rules=_BASELINE_B_V2_RULES,
    graph_view="summary",
    notes="Baseline B plus explicit connected-components, diameter-witness and print rules.",
)

HARNESSES = {h.name: h for h in (BASELINE_A, BASELINE_B, BASELINE_B_V2)}


def get_harness(name_or_path: str) -> Harness:
    """Look up a built-in harness or load a saved checkpoint from JSON."""
    path = Path(name_or_path)
    if path.exists():
        return Harness.load(path)
    try:
        return HARNESSES[name_or_path].copy()
    except KeyError:
        raise KeyError(
            f"unknown harness {name_or_path!r}; built-ins: {sorted(HARNESSES)} "
            "or pass a path to a harness JSON file"
        ) from None


def _summary_from_graph_json(g: dict) -> str:
    G = nx.DiGraph() if g.get("directed") else nx.Graph()
    G.add_nodes_from(g["nodes"])
    G.add_edges_from(tuple(e) for e in g["edges"])
    for u, v, attrs in g.get("edge_attrs", []):
        G.edges[u, v].update(attrs)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return "The graph is empty."
    degrees = [d for _, d in G.degree()]
    density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    weighted = " weighted" if any("weight" in d for _, _, d in G.edges(data=True)) else ""
    directed = "directed" if G.is_directed() else "undirected"
    return (
        f"{directed.capitalize()}{weighted} graph with {n} nodes and {m} edges "
        f"(density {density:.4g}).\n"
        f"Degree: min {min(degrees)}, median {statistics.median(degrees):g}, "
        f"mean {statistics.fmean(degrees):.2f}, max {max(degrees)}; "
        f"isolated nodes: {sum(d == 0 for d in degrees)}.\n"
        "Node ids are arbitrary labels."
    )


def answer_instruction(record: dict) -> str:
    try:
        kind = get_task(record["task"]).answer_kind
    except KeyError:
        kind = "nodes"
    examples = {
        "nodes": "<answer>[a, b, c]</answer>",
        "number": "<answer>42</answer>",
        "boolean": "<answer>yes</answer> or <answer>no</answer>",
        "string": "<answer>label</answer>",
        "mapping": "<answer>[0, 1, 2]</answer>",
        "two_lists": "<answer>[0, 1, 2], [2, 3, 4]</answer>",
    }
    return ("Answer format for this task: " + examples.get(kind, examples["nodes"]))


def graph_summary(record: dict) -> str:
    """Structural statistics that do not reveal any task answer."""
    if record.get("graphs"):
        labels = record.get("graph_labels") or ["A", "B"]
        return "\n\n".join(
            f"Graph {label}:\n{_summary_from_graph_json(g)}"
            for label, g in zip(labels, record["graphs"])
        )
    return _summary_from_graph_json(record["graph"])


def render_graph(record: dict, view: str = "summary") -> str:
    if view == "summary":
        return graph_summary(record)
    if record.get("graphs"):
        labels = record.get("graph_labels") or ["A", "B"]
        parts = []
        for label, g in zip(labels, record["graphs"]):
            parts.append(f"Graph {label}:\n" + STYLES[view]({"graph": g}))
        return "\n\n".join(parts)
    try:
        return STYLES[view](record)
    except KeyError:
        raise ValueError(f"unknown graph view {view!r}") from None


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------

@dataclass
class Action:
    kind: str  # "code", "answer", "none"
    code: str | None = None
    answer: str | None = None
    note: str = ""


def _balanced_json_objects(text: str) -> list[tuple[int, int, dict]]:
    """Find JSON objects outside fenced code blocks, left to right."""
    fenced = [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]

    def inside_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fenced)

    found = []
    i = 0
    while i < len(text):
        if text[i] != "{" or inside_fence(i):
            i += 1
            continue
        depth, quote, escape = 0, False, False
        for j in range(i, len(text)):
            ch = text[j]
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    quote = False
            else:
                if ch == '"':
                    quote = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[i:j + 1]
                        try:
                            value = loads_relaxed(chunk)
                        except json.JSONDecodeError:
                            pass
                        else:
                            if isinstance(value, dict) and "tool" in value:
                                found.append((i, j + 1, value))
                        i = j
                        break
        i += 1
    return found


def _format_answer(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            return f"<answer>{text}</answer>"
        return f"<answer>[{text}]</answer>"
    if isinstance(value, bool):
        return "<answer>[]</answer>" if not value else "<answer>[true]</answer>"
    try:
        inner = ", ".join(str(v) for v in value)
    except TypeError:
        inner = str(value)
    return f"<answer>[{inner}]</answer>"


def parse_action(response: str) -> Action:
    """Pick the last meaningful action in a model response."""
    candidates: list[tuple[int, Action]] = []

    for match in _ANSWER_RE.finditer(response):
        candidates.append((match.start(), Action("answer", answer=match.group(0).strip())))

    for match in _FENCE_RE.finditer(response):
        lang = (match.group("lang") or "").lower()
        code = match.group("code")
        if lang in {"python", "py", "python3", "c", "cpp", "c11", ""}:
            candidates.append((match.start(), Action("code", code=code)))
        elif lang == "json":
            try:
                value = loads_relaxed(code)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "tool" in value:
                candidates.append((match.start(), _action_from_json(value)))

    for start, _, value in _balanced_json_objects(response):
        candidates.append((start, _action_from_json(value)))

    if candidates:
        # Code wins over an answer in the same message: the model may be
        # guessing the output before running it. Execute the code, return the
        # observation, and let the model answer on the next turn.
        code_candidates = [(pos, action) for pos, action in candidates
                           if action.kind == "code"]
        if code_candidates:
            code_candidates.sort(key=lambda item: item[0])
            return code_candidates[-1][1]
        candidates.sort(key=lambda item: item[0])
        action = candidates[-1][1]
        if action.kind != "none":
            return action

    # A model that ignores the protocol but still ends with a bracketed list is
    # usually giving a usable answer; let the scorer decide.
    matches = list(_BRACKET_RE.finditer(response))
    if matches:
        last = matches[-1]
        return Action("answer", answer=f"<answer>[{last.group(1)}]</answer>")
    return Action("none")


def _find_any_answer(text: str) -> str | None:
    """Find an answer in a final turn even if the model also included code."""
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(0).strip()
    for _, _, value in _balanced_json_objects(text):
        action = _action_from_json(value)
        if action.kind == "answer" and action.answer:
            return action.answer
    matches = list(_BRACKET_RE.finditer(text))
    if matches:
        return f"<answer>[{matches[-1].group(1)}]</answer>"
    return None


def _action_from_json(value: dict) -> Action:
    tool = str(value.get("tool", "")).strip().lower()
    if tool in {"run_python", "python", "execute", "code", "run_c", "c", "compile"}:
        code = value.get("code")
        if not isinstance(code, str):
            return Action("none", note="run_python call has no string `code`")
        return Action("code", code=code)
    if tool in {"answer", "final", "submit"}:
        if "answer" not in value:
            return Action("none", note="answer call has no `answer` field")
        return Action("answer", answer=_format_answer(value["answer"]))
    return Action("none", note=f"unknown tool {tool!r}")


# --------------------------------------------------------------------------
# agent loop
# --------------------------------------------------------------------------

@dataclass
class AgentResult:
    id: str
    task: str
    response: str
    stopped_reason: str
    steps: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    tool_calls: int = 0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self, include_messages: bool = True) -> dict:
        out = {
            "id": self.id,
            "task": self.task,
            "response": self.response,
            "stopped_reason": self.stopped_reason,
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
            "duration_s": self.duration_s,
            "error": self.error,
        }
        if include_messages:
            out["messages"] = self.messages
        return out


def run_agent(
    harness: Harness,
    model: Model,
    record: dict,
    max_steps: int | None = None,
    save_messages: bool = True,
) -> AgentResult:
    """Run one problem to completion (or to the step limit)."""
    started = time.perf_counter()
    step_limit = max_steps or harness.max_steps
    messages = [
        {"role": "system", "content": harness.system_message()},
        {"role": "user", "content": harness.user_message(record)},
    ]
    steps: list[dict] = []
    prompt_tokens = completion_tokens = tool_calls = 0
    final_response = ""
    stopped_reason = "max_steps"
    sandbox: PythonSandbox | None = None
    task = record.get("task", "")

    try:
        for step_index in range(1, step_limit + 1):
            result: ChatResult = model.chat(messages, temperature=harness.temperature)
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            text = result.text or ""
            messages.append({"role": "assistant", "content": text})
            action = parse_action(text)
            steps.append({
                "step": step_index,
                "response": text,
                "action": action.kind,
                "note": action.note,
            })

            if action.kind == "answer" and action.answer:
                final_response = action.answer
                stopped_reason = "answer"
                break

            if action.kind == "code" and action.code is not None:
                if not harness.allow_tools:
                    messages.append({"role": "user", "content":
                        "Tools are disabled for this task. Answer from the graph text."})
                    continue
                tool_calls += 1
                if sandbox is None:
                    if harness.tool_backend == "c":
                        sandbox = CSandbox(record, limits=harness.sandbox_limits())
                    else:
                        sandbox = PythonSandbox(record, limits=harness.sandbox_limits(),
                                                preamble=harness.tool_preamble)
                run = sandbox.run(action.code)
                observation = run.observation()
                steps[-1].update({
                    "code": action.code,
                    "ok": run.ok,
                    "timed_out": run.timed_out,
                    "duration_s": run.duration_s,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                })
                messages.append({"role": "user", "content": observation})
                continue

            # No usable action: nudge the model once per step.
            if harness.allow_tools:
                nudge = (
                    "I could not find a tool call or a final answer. Reply with a "
                    '{"tool": "run_python", ...} object, a ```python block, or '
                    "<answer>[...]</answer>."
                )
            else:
                nudge = ("I could not find a final answer. Tools are disabled; "
                         "reply with <answer>...</answer> in the requested format.")
            messages.append({"role": "user", "content": nudge})

        if stopped_reason == "max_steps":
            # One extra turn with no tool use: summarize the best result instead
            # of returning an empty prediction. This is a final-answer budget,
            # separate from the tool-step budget.
            final_nudge = (
                "You have reached the step limit. Do not run more code. Based on "
                "the observations above, give your final answer now as "
                "<answer>[...]</answer> (or [] for a negative answer)."
            )
            messages.append({"role": "user", "content": final_nudge})
            try:
                result = model.chat(messages, temperature=harness.temperature)
            except Exception as exc:  # keep the trajectory; runner records the miss
                steps.append({"step": len(steps) + 1, "response": "",
                              "action": "none", "note": f"final turn failed: {exc}"})
            else:
                prompt_tokens += result.prompt_tokens
                completion_tokens += result.completion_tokens
                text = result.text or ""
                messages.append({"role": "assistant", "content": text})
                action = parse_action(text)
                steps.append({
                    "step": len(steps) + 1,
                    "response": text,
                    "action": action.kind,
                    "note": "final answer turn",
                })
                answer = _find_any_answer(text)
                if answer:
                    final_response = answer
                    stopped_reason = "final_answer"
    finally:
        if sandbox is not None:
            sandbox.close()

    duration = time.perf_counter() - started
    return AgentResult(
        id=record["id"],
        task=task,
        response=final_response,
        stopped_reason=stopped_reason,
        steps=steps,
        messages=messages if save_messages else [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_s=duration,
        tool_calls=tool_calls,
    )


def expected_task_description(record: dict) -> str:
    """Short task label for logs (does not include the answer)."""
    try:
        kind = get_task(record["task"]).kind
    except KeyError:
        kind = "?"
    return f"{record['task']} ({kind})"
