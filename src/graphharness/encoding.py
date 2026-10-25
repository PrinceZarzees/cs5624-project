"""Simple ways to write a graph as text for a prompt.

These are starting points for the baseline harnesses. Choosing or improving
the encoding is part of what the harness (and the optimizer) controls.

    from graphharness.encoding import build_prompt
    prompt = build_prompt(record, style="edge_list")
"""

from __future__ import annotations

from collections import defaultdict


def edge_list(record: dict) -> str:
    g = record["graph"]
    lines = [
        f"The graph is undirected with {len(g['nodes'])} nodes and {len(g['edges'])} edges.",
        "Nodes: " + ", ".join(str(v) for v in g["nodes"]),
        "Edges (one per line, 'u - v'):",
    ]
    lines += [f"{u} - {v}" for u, v in g["edges"]]
    return "\n".join(lines)


def adjacency_list(record: dict) -> str:
    g = record["graph"]
    adj = defaultdict(list)
    for u, v in g["edges"]:
        adj[u].append(v)
        adj[v].append(u)
    lines = [
        f"The graph is undirected with {len(g['nodes'])} nodes and {len(g['edges'])} edges.",
        "Each line lists a node and its neighbors:",
    ]
    for v in g["nodes"]:
        nbrs = ", ".join(str(x) for x in sorted(adj[v], key=str))
        lines.append(f"{v}: {nbrs}" if nbrs else f"{v}: (no neighbors)")
    return "\n".join(lines)


STYLES = {"edge_list": edge_list, "adjacency_list": adjacency_list}


def build_prompt(record: dict, style: str = "edge_list") -> str:
    """Graph text + the task question. Asks for the answer in <answer> tags."""
    graph_text = STYLES[style](record)
    return (
        f"{graph_text}\n\n"
        f"Question: {record['question']}\n"
        "Write your final answer as a bracketed list inside <answer></answer> tags, "
        "for example: <answer>[1, 2, 3]</answer>."
    )
