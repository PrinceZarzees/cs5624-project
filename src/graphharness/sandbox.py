"""Run model-written Python for one graph problem in a resource-limited process.

The sandbox is deliberately small: each call starts a fresh Python process in a
temporary working directory that contains

* ``graph.json`` with ``{"graph": ..., "params": ...}`` (never the answer), and
* a preamble that defines ``G`` (a networkx graph), ``params`` and ``nodes``.

Calls do not share Python state, but files written in the working directory do
persist between calls, which is enough for most multi-step solutions.

Isolation is process-level: a separate working directory, a clean environment,
CPU/address-space/output limits and a hard wall-clock timeout.  This is meant to
protect an experiment from runaway model code, not to run untrusted code from an
adversary.  Do not point it at code you would not run yourself.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits for one code execution."""

    timeout_s: float = 30.0
    cpu_seconds: int = 30
    memory_mb: int = 2048
    max_output_bytes: int = 16_000

    def __post_init__(self):
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.cpu_seconds <= 0:
            raise ValueError("cpu_seconds must be positive")
        if self.memory_mb < 128:
            raise ValueError("memory_mb must be at least 128")
        if self.max_output_bytes < 256:
            raise ValueError("max_output_bytes must be at least 256")


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int | None
    duration_s: float
    timed_out: bool = False
    truncated: bool = False

    def observation(self, max_chars: int = 12_000) -> str:
        """A compact message to send back to the model."""
        if self.timed_out:
            text = (
                f"The code timed out after {self.duration_s:.1f}s and was killed. "
                "Use a simpler or more efficient algorithm; do not rerun the same code."
            )
        elif self.ok:
            text = f"The code finished successfully ({self.duration_s:.2f}s)."
        else:
            text = (
                f"The code failed with exit code {self.returncode} "
                f"({self.duration_s:.2f}s)."
            )
        if self.stdout:
            text += f"\n\nstdout:\n{self.stdout}"
        else:
            text += "\n\nstdout: (empty)"
        if self.stderr:
            text += f"\n\nstderr:\n{self.stderr}"
        if self.truncated:
            text += "\n\n(output was truncated)"
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n(output truncated by the harness)"
        return text


_PREAMBLE = """\
# --- injected by graphharness.sandbox ---
import ast as _gh_ast
import json as _gh_json
import networkx as _gh_nx

with open("graph.json", "r", encoding="utf-8") as _gh_f:
    _gh_data = _gh_json.load(_gh_f)

def _gh_load_graph(_g):
    _H = _gh_nx.DiGraph() if _g.get("directed") else _gh_nx.Graph()
    _H.add_nodes_from(_g["nodes"])
    _H.add_edges_from(tuple(e) for e in _g["edges"])
    for _node, _attrs in _g.get("node_attrs", []):
        _H.nodes[_node].update(_attrs)
    for _u, _v, _attrs in _g.get("edge_attrs", []):
        _H.edges[_u, _v].update(_attrs)
    return _H

if _gh_data.get("graphs"):
    graphs = [_gh_load_graph(_g) for _g in _gh_data["graphs"]]
    G1, G2 = graphs[0], graphs[1]
    G = G1
else:
    G = _gh_load_graph(_gh_data["graph"])
    graphs = [G]
params = _gh_data["params"]
nodes = list(G.nodes)
del _gh_data, _gh_f, _gh_load_graph

# Execute user code with notebook-like behaviour: if the last statement is an
# expression, print its repr. Models often write `path` instead of `print(path)`;
# without this the observation is empty and the model tends to guess.
with open("user_code.py", "r", encoding="utf-8") as _gh_f:
    _gh_source = _gh_f.read()
_gh_tree = _gh_ast.parse(_gh_source, filename="user_code.py")
_gh_last = None
if _gh_tree.body and isinstance(_gh_tree.body[-1], _gh_ast.Expr):
    _gh_last = _gh_ast.Expression(_gh_tree.body[-1].value)
    _gh_tree.body = _gh_tree.body[:-1]
exec(compile(_gh_tree, "user_code.py", "exec"), globals())
if _gh_last is not None:
    _gh_value = eval(compile(_gh_last, "user_code.py", "eval"), globals())
    if _gh_value is not None:
        print(repr(_gh_value))
# --- model code has finished ---
"""


def _limited_starter(limits: SandboxLimits) -> str:
    """A tiny wrapper that sets rlimits and then execs the step script.

    Doing this in a wrapper avoids ``preexec_fn`` in the parent, which is not
    safe when the experiment runner uses worker threads.
    """
    return f"""\
import os
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, ({limits.cpu_seconds}, {limits.cpu_seconds + 1}))
resource.setrlimit(resource.RLIMIT_AS, ({limits.memory_mb * 1024 * 1024},) * 2)
resource.setrlimit(resource.RLIMIT_FSIZE, ({limits.max_output_bytes * 4},) * 2)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

os.execv(sys.executable, [sys.executable, "-I", "-u", "step.py"])
"""


_PREAMBLE_NOLIB = """\
# --- injected by graphharness.sandbox (no NetworkX) ---
import ast as _gh_ast
import json as _gh_json
import sys as _gh_sys

class _GhBlockNX:
    def find_spec(self, name, path=None, target=None):
        if name == "networkx" or name.startswith("networkx."):
            raise ImportError("NetworkX is not available in this sandbox")
        return None

if not any(getattr(f, "__name__", "") == "_GhBlockNX" for f in _gh_sys.meta_path):
    _gh_sys.meta_path.insert(0, _GhBlockNX())

with open("graph.json", "r", encoding="utf-8") as _gh_f:
    _gh_data = _gh_json.load(_gh_f)

def _gh_adj(_g):
    a = {v: [] for v in _g["nodes"]}
    for e in _g["edges"]:
        u, v = e[0], e[1]
        a.setdefault(u, []).append(v)
        a.setdefault(v, []).append(u)
    return a

if _gh_data.get("graphs"):
    adj1 = _gh_adj(_gh_data["graphs"][0]); adj2 = _gh_adj(_gh_data["graphs"][1])
    edge_list1 = [tuple(e[:2]) for e in _gh_data["graphs"][0]["edges"]]
    edge_list2 = [tuple(e[:2]) for e in _gh_data["graphs"][1]["edges"]]
    adj, edge_list = adj1, edge_list1
else:
    adj = _gh_adj(_gh_data["graph"])
    edge_list = [tuple(e[:2]) for e in _gh_data["graph"]["edges"]]
params = _gh_data["params"]
nodes = list(adj)
del _gh_data, _gh_f

with open("user_code.py", "r", encoding="utf-8") as _gh_f:
    _gh_source = _gh_f.read()
_gh_tree = _gh_ast.parse(_gh_source, filename="user_code.py")
_gh_last = None
if _gh_tree.body and isinstance(_gh_tree.body[-1], _gh_ast.Expr):
    _gh_last = _gh_ast.Expression(_gh_tree.body[-1].value)
    _gh_tree.body = _gh_tree.body[:-1]
exec(compile(_gh_tree, "user_code.py", "exec"), globals())
if _gh_last is not None:
    _gh_value = eval(compile(_gh_last, "user_code.py", "eval"), globals())
    if _gh_value is not None:
        print(repr(_gh_value))
# --- model code has finished ---
"""


class PythonSandbox:
    """A temporary working directory plus a method to execute code in it."""

    def __init__(self, record: dict, limits: SandboxLimits | None = None,
                 workdir: str | Path | None = None, preamble: str = "",
                 block_networkx: bool = False):
        self.record = record
        self.limits = limits or SandboxLimits()
        self.preamble = preamble or ""
        self.block_networkx = block_networkx
        self._owns_dir = workdir is None
        if workdir:
            self.workdir = Path(workdir)
        else:
            base = os.environ.get("GRAPH_HARNESS_SANDBOX_DIR")
            if base:
                Path(base).mkdir(parents=True, exist_ok=True)
            self.workdir = Path(tempfile.mkdtemp(prefix="gh-sandbox-", dir=base))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._write_task()
        self._starter = self.workdir / "_start.py"
        self._starter.write_text(_limited_starter(self.limits), encoding="utf-8")
        self._closed = False

    def _write_task(self) -> None:
        payload = {
            "graph": self.record["graph"],
            "params": self.record.get("params") or {},
        }
        if self.record.get("graphs"):
            payload["graphs"] = self.record["graphs"]
        # Make sure a record's ground truth can never leak into the sandbox.
        assert "answer" not in payload
        (self.workdir / "graph.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )

    def run(self, code: str) -> SandboxResult:
        if self._closed:
            raise RuntimeError("sandbox is closed")
        if not isinstance(code, str):
            code = str(code)
        step = self.workdir / "step.py"
        user_code = (self.preamble + "\n" + code) if self.preamble else code
        (self.workdir / "user_code.py").write_text(user_code + "\n", encoding="utf-8")
        step.write_text(_PREAMBLE_NOLIB if self.block_networkx else _PREAMBLE,
                        encoding="utf-8")

        out_path = self.workdir / "_stdout.txt"
        err_path = self.workdir / "_stderr.txt"
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.workdir),
            "TMPDIR": str(self.workdir),
            "PYTHONHASHSEED": "0",
            "LANG": "C.UTF-8",
        }
        started = time.perf_counter()
        timed_out = False
        returncode: int | None = None
        try:
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                proc = subprocess.Popen(
                    [sys.executable, str(self._starter)],
                    cwd=str(self.workdir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=self.limits.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    proc.wait(timeout=5)
                    returncode = proc.returncode
        except Exception as exc:  # infrastructure failure, not model code
            duration = time.perf_counter() - started
            return SandboxResult(False, "", f"sandbox error: {exc}", None, duration)

        duration = time.perf_counter() - started
        stdout, trunc_out = _read_limited(out_path, self.limits.max_output_bytes)
        stderr, trunc_err = _read_limited(err_path, self.limits.max_output_bytes)
        return SandboxResult(
            ok=(returncode == 0 and not timed_out),
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            duration_s=duration,
            timed_out=timed_out,
            truncated=trunc_out or trunc_err,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_dir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self) -> "PythonSandbox":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _read_limited(path: Path, limit: int) -> tuple[str, bool]:
    try:
        data = path.read_bytes()
    except OSError:
        return "", False
    truncated = len(data) > limit
    if truncated:
        data = data[:limit]
    return data.decode("utf-8", errors="replace"), truncated
