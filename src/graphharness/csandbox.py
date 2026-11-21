"""Compile and run model-written C against a tiny graph I/O library.

This is the C tool backend for the language/library ablation. The model gets a
``graph.txt`` file, ``graph_io.h`` and ``gcc``; it must implement graph
algorithms itself. It does not get NetworkX or Python.

The harness/prompt/orchestration remains Python; only the model-facing tool is
C. Rewriting the whole agent loop in C would add no scientific signal and would
remove the model API/JSON layer this project depends on.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .sandbox import SandboxLimits, SandboxResult

GRAPH_IO_H = r"""
#ifndef GRAPH_IO_H
#define GRAPH_IO_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n;          /* number of nodes */
    int m;          /* number of edges */
    int max_id;     /* largest node id */
    int *eu;
    int *ev;
    int *ew;
} Graph;

Graph load_graph(void);
const char *get_param(const char *key);
void free_graph(Graph *g);
#endif
"""

GRAPH_IO_C = r"""
#include "graph_io.h"

Graph load_graph(void) {
    Graph g = {0, 0, -1, NULL, NULL, NULL};
    FILE *f = fopen("graph.txt", "r");
    if (!f) { perror("graph.txt"); exit(2); }
    int n = 0, m = 0;
    if (fscanf(f, "%d %d", &n, &m) != 2) { exit(2); }
    g.m = m;
    g.n = n;
    g.eu = malloc(sizeof(int) * (m > 0 ? m : 1));
    g.ev = malloc(sizeof(int) * (m > 0 ? m : 1));
    g.ew = malloc(sizeof(int) * (m > 0 ? m : 1));
    for (int i = 0; i < m; i++) {
        if (fscanf(f, "%d %d %d", &g.eu[i], &g.ev[i], &g.ew[i]) != 3) { exit(2); }
        if (g.eu[i] > g.max_id) g.max_id = g.eu[i];
        if (g.ev[i] > g.max_id) g.max_id = g.ev[i];
    }
    fclose(f);
    return g;
}

const char *get_param(const char *key) {
    static char value[512];
    FILE *f = fopen("params.txt", "r");
    if (!f) return "";
    char k[128], v[512];
    while (fscanf(f, "%127s %511s", k, v) == 2) {
        if (strcmp(k, key) == 0) { strcpy(value, v); fclose(f); return value; }
    }
    fclose(f);
    return "";
}

void free_graph(Graph *g) {
    free(g->eu); free(g->ev); free(g->ew);
    g->eu = g->ev = g->ew = NULL;
}
"""

LIMIT_RUNNER = r"""
import os, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu} + 1))
resource.setrlimit(resource.RLIMIT_AS, ({mem},) * 2)
resource.setrlimit(resource.RLIMIT_FSIZE, ({fsize},) * 2)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.execv("./solution", ["./solution"])
"""


class CSandbox:
    def __init__(self, record: dict, limits: SandboxLimits | None = None,
                 workdir: str | Path | None = None):
        self.record = record
        self.limits = limits or SandboxLimits()
        self._owns_dir = workdir is None
        if workdir:
            self.workdir = Path(workdir)
        else:
            base = os.environ.get("GRAPH_HARNESS_SANDBOX_DIR")
            if base:
                Path(base).mkdir(parents=True, exist_ok=True)
            self.workdir = Path(tempfile.mkdtemp(prefix="gh-c-sandbox-", dir=base))
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "graph_io.h").write_text(GRAPH_IO_H)
        (self.workdir / "graph_io.c").write_text(GRAPH_IO_C)
        (self.workdir / "_run_limits.py").write_text(LIMIT_RUNNER.format(
            cpu=self.limits.cpu_seconds,
            mem=self.limits.memory_mb * 1024 * 1024,
            fsize=self.limits.max_output_bytes * 4,
        ))
        self._write_task()
        self._closed = False

    def _write_task(self):
        g = self.record["graph"]
        nodes = g["nodes"]
        if any(not isinstance(v, int) for v in nodes):
            raise ValueError("C backend currently requires integer node ids")
        edges = g["edges"]
        weights = {}
        for u, v, d in g.get("edge_attrs", []):
            if "weight" in d:
                weights[(u, v)] = int(d["weight"])
                weights[(v, u)] = int(d["weight"])
        lines = [f"{len(nodes)} {len(edges)}"]
        for u, v in edges:
            lines.append(f"{u} {v} {weights.get((u, v), 1)}")
        (self.workdir / "graph.txt").write_text("\n".join(lines) + "\n")
        params = self.record.get("params") or {}
        lines = []
        for k, v in params.items():
            if isinstance(v, (int, str)):
                lines.append(f"{k} {v}")
        (self.workdir / "params.txt").write_text("\n".join(lines) + "\n")

    def run(self, code: str) -> SandboxResult:
        if self._closed:
            raise RuntimeError("sandbox is closed")
        (self.workdir / "solution.c").write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "HOME": str(self.workdir), "TMPDIR": str(self.workdir),
               "LANG": "C.UTF-8"}
        started = time.perf_counter()
        try:
            compile_proc = subprocess.run(
                ["gcc", "-O2", "-std=c11", "-o", "solution", "solution.c", "graph_io.c", "-lm"],
                cwd=self.workdir, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "", "gcc compile timed out", None,
                                 time.perf_counter() - started, timed_out=True)
        if compile_proc.returncode != 0:
            return SandboxResult(False, compile_proc.stdout, compile_proc.stderr,
                                 compile_proc.returncode, time.perf_counter() - started)
        out_path = self.workdir / "_stdout.txt"
        err_path = self.workdir / "_stderr.txt"
        timed_out = False
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(
                [sys.executable, str(self.workdir / "_run_limits.py")],
                cwd=self.workdir, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=self.limits.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=5)
                rc = proc.returncode
        stdout = out_path.read_text(errors="replace")[:self.limits.max_output_bytes]
        stderr = err_path.read_text(errors="replace")[:self.limits.max_output_bytes]
        return SandboxResult(rc == 0 and not timed_out, stdout, stderr, rc,
                             time.perf_counter() - started, timed_out=timed_out,
                             truncated=out_path.stat().st_size > self.limits.max_output_bytes)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._owns_dir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
