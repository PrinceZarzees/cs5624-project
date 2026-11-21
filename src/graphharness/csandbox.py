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


GRAPH_ALGOS_H = r"""
#ifndef GRAPH_ALGOS_H
#define GRAPH_ALGOS_H
#include "graph_io.h"

int ga_components(const Graph *g, int *comp);
int ga_has_cycle(const Graph *g);
int ga_diameter(const Graph *g);
int ga_max_clique(const Graph *g, int *out, int cap);
int ga_max_independent_set(const Graph *g, int *out, int cap);
int ga_min_vertex_cover(const Graph *g, int *out, int cap);
#endif
"""

GRAPH_ALGOS_C = r"""
#include "graph_algos.h"
#include <stdlib.h>
#include <string.h>

static char *matrix(const Graph *g, int n, int complement) {
    char *A = calloc((size_t)n * n, 1);
    if (!A) return NULL;
    if (!complement) {
        for (int i = 0; i < g->m; i++) {
            int u = g->eu[i], v = g->ev[i];
            if (u >= 0 && v >= 0 && u < n && v < n) { A[u*n+v] = 1; A[v*n+u] = 1; }
        }
    } else {
        for (int u = 0; u < n; u++)
            for (int v = u + 1; v < n; v++)
                if (!A[u*n+v]) { A[u*n+v] = 1; A[v*n+u] = 1; }
        /* fill complement properly from edges */
        for (int u = 0; u < n; u++) for (int v = 0; v < n; v++) A[u*n+v] = 1;
        for (int u = 0; u < n; u++) A[u*n+u] = 0;
        for (int i = 0; i < g->m; i++) {
            int u = g->eu[i], v = g->ev[i];
            if (u >= 0 && v >= 0 && u < n && v < n) { A[u*n+v] = 0; A[v*n+u] = 0; }
        }
    }
    return A;
}

int ga_components(const Graph *g, int *comp) {
    int n = g->max_id + 1;
    char *A = matrix(g, n, 0);
    int *q = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) comp[i] = -1;
    int c = 0;
    for (int s = 0; s < n; s++) {
        if (comp[s] != -1) continue;
        int head = 0, tail = 0;
        q[tail++] = s; comp[s] = c;
        while (head < tail) {
            int u = q[head++];
            for (int v = 0; v < n; v++) if (A[u*n+v] && comp[v] == -1) { comp[v] = c; q[tail++] = v; }
        }
        c++;
    }
    free(q); free(A);
    return c;
}

static int uf_find(int *p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }

int ga_has_cycle(const Graph *g) {
    int n = g->max_id + 1;
    int *p = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) p[i] = i;
    for (int i = 0; i < g->m; i++) {
        int u = g->eu[i], v = g->ev[i];
        if (u == v) { free(p); return 1; }
        int a = uf_find(p, u), b = uf_find(p, v);
        if (a == b) { free(p); return 1; }
        p[a] = b;
    }
    free(p);
    return 0;
}

int ga_diameter(const Graph *g) {
    int n = g->max_id + 1;
    char *A = matrix(g, n, 0);
    int *dist = malloc(sizeof(int) * n), *q = malloc(sizeof(int) * n);
    int best = 0;
    for (int s = 0; s < n; s++) {
        for (int i = 0; i < n; i++) dist[i] = -1;
        int head = 0, tail = 0;
        q[tail++] = s; dist[s] = 0;
        while (head < tail) {
            int u = q[head++];
            for (int v = 0; v < n; v++) if (A[u*n+v] && dist[v] == -1) {
                dist[v] = dist[u] + 1; q[tail++] = v;
            }
        }
        for (int i = 0; i < n; i++) {
            if (dist[i] < 0) { free(dist); free(q); free(A); return -1; }
            if (dist[i] > best) best = dist[i];
        }
    }
    free(dist); free(q); free(A);
    return best;
}

static void bk(const char *A, int n, int *R, int r, int *P, int p, int *X, int x,
               int *best, int *bestn) {
    if (p == 0 && x == 0) {
        if (r > *bestn) { *bestn = r; memcpy(best, R, sizeof(int) * r); }
        return;
    }
    int pivot = -1, bestdeg = -1;
    for (int i = 0; i < p + x; i++) {
        int u = (i < p) ? P[i] : X[i - p];
        int deg = 0;
        for (int j = 0; j < p; j++) if (A[u*n+P[j]]) deg++;
        if (deg > bestdeg) { bestdeg = deg; pivot = u; }
    }
    int *cand = malloc(sizeof(int) * (p > 0 ? p : 1));
    int nc = 0;
    for (int i = 0; i < p; i++) if (pivot < 0 || !A[pivot*n+P[i]]) cand[nc++] = P[i];
    for (int i = 0; i < nc; i++) {
        int v = cand[i];
        R[r] = v;
        int *np = malloc(sizeof(int) * (p > 0 ? p : 1));
        int *nx = malloc(sizeof(int) * (p > 0 ? p : 1));
        int npi = 0, nxi = 0;
        for (int j = 0; j < p; j++) if (A[v*n+P[j]]) np[npi++] = P[j];
        for (int j = 0; j < x; j++) if (A[v*n+X[j]]) nx[nxi++] = X[j];
        bk(A, n, R, r + 1, np, npi, nx, nxi, best, bestn);
        free(np); free(nx);
        int pos = -1;
        for (int j = 0; j < p; j++) if (P[j] == v) { pos = j; break; }
        if (pos >= 0) { for (int j = pos; j < p - 1; j++) P[j] = P[j + 1]; p--; }
        X[x++] = v;
    }
    free(cand);
}

static int clique_of(const char *A, int n, int *out, int cap) {
    int *R = malloc(sizeof(int) * n), *P = malloc(sizeof(int) * n), *X = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) P[i] = i;
    int bestn = 0, *best = malloc(sizeof(int) * n);
    bk(A, n, R, 0, P, n, X, 0, best, &bestn);
    int k = bestn < cap ? bestn : cap;
    memcpy(out, best, sizeof(int) * k);
    free(R); free(P); free(X); free(best);
    return bestn;
}

int ga_max_clique(const Graph *g, int *out, int cap) {
    int n = g->max_id + 1;
    char *A = matrix(g, n, 0);
    int k = clique_of(A, n, out, cap);
    free(A);
    return k;
}

int ga_max_independent_set(const Graph *g, int *out, int cap) {
    int n = g->max_id + 1;
    char *A = matrix(g, n, 1);
    int k = clique_of(A, n, out, cap);
    free(A);
    return k;
}

int ga_min_vertex_cover(const Graph *g, int *out, int cap) {
    int n = g->max_id + 1;
    char *A = matrix(g, n, 1);
    int *is = malloc(sizeof(int) * n);
    int k = clique_of(A, n, is, n);
    char *in_is = calloc(n, 1);
    for (int i = 0; i < k; i++) in_is[is[i]] = 1;
    int c = 0;
    for (int v = 0; v < n && c < cap; v++) if (!in_is[v]) out[c++] = v;
    free(in_is); free(is); free(A);
    return n - k;
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
                 workdir: str | Path | None = None, library: bool = False):
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
        self.library = library
        if library:
            (self.workdir / "graph_algos.h").write_text(GRAPH_ALGOS_H)
            (self.workdir / "graph_algos.c").write_text(GRAPH_ALGOS_C)
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
            sources = ["solution.c", "graph_io.c"] + (["graph_algos.c"] if self.library else [])
            compile_proc = subprocess.run(
                ["gcc", "-O2", "-std=c11", "-o", "solution", *sources, "-lm"],
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
