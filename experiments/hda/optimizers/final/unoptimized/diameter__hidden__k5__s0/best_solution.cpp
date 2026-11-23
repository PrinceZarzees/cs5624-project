// ---------------------------------------------------------------------------
// HDA graph-loading template (given verbatim to the Harness Developing Agent).
// Input format (stdin):
//   line 1 : n m directed weighted      (directed, weighted are 0/1)
//   next m : u v        (weighted == 0)   or   u v w   (weighted == 1)
//            nodes are 0 .. n-1, no self-loops, no duplicate edges
//            (for directed graphs u->v and v->u may both appear)
//   last   : k q1 q2 ... qk               (k = NUMBER of query arguments; not a type code)
// Output (stdout): exactly one line holding the answer.
// Rules: C++17, standard library only, no file I/O, no threads, single file.
// ---------------------------------------------------------------------------
#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; long long w; };

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;          // edge list exactly as read
    int query_k = 0;             // number of query arguments (k on the last line)
    vector<long long> query;     // query arguments q1..qk (may be empty)
};

static Graph read_graph() {
    Graph g;
    int d = 0, w = 0;
    if (!(cin >> g.n >> g.m >> d >> w)) return g;
    g.directed = d; g.weighted = w;
    g.edges.resize(g.m);
    for (auto &e : g.edges) {
        cin >> e.u >> e.v;
        e.w = 1;
        if (g.weighted) cin >> e.w;
    }
    int k = 0;
    if (cin >> k) { g.query_k = k; g.query.resize(k); for (auto &q : g.query) cin >> q; }
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();

    // Build undirected adjacency list
    vector<vector<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back(e.v);
        if (!g.directed) {
            adj[e.v].push_back(e.u);
        }
    }

    // If k != 2, return 0 (no query)
    if (g.query_k != 2) {
        cout << 0 << "\n";
        return 0;
    }

    int s = g.query[0], t = g.query[1];
    // Find connected component of s using BFS
    vector<bool> visited(g.n, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    vector<int> component;
    component.push_back(s);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
                component.push_back(v);
            }
        }
    }

    // Count edges in component that are not incident to s
    int edge_count = 0;
    for (int u : component) {
        for (int v : adj[u]) {
            if (u != s && v != s) {
                edge_count++;
            }
        }
    }

    // Each edge is counted twice in this loop (u->v and v->u), so divide by 2
    edge_count /= 2;

    cout << edge_count << "\n";
    return 0;
}
