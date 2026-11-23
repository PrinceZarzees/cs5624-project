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
    
    // Build adjacency list: for each node, store its neighbors (as unordered_set)
    vector<unordered_set<int>> adj(g.n);
    for (const auto& e : g.edges) {
        adj[e.u].insert(e.v);
        if (!g.directed) {
            adj[e.v].insert(e.u);
        }
    }
    
    long long triangles = 0;
    // For each edge (u, v), check for each neighbor w of v if w has an edge to u
    for (int u = 0; u < g.n; u++) {
        for (int v : adj[u]) {
            for (int w : adj[v]) {
                if (w != u && adj[w].find(u) != adj[w].end()) {
                    triangles++;
                }
            }
        }
    }
    // Each triangle is counted 3 times (once for each edge), so divide by 3
    triangles /= 3;
    
    cout << triangles << "\n";
    return 0;
}
