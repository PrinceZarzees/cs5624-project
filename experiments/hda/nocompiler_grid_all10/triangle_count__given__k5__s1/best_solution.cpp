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

// Optional helper: adjacency lists (for undirected graphs each edge is added
// in both directions). Pair = (neighbor, weight).
static vector<vector<pair<int,long long>>> adjacency(const Graph &g) {
    vector<vector<pair<int,long long>>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back({e.v, e.w});
        if (!g.directed) adj[e.v].push_back({e.u, e.w});
    }
    return adj;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    auto adj = adjacency(g);
    
    long long triangles = 0;
    for (int u = 0; u < g.n; u++) {
        int n_neighbors = adj[u].size();
        if (n_neighbors < 2) continue;
        
        // Use a set to store neighbor indices for fast lookup
        set<int> neighbors;
        for (auto &p : adj[u]) {
            neighbors.insert(p.first);
        }
        
        // Count pairs of neighbors that are adjacent
        for (int i = 0; i < n_neighbors; i++) {
            int v = adj[u][i].first;
            for (int j = i + 1; j < n_neighbors; j++) {
                int w = adj[u][j].first;
                // Check if v and w are adjacent
                if (neighbors.find(v) != neighbors.end() && neighbors.find(w) != neighbors.end()) {
                    // Check if v and w are connected
                    for (auto &p : adj[v]) {
                        if (p.first == w) {
                            triangles++;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Each triangle is counted 3 times (once per vertex), so divide by 3
    triangles /= 3;
    
    cout << triangles << "\n";
    return 0;
}
