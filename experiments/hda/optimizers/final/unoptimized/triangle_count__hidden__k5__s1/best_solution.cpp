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

static vector<set<int>> adjacency_set(const Graph &g) {
    vector<set<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].insert(e.v);
        if (!g.directed) adj[e.v].insert(e.u);
    }
    return adj;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    auto adj = adjacency_set(g);
    
    long long triangles = 0;
    for (int u = 0; u < g.n; u++) {
        for (auto it1 = adj[u].begin(); it1 != adj[u].end(); ++it1) {
            int v = *it1;
            for (auto it2 = next(it1); it2 != adj[u].end(); ++it2) {
                int w = *it2;
                if (adj[v].find(w) != adj[v].end()) {
                    triangles++;
                }
            }
        }
    }
    
    cout << triangles << "\n";
    return 0;
}
