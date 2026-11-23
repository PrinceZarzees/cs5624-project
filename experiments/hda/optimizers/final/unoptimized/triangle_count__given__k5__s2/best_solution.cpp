#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; long long w; };

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;
    int query_k = 0;
    vector<long long> query;
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

static vector<set<int>> adjacency(const Graph &g) {
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
    auto adj = adjacency(g);
    
    long long triangles = 0;
    // Process each edge only once: u < v
    for (const auto &e : g.edges) {
        int u = e.u, v = e.v;
        if (u > v) swap(u, v);
        // For each neighbor w of v, check if w is in adj[u]
        for (int w : adj[v]) {
            if (w == u) continue;
            if (adj[u].find(w) != adj[u].end()) {
                triangles++;
            }
        }
    }
    triangles /= 3;
    cout << triangles << "\n";
    return 0;
}
