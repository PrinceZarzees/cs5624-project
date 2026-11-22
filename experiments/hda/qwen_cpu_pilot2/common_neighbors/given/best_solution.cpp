#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; long long w; };

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;
    vector<long long> query;
};

Graph read_graph() {
    Graph g;
    if (!(cin >> g.n >> g.m >> g.directed >> g.weighted)) return g;
    g.edges.resize(g.m);
    for (auto &e : g.edges) {
        cin >> e.u >> e.v;
        e.w = 1;
        if (g.weighted) cin >> e.w;
    }
    int k = 0;
    if (cin >> k) {
        g.query.resize(k);
        for (auto &q : g.query) cin >> q;
    }
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Graph g = read_graph();

    // Build adjacency list: undirected graph
    vector<vector<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back(e.v);
        if (!g.directed) {
            adj[e.v].push_back(e.u);
        }
    }

    int total_common_neighbors = 0;

    // Process each query
    for (int i = 0; i < g.query.size(); ++i) {
        if (g.query[i] == 2) {
            if (i + 1 >= g.query.size()) break;
            int u = g.query[i+1];
            int v = g.query[i+2];
            // Validate node indices
            if (u < 0 || u >= g.n || v < 0 || v >= g.n) continue;

            set<int> neighbors_u(adj[u].begin(), adj[u].end());
            set<int> neighbors_v(adj[v].begin(), adj[v].end());

            for (int w : neighbors_u) {
                if (neighbors_v.find(w) != neighbors_v.end()) {
                    total_common_neighbors++;
                }
            }
        }
    }

    cout << total_common_neighbors << "\n";
    return 0;
}
