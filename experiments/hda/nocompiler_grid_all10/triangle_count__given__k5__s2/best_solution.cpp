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
    
    // Build neighbor sets for each node
    vector<set<int>> neighbors(g.n);
    for (const auto &e : g.edges) {
        neighbors[e.u].insert(e.v);
        if (!g.directed) {
            neighbors[e.v].insert(e.u);
        }
    }
    
    long long triangle_count = 0;
    for (int u = 0; u < g.n; u++) {
        auto& neighbor_set = neighbors[u];
        for (auto it1 = neighbor_set.begin(); it1 != neighbor_set.end(); ++it1) {
            for (auto it2 = next(it1); it2 != neighbor_set.end(); ++it2) {
                // Check if *it1 and *it2 are adjacent
                if (neighbors[*it1].find(*it2) != neighbors[*it1].end()) {
                    triangle_count++;
                }
            }
        }
    }
    
    // Each triangle is counted 3 times (once for each vertex)
    cout << triangle_count / 3 << "\n";
    return 0;
}
