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
    if (cin >> k) { 
        g.query_k = k; 
        g.query.resize(k); 
        for (auto &q : g.query) cin >> q; 
    }
    return g;
}

// Optional helper: adjacency lists (for undirected graphs each edge is added
// in both directions). Pair = (neighbor, weight).
static vector<vector<pair<int, long long>>> adjacency(const Graph &g) {
    vector<vector<pair<int, long long>>> adj(g.n);
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

    // Build neighbor lists (only node IDs)
    vector<vector<int>> neighbors(g.n);
    for (int u = 0; u < g.n; ++u) {
        for (const auto& p : adj[u]) {
            neighbors[u].push_back(p.first);
        }
    }

    int triangles = 0;
    for (int u = 0; u < g.n; ++u) {
        int n = neighbors[u].size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int v = neighbors[u][i];
                int w = neighbors[u][j];
                // Check if v and w are adjacent
                auto it = find(neighbors[v].begin(), neighbors[v].end(), w);
                if (it != neighbors[v].end()) {
                    triangles++;
                }
            }
        }
    }

    cout << triangles / 3 << "\n"; // Each triangle is counted three times (once per vertex)
    return 0;
}
