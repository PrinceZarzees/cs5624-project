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

static vector<vector<int>> adjacency(const Graph &g) {
    vector<vector<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }
    return adj;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    auto adj = adjacency(g);

    vector<bool> active(g.n, true);
    vector<int> result;

    // Create node-degree map and sort by degree descending
    vector<pair<int, int>> nodes;  // (degree, node)
    for (int u = 0; u < g.n; ++u) {
        nodes.push_back({adj[u].size(), u});
    }
    sort(nodes.begin(), nodes.end(), greater<pair<int, int>>());

    for (auto &[deg, u] : nodes) {
        if (!active[u]) continue;
        result.push_back(u);
        active[u] = false;
        // Mark neighbors as inactive
        for (int v : adj[u]) {
            active[v] = false;
        }
    }

    cout << "[";
    for (int i = 0; i < result.size(); ++i) {
        if (i > 0) cout << ",";
        cout << result[i];
    }
    cout << "]" << "\n";
    return 0;
}
