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
    vector<bool> visited(g.n, false);
    // DFS to find all reachable nodes from each node
    auto dfs = [&](auto&& self, int u) -> void {
        visited[u] = true;
        for (const auto& [v, w] : adj[u]) {
            if (!visited[v]) {
                self(self, v);
            }
        }
    };

    bool strongly_connected = true;
    for (int i = 0; i < g.n; i++) {
        // Reset visited
        fill(visited.begin(), visited.end(), false);
        dfs(dfs, i);
        // Check if all nodes are reachable
        for (int j = 0; j < g.n; j++) {
            if (!visited[j]) {
                strongly_connected = false;
                break;
            }
        }
        if (!strongly_connected) break;
    }

    cout << (strongly_connected ? "YES" : "NO") << "\n";
    return 0;
}
