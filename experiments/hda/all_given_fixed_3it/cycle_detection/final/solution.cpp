#include <iostream>
#include <vector>
#include <functional>
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
    if (cin >> k) { 
        g.query_k = k; 
        g.query.resize(k); 
        for (auto &q : g.query) cin >> q; 
    }
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    
    // Build adjacency list for undirected graph
    vector<vector<int>> adj(g.n);
    for (const auto& e : g.edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }
    
    // DFS to detect cycle
    vector<bool> visited(g.n, false);
    vector<int> parent(g.n, -1);
    bool has_cycle = false;

    function<void(int)> dfs = [&](int u) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) {
                parent[v] = u;
                dfs(v);
            } else if (v != parent[u]) {
                has_cycle = true;
                return;
            }
        }
    };

    for (int i = 0; i < g.n; ++i) {
        if (!visited[i]) {
            dfs(i);
        }
    }

    cout << (has_cycle ? "YES" : "NO") << "\n";
    
    return 0;
}
