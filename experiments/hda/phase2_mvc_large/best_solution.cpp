#include <iostream>
#include <vector>
#include <algorithm>
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

    vector<bool> in_cover(g.n, false);
    vector<int> cover;

    while (true) {
        int max_degree = -1;
        int chosen_node = -1;

        for (int u = 0; u < g.n; ++u) {
            if (in_cover[u]) continue;
            int deg = adj[u].size();
            if (deg > max_degree) {
                max_degree = deg;
                chosen_node = u;
            }
        }

        if (chosen_node == -1) break;

        in_cover[chosen_node] = true;
        cover.push_back(chosen_node);

        for (const auto &[v, _] : adj[chosen_node]) {
            in_cover[v] = true;
        }
    }

    cout << "[";
    for (int i = 0; i < cover.size(); ++i) {
        if (i > 0) cout << ",";
        cout << cover[i];
    }
    cout << "]\n";

    return 0;
}
