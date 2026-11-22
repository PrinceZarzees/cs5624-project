#include <iostream>
#include <vector>
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

static vector<vector<int>> adjacency(const Graph &g) {
    vector<vector<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back(e.v);
        if (!g.directed) adj[e.v].push_back(e.u);
    }
    return adj;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    auto adj = adjacency(g);

    vector<bool> selected(g.n, false);
    vector<int> result;

    while (true) {
        int best_node = -1;
        int min_degree = g.n + 1;

        for (int i = 0; i < g.n; ++i) {
            if (!selected[i]) {
                int deg = adj[i].size();
                if (deg < min_degree) {
                    min_degree = deg;
                    best_node = i;
                }
            }
        }

        if (best_node == -1) break;

        selected[best_node] = true;
        result.push_back(best_node);

        // Mark all neighbors as ineligible
        for (int neighbor : adj[best_node]) {
            selected[neighbor] = true;
        }
    }

    cout << "[";
    for (int i = 0; i < result.size(); ++i) {
        if (i > 0) cout << ",";
        cout << result[i];
    }
    cout << "]" << endl;

    return 0;
}
