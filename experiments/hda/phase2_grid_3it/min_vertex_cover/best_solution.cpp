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

// Optional helper: adjacency lists without weights
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

    vector<bool> in_cover(g.n, false);
    vector<bool> removed(g.n, false);

    while (true) {
        int max_degree = -1;
        int chosen_node = -1;

        for (int u = 0; u < g.n; ++u) {
            if (removed[u]) continue;
            int degree = adj[u].size();
            if (degree > max_degree) {
                max_degree = degree;
                chosen_node = u;
            }
        }

        if (chosen_node == -1) break;

        in_cover[chosen_node] = true;
        removed[chosen_node] = true;

        // Remove all edges incident to chosen_node
        for (int v : adj[chosen_node]) {
            if (!removed[v]) {
                auto it = find(adj[v].begin(), adj[v].end(), chosen_node);
                if (it != adj[v].end()) {
                    adj[v].erase(it);
                }
            }
        }
    }

    // Output the cover in sorted order
    vector<int> result;
    for (int i = 0; i < g.n; ++i) {
        if (in_cover[i]) {
            result.push_back(i);
        }
    }

    cout << "[";
    for (int i = 0; i < result.size(); ++i) {
        cout << result[i];
        if (i < result.size() - 1) cout << ",";
    }
    cout << "]\n";

    return 0;
}
