#include <iostream>
#include <vector>
using namespace std;

struct Edge { int u, v; long long w; };

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;
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
        g.query.resize(k); 
        for (auto &q : g.query) cin >> q; 
    }
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Graph g = read_graph();

    // Union-Find data structure
    vector<int> parent(g.n), rank(g.n, 0);
    for (int i = 0; i < g.n; ++i) {
        parent[i] = i;
    }

    auto find = [&](int x) -> int {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    };

    auto union_nodes = [&](int u, int v) {
        int root_u = find(u), root_v = find(v);
        if (root_u == root_v) return;
        if (rank[root_u] < rank[root_v]) {
            parent[root_u] = root_v;
        } else if (rank[root_u] > rank[root_v]) {
            parent[root_v] = root_u;
        } else {
            parent[root_v] = root_u;
            ++rank[root_u];
        }
    };

    // Process all edges
    for (const auto &e : g.edges) {
        union_nodes(e.u, e.v);
    }

    // Count connected components
    int components = 0;
    for (int i = 0; i < g.n; ++i) {
        if (find(i) == i) {
            ++components;
        }
    }

    cout << components << "\n";
    return 0;
}
