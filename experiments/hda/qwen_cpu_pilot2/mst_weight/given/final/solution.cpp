#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v;
    long long w;
};

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;          // edge list exactly as read
    vector<long long> query;     // query arguments (may be empty)
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

// Union-Find with path compression and union by rank
struct DSU {
    vector<int> parent;
    vector<int> rank;
    
    DSU(int n) {
        parent.assign(n, 0);
        rank.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // path compression
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) rank[x]++;
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Graph g = read_graph();
    
    // Only process undirected weighted graphs
    if (g.directed || !g.weighted) {
        cout << 0 << "\n";
        return 0;
    }

    // Sort edges by weight
    sort(g.edges.begin(), g.edges.end(), [](const Edge& a, const Edge& b) {
        return a.w < b.w;
    });

    DSU dsu(g.n);
    long long total_weight = 0;
    int edge_count = 0;

    for (const auto& e : g.edges) {
        if (dsu.unite(e.u, e.v)) {
            total_weight += e.w;
            edge_count++;
            if (edge_count == g.n - 1) break;
        }
    }

    cout << total_weight << "\n";
    return 0;
}
