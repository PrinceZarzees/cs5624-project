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

// Find root with path compression
int find_root(vector<int>& parent, int x) {
    if (parent[x] != x) {
        parent[x] = find_root(parent, parent[x]);
    }
    return parent[x];
}

// Union two sets and return true if merged
bool union_sets(vector<int>& parent, int x, int y) {
    int rx = find_root(parent, x), ry = find_root(parent, y);
    if (rx == ry) return false;
    parent[rx] = ry;
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    
    // Use Kruskal's algorithm to find MST
    vector<Edge> sorted_edges = g.edges;
    sort(sorted_edges.begin(), sorted_edges.end(), [](const Edge& a, const Edge& b) {
        return a.w < b.w;
    });

    vector<int> parent(g.n);
    for (int i = 0; i < g.n; ++i) {
        parent[i] = i;
    }

    long long mst_weight = 0;
    for (const Edge& e : sorted_edges) {
        if (union_sets(parent, e.u, e.v)) {
            mst_weight += e.w;
        }
    }

    cout << mst_weight << "\n";
    return 0;
}
