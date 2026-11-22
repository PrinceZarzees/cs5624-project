#include <iostream>
#include <vector>
#include <set>
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

Graph read_graph() {
    Graph g;
    int d = 0, w = 0;
    cin >> g.n >> g.m >> d >> w;
    g.directed = (d == 1);
    g.weighted = (w == 1);
    g.edges.resize(g.m);
    for (auto &e : g.edges) {
        cin >> e.u >> e.v;
        if (g.weighted) cin >> e.w;
        else e.w = 1;
    }
    cin >> g.query_k;
    g.query.resize(g.query_k);
    for (auto &q : g.query) cin >> q;
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Graph g = read_graph();

    if (g.query_k < 2) {
        cout << 0 << "\n";
        return 0;
    }

    int u = g.query[0], v = g.query[1];

    // Build neighbor sets for u and v
    set<int> neighbors_u, neighbors_v;

    for (const Edge& e : g.edges) {
        if (e.u == u || e.v == u) {
            if (e.u == u) {
                neighbors_u.insert(e.v);
            } else {
                neighbors_u.insert(e.u);
            }
        }
        if (e.u == v || e.v == v) {
            if (e.u == v) {
                neighbors_v.insert(e.v);
            } else {
                neighbors_v.insert(e.u);
            }
        }
    }

    // Count common neighbors
    int count = 0;
    for (int node : neighbors_u) {
        if (neighbors_v.find(node) != neighbors_v.end()) {
            count++;
        }
    }

    cout << count << "\n";
    return 0;
}
