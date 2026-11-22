#include <bits/stdc++.h>
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
    
    if (g.query_k >= 2 && g.query[0] < g.n && g.query[1] < g.n) {
        vector<bool> visited(g.n, false);
        stack<int> stk;
        stk.push(g.query[0]);
        visited[g.query[0]] = true;

        while (!stk.empty()) {
            int u = stk.top();
            stk.pop();
            if (u == g.query[1]) {
                cout << "YES\n";
                return 0;
            }
            for (const auto &e : g.edges) {
                if (e.u == u && !visited[e.v]) {
                    visited[e.v] = true;
                    stk.push(e.v);
                }
            }
        }
    }
    cout << "NO\n";
    return 0;
}
