#include <iostream>
#include <vector>
#include <queue>
using namespace std;

struct Edge { int u, v; long long w; };

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;
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
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Graph g = read_graph();

    // For undirected graphs, check bipartiteness using BFS
    vector<int> color(g.n, -1); // -1: uncolored, 0 or 1: colors

    for (int i = 0; i < g.n; ++i) {
        if (color[i] == -1) {
            queue<int> q;
            q.push(i);
            color[i] = 0;

            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (const Edge& e : g.edges) {
                    if (e.u == u) {
                        if (color[e.v] == -1) {
                            color[e.v] = color[u] ^ 1;
                            q.push(e.v);
                        } else if (color[e.v] == color[u]) {
                            cout << "NO\n";
                            return 0;
                        }
                    }
                }
            }
        }
    }

    cout << "YES\n";
    return 0;
}
