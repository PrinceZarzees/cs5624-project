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
    int k = 0;
    // We don't need query data since we're only computing components
    return g;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    
    vector<bool> visited(g.n, false);
    int components = 0;

    for (int i = 0; i < g.n; ++i) {
        if (!visited[i]) {
            components++;
            queue<int> q;
            q.push(i);
            visited[i] = true;
            
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                // For undirected graph, add both directions
                for (const auto &edge : g.edges) {
                    if (edge.u == u && !visited[edge.v]) {
                        visited[edge.v] = true;
                        q.push(edge.v);
                    }
                    if (!g.directed && edge.v == u && !visited[edge.u]) {
                        visited[edge.u] = true;
                        q.push(edge.u);
                    }
                }
            }
        }
    }

    cout << components << "\n";
    return 0;
}
