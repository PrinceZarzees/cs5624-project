#include <iostream>
#include <vector>
#include <queue>
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

// Optional helper: adjacency lists (for undirected graphs each edge is added
// in both directions). Pair = (neighbor, weight).
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

    // BFS from a starting node to find the farthest node
    vector<int> dist(g.n, -1);
    queue<int> q;
    dist[0] = 0;
    q.push(0);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (const auto &[v, _] : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    // Find the farthest node from start
    int u = 0;
    for (int i = 1; i < g.n; ++i) {
        if (dist[i] > dist[u]) u = i;
    }

    // BFS from that farthest node to compute diameter
    vector<int> dist2(g.n, -1);
    queue<int> q2;
    dist2[u] = 0;
    q2.push(u);

    while (!q2.empty()) {
        int v = q2.front(); q2.pop();
        for (const auto &[w, _] : adj[v]) {
            if (dist2[w] == -1) {
                dist2[w] = dist2[v] + 1;
                q2.push(w);
            }
        }
    }

    // The maximum distance is the diameter
    int diameter = *max_element(dist2.begin(), dist2.end());
    cout << diameter << "\n";
    return 0;
}
