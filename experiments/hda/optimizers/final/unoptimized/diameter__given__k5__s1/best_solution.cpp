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

// Optional helper: adjacency lists (for undirected graphs each edge is added
// in both directions). Pair = (neighbor, weight).
static vector<vector<pair<int,long long>>> adjacency(const Graph &g) {
    vector<vector<pair<int,long long>>> adj(g.n);
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
    
    int diameter = 0;
    
    // Try BFS from each node to find the farthest node and compute diameter
    for (int start = 0; start < g.n; start++) {
        vector<int> dist(g.n, -1);
        queue<int> q;
        dist[start] = 0;
        q.push(start);
        
        int farthest = start;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (const auto &ne : adj[v]) {
                if (dist[ne.first] == -1) {
                    dist[ne.first] = dist[v] + 1;
                    if (dist[ne.first] > dist[farthest]) {
                        farthest = ne.first;
                    }
                    q.push(ne.first);
                }
            }
        }
        
        // BFS from farthest node to find the diameter
        vector<int> dist2(g.n, -1);
        queue<int> q2;
        dist2[farthest] = 0;
        q2.push(farthest);
        
        int current_diameter = 0;
        while (!q2.empty()) {
            int v = q2.front(); q2.pop();
            for (const auto &ne : adj[v]) {
                if (dist2[ne.first] == -1) {
                    dist2[ne.first] = dist2[v] + 1;
                    current_diameter = max(current_diameter, dist2[ne.first]);
                    q2.push(ne.first);
                }
            }
        }
        
        diameter = max(diameter, current_diameter);
    }
    
    cout << diameter << "\n";
    return 0;
}
