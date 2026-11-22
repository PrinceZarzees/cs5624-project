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
    if (cin >> k) { 
        g.query_k = k; 
        g.query.resize(k); 
        for (auto &q : g.query) cin >> q; 
    }
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

    // If no valid query or invalid indices
    if (g.query_k != 2 || g.query[0] < 0 || g.query[1] < 0 || g.query[0] >= g.n || g.query[1] >= g.n) {
        cout << 0 << "\n";
        return 0;
    }

    int s = g.query[0], t = g.query[1];
    if (s == t) {
        cout << 0 << "\n";
        return 0;
    }

    // Residual graph: capacity from u to v
    vector<vector<long long>> cap(g.n, vector<long long>(g.n, 0));
    for (const auto &e : g.edges) {
        cap[e.u][e.v] = e.w;
    }

    long long max_flow = 0;

    // Edmonds-Karp: BFS to find augmenting path
    while (true) {
        vector<int> parent(g.n, -1);
        queue<int> q;
        q.push(s);
        parent[s] = -1;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v = 0; v < g.n; ++v) {
                if (parent[v] == -1 && cap[u][v] > 0) {
                    parent[v] = u;
                    if (v == t) break;
                    q.push(v);
                }
            }
        }

        // If no path to t, we're done
        if (parent[t] == -1) break;

        // Find minimum residual capacity on the path
        long long flow = LLONG_MAX;  // Use LLONG_MAX for safety with large flows
        int cur = t;
        while (cur != s) {
            int prev = parent[cur];
            flow = min(flow, cap[prev][cur]);
            cur = prev;
        }

        // Update residual capacities
        cur = t;
        while (cur != s) {
            int prev = parent[cur];
            cap[prev][cur] -= flow;
            cap[cur][prev] += flow;  // reverse edge in residual graph
            cur = prev;
        }

        max_flow += flow;
    }

    cout << max_flow << "\n";
    return 0;
}
