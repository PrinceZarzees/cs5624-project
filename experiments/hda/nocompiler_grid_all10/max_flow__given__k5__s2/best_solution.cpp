// ---------------------------------------------------------------------------
// HDA graph-loading template (given verbatim to the Harness Developing Agent).
// Input format (stdin):
//   line 1 : n m directed weighted      (directed, weighted are 0/1)
//   next m : u v        (weighted == 0)   or   u v w   (weighted == 1)
//            nodes are 0 .. n-1, no self-loops, no duplicate edges
//            (for directed graphs u->v and v->u may both appear)
//   last   : k q1 q2 ... qk               (k = NUMBER of query arguments; not a type code)
// Output (stdout): exactly one line holding the answer.
// Rules: C++17, standard library only, no file I/O, no threads, single file.
// ---------------------------------------------------------------------------
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
    
    // Build residual graph: residual[u][v] = capacity from u to v
    vector<vector<long long>> residual(g.n, vector<long long>(g.n, 0));
    for (const auto &e : g.edges) {
        residual[e.u][e.v] = e.w;
    }

    // Query: k=2, s and t
    long long s = g.query[0], t = g.query[1];
    
    // Max flow using Edmonds-Karp (BFS-based Ford-Fulkerson)
    long long max_flow = 0;
    while (true) {
        // BFS to find augmenting path from s to t
        vector<int> parent(g.n, -1);
        vector<bool> visited(g.n, false);
        queue<int> q;
        q.push(s);
        visited[s] = true;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v = 0; v < g.n; ++v) {
                if (visited[v] || residual[u][v] == 0) continue;
                visited[v] = true;
                parent[v] = u;
                q.push(v);
                if (v == t) break;
            }
        }

        // If no path found, break
        if (!visited[t]) break;

        // Find minimum residual capacity on the path
        long long path_flow = LLONG_MAX;
        int v = t;
        while (v != s) {
            int u = parent[v];
            path_flow = min(path_flow, residual[u][v]);
            v = u;
        }

        // Update residual capacities
        v = t;
        while (v != s) {
            int u = parent[v];
            residual[u][v] -= path_flow;
            residual[v][u] += path_flow;
            v = u;
        }

        max_flow += path_flow;
    }

    cout << max_flow << "\n";
    return 0;
}
