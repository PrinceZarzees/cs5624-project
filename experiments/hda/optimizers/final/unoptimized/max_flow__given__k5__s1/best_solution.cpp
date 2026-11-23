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

// Adjacency list with residual capacities
struct ResidualGraph {
    vector<vector<pair<int, long long>>> adj;
    ResidualGraph(int n) : adj(n) {}
    void add_edge(int u, int v, long long cap) {
        adj[u].push_back({v, cap});
    }
    long long get_capacity(int u, int v) const {
        for (auto &e : adj[u]) {
            if (e.first == v) return e.second;
        }
        return 0;
    }
    void subtract_capacity(int u, int v, long long flow) {
        for (auto &e : adj[u]) {
            if (e.first == v) {
                e.second -= flow;
                return;
            }
        }
    }
    void add_reverse(int u, int v, long long flow) {
        // Reverse edge is implicitly added when we do BFS and update
        // We don't store it explicitly here — we rely on the fact that
        // we only traverse forward edges and update reverse edges in residual
        // But we need to maintain reverse edges in the graph.
        // Instead, we will modify the residual graph to store reverse edges.
    }
};

// We'll use a more efficient residual graph with explicit reverse edges
long long max_flow(const Graph &g, int s, int t) {
    vector<vector<pair<int, long long>>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back({e.v, e.w});
    }

    // Residual graph: for each node, list of (neighbor, capacity)
    vector<vector<pair<int, long long>>> residual(g.n);
    for (const auto &e : g.edges) {
        residual[e.u].push_back({e.v, e.w});
    }

    long long total_flow = 0;

    while (true) {
        // BFS to find path from s to t
        queue<int> q;
        vector<int> parent(g.n, -1);
        vector<bool> visited(g.n, false);
        
        q.push(s);
        visited[s] = true;
        
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (const auto &ne : residual[u]) {
                int v = ne.first;
                long long cap = ne.second;
                if (!visited[v] && cap > 0) {
                    visited[v] = true;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        if (!visited[t]) break;

        // Find minimum residual capacity on path
        long long path_flow = LLONG_MAX;
        int v = t;
        while (v != s) {
            int u = parent[v];
            for (const auto &ne : residual[u]) {
                if (ne.first == v) {
                    path_flow = min(path_flow, ne.second);
                    break;
                }
            }
            v = u;
        }

        // Update residual capacities
        v = t;
        while (v != s) {
            int u = parent[v];
            for (auto &ne : residual[u]) {
                if (ne.first == v) {
                    ne.second -= path_flow;
                    break;
                }
            }
            // Add reverse edge if not present
            for (auto &ne : residual[v]) {
                if (ne.first == u) {
                    ne.second += path_flow;
                    break;
                }
            }
            v = u;
        }

        total_flow += path_flow;
    }

    return total_flow;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    if (g.query_k < 2) {
        cout << 0 << "\n";
        return 0;
    }
    int s = g.query[0], t = g.query[1];
    long long flow = max_flow(g, s, t);
    cout << flow << "\n";
    return 0;
}
