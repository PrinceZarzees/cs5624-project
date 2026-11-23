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

// Residual graph: capacity from u to v
vector<vector<long long>> residual;

// BFS to find augmenting path from s to t
long long bfs(int s, int t, vector<int>& parent) {
    vector<bool> visited(s + 1, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v = 0; v < residual[u].size(); ++v) {
            if (residual[u][v] > 0 && !visited[v]) {
                visited[v] = true;
                parent[v] = u;
                q.push(v);
                if (v == t) return 1;
            }
        }
    }
    return 0;
}

// Edmonds-Karp: compute max flow from s to t
long long max_flow(int s, int t, const Graph& g) {
    int n = g.n;
    residual.resize(n, vector<long long>(n, 0));
    for (const auto& e : g.edges) {
        residual[e.u][e.v] += e.w;
    }

    long long total_flow = 0;
    while (true) {
        vector<int> parent(n, -1);
        long long path_flow = bfs(s, t, parent);
        if (path_flow == 0) break;

        // Update residual capacities along the path
        int v = t;
        while (v != s) {
            int u = parent[v];
            residual[u][v] -= path_flow;
            residual[v][u] += path_flow;
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
    auto adj = adjacency(g);
    (void)adj;

    if (g.query_k != 2) {
        cout << 0 << "\n";
        return 0;
    }

    int s = g.query[0], t = g.query[1];
    long long result = max_flow(s, t, g);
    cout << result << "\n";
    return 0;
}
