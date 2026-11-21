//// HDA graph-loading template (given verbatim to the Harness Developing Agent).
// Input format (stdin):
//   line 1 : n m directed weighted      (directed, weighted are 0/1)
//   next m : u v        (weighted == 0)   or   u v w   (weighted == 1)
//            nodes are 0 .. n-1, no self-loops, no duplicate edges
//            (for directed graphs u->v and v->u may both appear)
//   last   : k q1 q2 ... qk               (query arguments; k may be 0)
// Output (stdout): exactly one line holding the answer.
// Rules: C++17, standard library only, no file I/O, no threads, single file.
//#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; long long w; };

struct Graph {
    int n = 0;
    long long m = 0;
    bool directed = false, weighted = false;
    vector<Edge> edges;          // edge list exactly as read
    vector<long long> query;     // query arguments (may be empty)
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
    if (cin >> k) { g.query.resize(k); for (auto &q : g.query) cin >> q; }
    return g;
}

// Build residual graph as adjacency list: res[u][v] = remaining capacity
static vector<vector<long long>> build_residual(const Graph& g, int s, int t) {
    vector<vector<long long>> res(g.n, vector<long long>(g.n, 0));
    for (const auto &e : g.edges) {
        if (e.u == s || e.v == t) continue;
        res[e.u][e.v] = e.w; // residual capacity
    }
    return res;
}

// BFS to find augmenting path from source to sink in residual graph
static bool bfs(const vector<vector<long long>>& res, int src, int dst, vector<int>& parent) {
    vector<bool> visited(res.size(), false);
    queue<int> q;
    q.push(src);
    visited[src] = true;
    parent[src] = -1;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v = 0; v < res[u].size(); ++v) {
            if (!visited[v] && res[u][v] > 0) {
                visited[v] = true;
                parent[v] = u;
                q.push(v);
                if (v == dst) return true;
            }
        }
    }
    return false;
}

// Compute maximum flow using Edmonds-Karp
static long long max_flow(const Graph& g, int s, int t) {
    if (s == t || g.n <= 0) return 0;

    vector<vector<long long>> residual = build_residual(g, s, t);
    
    long long total_flow = 0;
    while (true) {
        vector<int> parent(g.n, -1);
        if (!bfs(residual, s, t, parent)) break;

        // Find the minimum capacity along the path
        int current = t;
        long long path_flow = LONG_MAX;
        while (current != s) {
            int prev = parent[current];
            path_flow = min(path_flow, residual[prev][current]);
            current = prev;
        }

        // Update residual capacities
        current = t;
        while (current != s) {
            int prev = parent[current];
            residual[prev][current] -= path_flow;
            residual[current][prev] += path_flow;
            current = prev;
        }

        total_flow += path_flow;
    }
    return total_flow;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();

    // Check if query exists and is of type 2 (flow query)
    if (g.query.empty()) {
        cout << 0 << "\n";
        return 0;
    }

    int k = static_cast<int>(g.query.size());
    for (int i = 0; i < k; ++i) {
        if (g.query[i] == 2) {
            int s = g.query[++i];
            int t = g.query[++i];
            long long result = max_flow(g, s, t);
            cout << result << "\n";
            return 0;
        }
    }

    // No valid flow query found
    cout << 0 << "\n";
    return 0;
}
