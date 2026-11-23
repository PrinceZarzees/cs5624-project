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
    
    if (g.n == 0) {
        cout << 0 << "\n";
        return 0;
    }
    
    // Build reverse adjacency list
    vector<vector<int>> rev_adj(g.n);
    for (const auto &e : g.edges) {
        rev_adj[e.v].push_back(e.u);
    }
    
    // Step 1: DFS on original graph to get finishing order
    vector<bool> visited(g.n, false);
    vector<int> order;
    function<void(int)> dfs1 = [&](int u) {
        visited[u] = true;
        for (const auto &v : adj[u]) {
            if (!visited[v.first]) {
                dfs1(v.first);
            }
        }
        order.push_back(u);
    };
    
    for (int i = 0; i < g.n; ++i) {
        if (!visited[i]) {
            dfs1(i);
        }
    }
    
    // Step 2: DFS on reverse graph in reverse order
    vector<bool> visited2(g.n, false);
    vector<int> scc_sizes;
    function<void(int)> dfs2 = [&](int u) {
        visited2[u] = true;
        scc_sizes.back()++;
        for (int v : rev_adj[u]) {
            if (!visited2[v]) {
                dfs2(v);
            }
        }
    };
    
    reverse(order.begin(), order.end());
    for (int u : order) {
        if (!visited2[u]) {
            scc_sizes.push_back(0);
            dfs2(u);
        }
    }
    
    // Answer is the size of the largest SCC
    int max_size = 0;
    for (int size : scc_sizes) {
        if (size > max_size) {
            max_size = size;
        }
    }
    
    cout << max_size << "\n";
    return 0;
}
