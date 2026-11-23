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

// Strongly connected components using Kosaraju's algorithm
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    
    if (g.n == 1) {
        cout << "YES\n";
        return 0;
    }
    
    // Build adjacency list
    vector<vector<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back(e.v);
    }
    
    // Step 1: DFS to get order of nodes
    vector<bool> visited(g.n, false);
    vector<int> order;
    
    function<void(int)> dfs1 = [&](int u) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) {
                dfs1(v);
            }
        }
        order.push_back(u);
    };
    
    for (int i = 0; i < g.n; ++i) {
        if (!visited[i]) {
            dfs1(i);
        }
    }
    
    // Step 2: Build reverse graph
    vector<vector<int>> rev_adj(g.n);
    for (const auto &e : g.edges) {
        rev_adj[e.v].push_back(e.u);
    }
    
    // Step 3: DFS on reverse graph in reverse order
    vector<bool> rev_visited(g.n, false);
    int scc_count = 0;
    
    function<void(int)> dfs2 = [&](int u) {
        rev_visited[u] = true;
        for (int v : rev_adj[u]) {
            if (!rev_visited[v]) {
                dfs2(v);
            }
        }
    };
    
    // Check if all nodes are in one SCC
    for (int i = g.n - 1; i >= 0; --i) {
        int u = order[i];
        if (!rev_visited[u]) {
            dfs2(u);
            scc_count++;
        }
    }
    
    // Strongly connected if only one SCC and n > 1
    cout << (scc_count == 1 ? "YES" : "NO") << "\n";
    return 0;
}
