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
    
    if (g.query_k == 0) {
        // Entire graph has a cycle if edges >= nodes
        if (g.m >= g.n) {
            cout << "YES\n";
        } else {
            cout << "NO\n";
        }
        return 0;
    }
    
    if (g.query_k != 2) {
        cout << "NO\n";
        return 0;
    }
    
    int s = g.query[0], t = g.query[1];
    
    // BFS to find component of s and t
    vector<bool> visited(g.n, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    vector<int> comp_nodes;
    comp_nodes.push_back(s);
    
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (const auto &neigh : adj[u]) {
            if (!visited[neigh.first]) {
                visited[neigh.first] = true;
                q.push(neigh.first);
                comp_nodes.push_back(neigh.first);
            }
        }
    }
    
    // Include t only if it's in the component (should be, since we start from s)
    // Count edges within the component
    int comp_size = comp_nodes.size();
    int comp_edges = 0;
    for (const auto &e : g.edges) {
        if (find(comp_nodes.begin(), comp_nodes.end(), e.u) != comp_nodes.end() &&
            find(comp_nodes.begin(), comp_nodes.end(), e.v) != comp_nodes.end()) {
            comp_edges++;
        }
    }
    
    // Has cycle if edges >= nodes
    if (comp_edges >= comp_size) {
        cout << "YES\n";
    } else {
        cout << "NO\n";
    }
    
    return 0;
}
