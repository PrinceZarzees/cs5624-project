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
    
    vector<int> color(g.n, -1); // -1: uncolored, 0 or 1: color
    bool is_bipartite = true;
    
    for (int u = 0; u < g.n; u++) {
        if (color[u] == -1) {
            queue<int> q;
            q.push(u);
            color[u] = 0;
            
            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                
                for (const auto &neighbor : adj[curr]) {
                    int v = neighbor.first;
                    if (color[v] == -1) {
                        color[v] = color[curr] ^ 1;
                        q.push(v);
                    } else if (color[v] == color[curr]) {
                        is_bipartite = false;
                        break;
                    }
                }
                if (!is_bipartite) break;
            }
            if (!is_bipartite) break;
        }
    }
    
    if (is_bipartite) {
        cout << "YES\n";
    } else {
        cout << "NO\n";
    }
    
    return 0;
}
