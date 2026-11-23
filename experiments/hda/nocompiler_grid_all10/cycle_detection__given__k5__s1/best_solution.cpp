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
    
    vector<bool> visited(g.n, false);
    bool has_cycle = false;
    
    for (int i = 0; i < g.n; i++) {
        if (!visited[i]) {
            stack<int> stk;
            stk.push(i);
            visited[i] = true;
            
            while (!stk.empty()) {
                int u = stk.top();
                stk.pop();
                
                for (const auto &[v, _] : adj[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        stk.push(v);
                    } else {
                        // If v is visited and not the parent, we found a cycle
                        // But we don't track parent explicitly — so we need to avoid false positives
                        // Instead, we use a visited state and detect back edges
                        // Since we don't track parent, we can't distinguish from parent
                        // So we must modify: use parent tracking per DFS tree
                    }
                }
            }
        }
    }
    
    // Reimplement with parent tracking properly
    vector<bool> visited2(g.n, false);
    vector<int> parent(g.n, -1);
    has_cycle = false;
    
    for (int i = 0; i < g.n; i++) {
        if (!visited2[i]) {
            stack<int> stk;
            stk.push(i);
            visited2[i] = true;
            parent[i] = -1;
            
            while (!stk.empty()) {
                int u = stk.top();
                stk.pop();
                
                for (const auto &[v, _] : adj[u]) {
                    if (!visited2[v]) {
                        visited2[v] = true;
                        parent[v] = u;
                        stk.push(v);
                    } else {
                        // If v is visited and not the parent of u, then cycle exists
                        if (parent[u] != v) {
                            has_cycle = true;
                            break;
                        }
                    }
                }
                if (has_cycle) break;
            }
            if (has_cycle) break;
        }
    }
    
    cout << (has_cycle ? "YES" : "NO") << "\n";
    return 0;
}
