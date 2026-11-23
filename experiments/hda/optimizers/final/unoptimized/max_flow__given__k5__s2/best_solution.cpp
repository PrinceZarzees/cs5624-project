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
    
    // Extract source and sink
    long long s = g.query[0], t = g.query[1];
    
    // Build residual graph as adjacency list
    vector<vector<pair<int, long long>>> residual(g.n);
    for (const auto &e : g.edges) {
        residual[e.u].push_back({e.v, e.w});
    }
    
    // Max flow using Edmonds-Karp (BFS-based Ford-Fulkerson)
    auto max_flow = [&](int src, int sink) {
        long long total_flow = 0;
        while (true) {
            // BFS to find augmenting path
            vector<int> parent(g.n, -1);
            vector<long long> dist(g.n, -1);
            queue<int> q;
            q.push(src);
            dist[src] = 0;
            
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (const auto &[v, cap] : residual[u]) {
                    if (dist[v] == -1 && cap > 0) {
                        dist[v] = dist[u] + 1;
                        parent[v] = u;
                        q.push(v);
                    }
                }
            }
            
            // If no path to sink, break
            if (dist[sink] == -1) break;
            
            // Find minimum capacity on the path
            long long path_flow = LLONG_MAX;
            int cur = sink;
            while (cur != src) {
                int prev = parent[cur];
                for (auto &[v, cap] : residual[prev]) {
                    if (v == cur) {
                        path_flow = min(path_flow, cap);
                        break;
                    }
                }
                cur = prev;
            }
            
            // Update residual capacities
            cur = sink;
            while (cur != src) {
                int prev = parent[cur];
                for (auto &[v, cap] : residual[prev]) {
                    if (v == cur) {
                        // Find and update the edge
                        auto it = find_if(residual[prev].begin(), residual[prev].end(),
                                          [&](const auto &p) { return p.first == v; });
                        if (it != residual[prev].end()) {
                            it->second -= path_flow;
                        }
                        // Add reverse edge if not present
                        bool found = false;
                        for (auto &rev : residual[v]) {
                            if (rev.first == prev) {
                                rev.second += path_flow;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            residual[v].push_back({prev, path_flow});
                        }
                        break;
                    }
                }
                cur = prev;
            }
            
            total_flow += path_flow;
        }
        return total_flow;
    };
    
    long long result = max_flow(s, t);
    cout << result << "\n";
    return 0;
}
