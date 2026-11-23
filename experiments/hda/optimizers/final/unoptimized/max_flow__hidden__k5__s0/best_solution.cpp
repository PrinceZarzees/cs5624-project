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
    
    if (g.query_k != 2) {
        cout << 0 << "\n";
        return 0;
    }
    
    int s = g.query[0], t = g.query[1];
    
    // Build residual graph for max flow
    vector<vector<pair<int, long long>>> graph(g.n);
    for (const auto &e : g.edges) {
        graph[e.u].push_back({e.v, e.w});
    }
    
    // Max flow using Edmonds-Karp (BFS for augmenting paths)
    auto max_flow = [&](int src, int sink) {
        vector<long long> dist(g.n, -1);
        vector<int> parent(g.n, -1);
        
        long long total_flow = 0;
        
        while (true) {
            // BFS to find augmenting path
            queue<int> q;
            q.push(src);
            dist.assign(g.n, -1);
            dist[src] = 0;
            parent.assign(g.n, -1);
            
            while (!q.empty()) {
                int u = q.front(); q.pop();
                if (u == sink) break;
                for (const auto &edge : graph[u]) {
                    int v = edge.first;
                    long long w = edge.second;
                    if (dist[v] == -1 && w > 0) {
                        dist[v] = dist[u] + 1;
                        parent[v] = u;
                        q.push(v);
                    }
                }
            }
            
            if (dist[sink] == -1) break;
            
            // Find minimum residual capacity on path
            long long path_flow = LLONG_MAX;
            int cur = sink;
            while (cur != src) {
                int prev = parent[cur];
                for (const auto &edge : graph[prev]) {
                    if (edge.first == cur) {
                        path_flow = min(path_flow, edge.second);
                        break;
                    }
                }
                cur = prev;
            }
            
            // Update residual capacities
            cur = sink;
            while (cur != src) {
                int prev = parent[cur];
                for (auto &edge : graph[prev]) {
                    if (edge.first == cur) {
                        edge.second -= path_flow;
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
