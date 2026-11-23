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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    
    if (g.query_k != 2) {
        cout << "NO\n";
        return 0;
    }
    
    // Build adjacency list (only indices)
    vector<vector<int>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back(e.v);
    }
    
    // Build reverse adjacency list
    vector<vector<int>> rev(g.n);
    for (const auto &e : g.edges) {
        rev[e.v].push_back(e.u);
    }
    
    // First DFS: get finishing order (iterative)
    vector<int> order;
    vector<bool> visited1(g.n, false);
    for (int i = 0; i < g.n; ++i) {
        if (!visited1[i]) {
            stack<int> st;
            st.push(i);
            while (!st.empty()) {
                int u = st.top();
                visited1[u] = true;
                bool found = false;
                for (int v : adj[u]) {
                    if (!visited1[v]) {
                        st.push(v);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    order.push_back(u);
                    st.pop();
                }
            }
        }
    }
    reverse(order.begin(), order.end());
    
    // Second DFS: assign SCC components
    vector<int> comp(g.n, -1);
    vector<bool> visited2(g.n, false);
    int scc_id = 0;
    
    for (int u : order) {
        if (!visited2[u]) {
            stack<int> st;
            st.push(u);
            while (!st.empty()) {
                int v = st.top();
                if (visited2[v]) {
                    st.pop();
                    continue;
                }
                visited2[v] = true;
                comp[v] = scc_id;
                for (int w : rev[v]) {
                    if (!visited2[w]) {
                        st.push(w);
                    }
                }
            }
            scc_id++;
        }
    }
    
    int u = g.query[0], v = g.query[1];
    if (comp[u] == comp[v]) {
        cout << "YES\n";
    } else {
        cout << "NO\n";
    }
    
    return 0;
}
