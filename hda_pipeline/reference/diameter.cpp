// ---------------------------------------------------------------------------
// HDA graph-loading template (given verbatim to the Harness Developing Agent).
// Input format (stdin):
//   line 1 : n m directed weighted      (directed, weighted are 0/1)
//   next m : u v        (weighted == 0)   or   u v w   (weighted == 1)
//            nodes are 0 .. n-1, no self-loops, no duplicate edges
//            (for directed graphs u->v and v->u may both appear)
//   last   : k q1 q2 ... qk               (query arguments; k may be 0)
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
    (void)adj;
    int best=0; vector<int> d(g.n); vector<int> q(g.n);
    for(int s=0;s<g.n;s++){ fill(d.begin(),d.end(),-1); int h=0,tl=0; q[tl++]=s; d[s]=0;
      while(h<tl){int u=q[h++]; for(auto [v,w]:adj[u]) if(d[v]<0){d[v]=d[u]+1;q[tl++]=v;}}
      best=max(best,d[q[tl-1]]); }
    cout<<best<<"\n";
    return 0;
}
