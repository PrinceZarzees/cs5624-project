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
    int s=g.query[0], t=g.query[1];
    struct E{int to; long long cap;}; vector<E> E_; vector<vector<int>> G(g.n);
    for(auto&e:g.edges){ G[e.u].push_back(E_.size()); E_.push_back({e.v,e.w}); G[e.v].push_back(E_.size()); E_.push_back({e.u,0}); }
    vector<int> lvl(g.n), it(g.n); long long flow=0;
    auto bfs=[&]{ fill(lvl.begin(),lvl.end(),-1); queue<int> q; q.push(s); lvl[s]=0;
      while(!q.empty()){int u=q.front();q.pop(); for(int id:G[u]) if(E_[id].cap>0&&lvl[E_[id].to]<0){lvl[E_[id].to]=lvl[u]+1;q.push(E_[id].to);}} return lvl[t]>=0; };
    // iterative DFS for blocking flow (avoids deep recursion on long paths)
    auto dfs=[&](long long lim){ long long pushed=0;
      while(true){ vector<int> path; int u=s; bool found=false;
        while(true){ if(u==t){found=true;break;}
          bool adv=false; for(int &i=it[u]; i<(int)G[u].size(); i++){int id=G[u][i]; int v=E_[id].to; if(E_[id].cap>0&&lvl[v]==lvl[u]+1){path.push_back(id);u=v;adv=true;break;}}
          if(!adv){ if(u==s) break; lvl[u]=-1; int id=path.back(); path.pop_back(); u=E_[id^1].to; it[u]++; } }
        if(!found) break; long long b=lim; for(int id:path) b=min(b,E_[id].cap);
        for(int id:path){E_[id].cap-=b;E_[id^1].cap+=b;} pushed+=b; }
      return pushed; };
    if(s!=t) while(bfs()){ fill(it.begin(),it.end(),0); flow+=dfs(LLONG_MAX); }
    cout<<flow<<"\n";
    return 0;
}
