//// HDA graph-loading template (given verbatim to the Harness Developing Agent).
// Input format (stdin):
//   line 1 : n m directed weighted      (directed, weighted are 0/1)
//   next m : u v        (weighted == 0)   or   u v w   (weighted == 1)
//            nodes are 0 .. n-1, no self-loops, no duplicate edges
//            (for directed graphs u->v and v->u may both appear)
//   last   : k q1 q2 ... qk               (query arguments; k may be 0)
// Output (stdout): exactly one line holding the answer.
// Rules: C++17, standard library only, no file I/O, no threads, single file.
//#include <bits/stdc++.h>
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

// Iterative BFS to compute distances from source
vector<int> bfs(const Graph& g, int source) {
    vector<int> dist(g.n, -1);
    queue<int> q;
    dist[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (const auto &neighbor : adjacency(g)[u]) {
            if (dist[neighbor.first] == -1) {
                dist[neighbor.first] = dist[u] + 1;
                q.push(neighbor.first);
            }
        }
    }

    return dist;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Graph g = read_graph();
    auto adj = adjacency(g);

    // If there is only one node, diameter is 0
    if (g.n == 1) {
        cout << 0 << "\n";
        return 0;
    }

    // First BFS from node 0 to find a farthest node
    vector<int> dist0 = bfs(g, 0);
    int u = 0;
    for (int i = 1; i < g.n; ++i) {
        if (dist0[i] > dist0[u]) {
            u = i;
        }
    }

    // Second BFS from node u to find the farthest distance
    vector<int> distu = bfs(g, u);
    int diameter = *max_element(distu.begin(), distu.end());

    cout << diameter << "\n";
    return 0;
}
