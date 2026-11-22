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
    if (cin >> k) {
        g.query_k = k;
        g.query.resize(k);
        for (auto &q : g.query) {
            cin >> q;
        }
    }
    return g;
}

static vector<vector<pair<int, long long>>> adjacency(const Graph &g) {
    vector<vector<pair<int, long long>>> adj(g.n);
    for (const auto &e : g.edges) {
        adj[e.u].push_back({e.v, e.w});
        if (!g.directed) {
            adj[e.v].push_back({e.u, e.w});
        }
    }
    return adj;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Graph g = read_graph();
    auto adj = adjacency(g);

    // Precompute neighbor sets for fast lookup
    vector<set<int>> neighbors(g.n);
    for (int u = 0; u < g.n; ++u) {
        for (const auto &p : adj[u]) {
            neighbors[u].insert(p.first);
        }
    }

    int best_size = 0;
    vector<int> best_clique;

    // Randomized greedy: multiple restarts to find larger cliques
    const int trials = 10;  // small number for speed, can be increased if needed
    random_device rd;
    mt19937 gen(rd());

    for (int t = 0; t < trials; ++t) {
        vector<int> nodes(g.n);
        iota(nodes.begin(), nodes.end(), 0);  // [0, 1, ..., n-1]
        shuffle(nodes.begin(), nodes.end(), gen);

        vector<int> clique;
        for (int u : nodes) {
            bool can_add = true;
            for (int v : clique) {
                if (neighbors[v].find(u) == neighbors[v].end()) {
                    can_add = false;
                    break;
                }
            }
            if (can_add) {
                clique.push_back(u);
            }
        }

        // Update best if larger
        if (clique.size() > best_size) {
            best_size = clique.size();
            best_clique = clique;
        }
    }

    sort(best_clique.begin(), best_clique.end());
    
    cout << "[";
    for (int i = 0; i < best_clique.size(); ++i) {
        if (i > 0) cout << ",";
        cout << best_clique[i];
    }
    cout << "]" << endl;
    
    return 0;
}
