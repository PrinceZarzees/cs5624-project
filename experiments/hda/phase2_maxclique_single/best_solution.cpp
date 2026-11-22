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

static vector<vector<pair<int, long long>>> adjacency(const Graph &g) {
    vector<vector<pair<int, long long>>> adj(g.n);
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

    // If no nodes or edges, output empty clique
    if (g.n == 0 || g.m == 0) {
        cout << "[]" << endl;
        return 0;
    }

    set<int> current_clique;

    // Start with a pair of adjacent nodes — find any edge and use it as seed
    bool found_edge = false;
    for (const auto& e : g.edges) {
        int u = e.u, v = e.v;
        if (u == v) continue;  // skip self-loop (not present)
        current_clique.insert(u);
        current_clique.insert(v);
        found_edge = true;
        break;
    }

    if (!found_edge) {
        // Should not happen
        cout << "[]" << endl;
        return 0;
    }

    // Grow the clique iteratively until no more nodes can be added
    bool changed;
    do {
        changed = false;
        for (int node = 0; node < g.n; ++node) {
            if (current_clique.find(node) != current_clique.end()) continue;

            // Check if this node is adjacent to every member of the current clique
            bool can_add = true;
            for (const int& c : current_clique) {
                bool found_edge = false;
                for (const auto& neighbor : adj[node]) {
                    if (neighbor.first == c) {
                        found_edge = true;
                        break;
                    }
                }
                if (!found_edge) {
                    can_add = false;
                    break;
                }
            }

            if (can_add) {
                current_clique.insert(node);
                changed = true;
            }
        }
    } while (changed);

    // Convert to sorted vector
    vector<int> result(current_clique.begin(), current_clique.end());
    sort(result.begin(), result.end());

    cout << "[";
    for (int i = 0; i < result.size(); ++i) {
        if (i > 0) cout << ",";
        cout << result[i];
    }
    cout << "]" << endl;

    return 0;
}
