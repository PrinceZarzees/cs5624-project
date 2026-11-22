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
        for (auto &q : g.query) cin >> q; 
    }
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

    // Find connected components using BFS
    vector<bool> visited(g.n, false);
    vector<int> component_id(g.n, -1);
    int comp_idx = 0;

    for (int i = 0; i < g.n; ++i) {
        if (!visited[i]) {
            queue<int> q;
            q.push(i);
            visited[i] = true;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                component_id[u] = comp_idx;
                for (auto &p : adj[u]) {
                    int v = p.first;
                    if (!visited[v]) {
                        visited[v] = true;
                        q.push(v);
                        component_id[v] = comp_idx;
                    }
                }
            }
            ++comp_idx;
        }
    }

    // For each connected component, find a maximum clique using greedy degree-based method
    vector<int> best_clique;
    for (int c = 0; c < comp_idx; ++c) {
        vector<int> nodes_in_comp;
        for (int i = 0; i < g.n; ++i) {
            if (component_id[i] == c) {
                nodes_in_comp.push_back(i);
            }
        }

        // Compute degrees within this component
        vector<int> local_degrees(nodes_in_comp.size(), 0);
        for (size_t j = 0; j < nodes_in_comp.size(); ++j) {
            int u = nodes_in_comp[j];
            for (auto &p : adj[u]) {
                if (component_id[p.first] == c) {
                    local_degrees[j]++;
                }
            }
        }

        // Sort by decreasing degree
        vector<int> node_order;
        for (int i = 0; i < nodes_in_comp.size(); ++i) {
            node_order.push_back(i);
        }
        sort(node_order.begin(), node_order.end(), [&](int a, int b) {
            return local_degrees[a] > local_degrees[b];
        });

        // Build clique greedily
        vector<int> current_clique;
        for (int idx : node_order) {
            int u = nodes_in_comp[idx];
            if (current_clique.empty()) {
                current_clique.push_back(u);
            } else {
                bool can_add = true;
                for (int v : current_clique) {
                    bool connected = false;
                    for (auto &p : adj[v]) {
                        if (p.first == u && component_id[p.first] == c) {
                            connected = true;
                            break;
                        }
                    }
                    if (!connected) {
                        can_add = false;
                        break;
                    }
                }
                if (can_add) {
                    current_clique.push_back(u);
                }
            }
        }

        // Update best clique if this one is larger
        if (current_clique.size() > best_clique.size()) {
            best_clique = current_clique;
        }
    }

    // Output result in required format
    cout << "[";
    for (int i = 0; i < best_clique.size(); ++i) {
        cout << best_clique[i];
        if (i < best_clique.size() - 1) {
            cout << ",";
        }
    }
    cout << "]" << endl;
    
    return 0;
}
