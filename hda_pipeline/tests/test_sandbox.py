from pathlib import Path
from hda.sandbox import compile_cpp, run_binary
from hda.policy import check_source

T = Path(__file__).resolve().parent.parent / "templates" / "graph_template.cpp"
HEAD = T.read_text().split("int main() {")[0]

def build(tmp_path, body):
    ok, log, exe = compile_cpp(HEAD + "int main(){ Graph g = read_graph(); auto adj = adjacency(g);" + body + " return 0; }", tmp_path)
    assert ok, log
    return exe

PATH = lambda n: f"{n} {n-1} 0 0\n" + "".join(f"{i} {i+1}\n" for i in range(n - 1)) + "0\n"

def test_verdicts(tmp_path):
    exe = build(tmp_path / "a", 'cout << g.n << "\\n";')
    assert run_binary(exe, PATH(5), 1, "5").verdict == "AC"
    assert run_binary(exe, PATH(5), 1, "4").verdict == "WA"
    exe = build(tmp_path / "b", "volatile long long x=0; while(true) x++;")
    r = run_binary(exe, PATH(5), 1, "5"); assert r.verdict == "TLE", r
    exe = build(tmp_path / "c", "vector<long long> v; while(true) v.resize(v.size()*2+1024);")
    assert run_binary(exe, PATH(5), 1, "5").verdict in ("MLE", "RE")

def test_deep_recursion_hits_8mb_stack(tmp_path):
    body = ("vector<char> vis(g.n,0); function<void(int)> dfs=[&](int u){vis[u]=1; for(auto [v,w]:adj[u]) if(!vis[v]) dfs(v);};"
            "int c=0; for(int i=0;i<g.n;i++) if(!vis[i]){c++; dfs(i);} cout<<c<<\"\\n\";")
    exe = build(tmp_path / "d", body)
    assert run_binary(exe, PATH(1000), 1, "1").verdict == "AC"
    assert run_binary(exe, PATH(300000), 5, "1").verdict == "RE"

def test_compile_error(tmp_path):
    ok, log, _ = compile_cpp("int main(){ return x; }", tmp_path)
    assert not ok and "x" in log

def test_policy():
    assert check_source(HEAD + "int main(){}") == []
    assert any("header" in v for v in check_source("#include <boost/graph/adjacency_list.hpp>\nint main(){}"))
    assert any("file I/O" in v for v in check_source("#include <cstdio>\nint main(){ fopen(\"x\",\"r\"); }"))
    assert any("system" in v for v in check_source("#include <cstdlib>\nint main(){ system(\"ls\"); }"))
    tbl = "int a[] = {" + ",".join(str(1000 + i) for i in range(40)) + "};"
    assert check_source("#include <vector>\n" + tbl + " int main(){}")
    assert any("example answers" in v for v in check_source("int main(){ if(1) puts(\"81114\"); else puts(\"98930\"); }", ["81114", "98930"]))
