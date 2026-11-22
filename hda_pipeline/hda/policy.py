"""Static policy check on HDA-written C++: standard library only, no I/O tricks,
no hardcoded answers. Returns a list of violations (empty list = accepted)."""
from __future__ import annotations
import re

ALLOWED_HEADERS = set("""
algorithm any array atomic bitset cassert cctype cerrno cfloat charconv chrono cinttypes climits
cmath complex cstddef cstdint cstdio cstdlib cstring ctime deque exception functional initializer_list
iomanip ios iosfwd iostream istream iterator limits list map memory memory_resource new numeric optional
ostream queue random ratio set sstream stack stdexcept string string_view system_error tuple
type_traits typeinfo unordered_map unordered_set utility valarray variant vector bits/stdc++.h
""".split())
# bits/stdc++.h is allowed: it only pulls in the standard library.

BANNED_PATTERNS = {
    r"\bsystem\s*\(": "calls system()",
    r"\bpopen\s*\(": "calls popen()",
    r"\bfork\s*\(": "calls fork()",
    r"\bexec[lv]p?e?\s*\(": "calls exec*()",
    r"\bfopen\s*\(|\bfreopen\s*\(|\bifstream\b|\bofstream\b|\bfstream\b": "file I/O (only stdin/stdout allowed)",
    r"\bstd::thread\b|\bthread\s*\(|\basync\s*\(|#\s*pragma\s+omp": "multithreading",
    r"\b__asm__\b|\basm\s*\(": "inline assembly",
    r"\bdlopen\b|\bsyscall\s*\(": "dynamic loading / raw syscalls",
    r"#\s*include\s*\"": "local include (single self-contained file required)",
}

def check_source(src: str, example_answers: list[str] | None = None,
                 max_big_literals: int = 24) -> list[str]:
    v = []
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)          # strip comments
    for h in re.findall(r"#\s*include\s*<([^>]+)>", code):
        if h.strip() not in ALLOWED_HEADERS:
            v.append(f"non-standard or disallowed header <{h}>")
    for pat, why in BANNED_PATTERNS.items():
        if re.search(pat, code):
            v.append(why)
    # Hardcoding heuristics: long literal tables, or string tables of answers.
    nums = re.findall(r"(?<![\w.])\d{2,}(?![\w.])", re.sub(r'"(?:\\.|[^"\\])*"', "", code))
    common = {"10", "100", "1000", "10000", "100000", "1000000", "1000000000", "1000000007",
              "998244353", "64", "32", "16", "128", "256", "512", "1024", "63", "31", "60", "62", "18", "20"}
    big = [x for x in nums if x not in common]
    if len(big) > max_big_literals:
        v.append(f"{len(big)} distinct-purpose numeric literals (possible hardcoded table)")
    if re.search(r"\{\s*(?:-?\d+\s*,\s*){20,}", code):
        v.append("large literal array initializer (possible hardcoded answers or graphs)")
    if example_answers:
        hits = [a for a in set(example_answers) if len(a) >= 3 and a.lstrip("-").isdigit()
                and re.search(rf"(?<![\w.]){re.escape(a)}(?![\w.])", code) and a not in common]
        if len(hits) >= 2:
            v.append(f"source contains {len(hits)} example answers verbatim: {sorted(hits)[:5]}")
    if len(src) > 60_000:
        v.append("source longer than 60k characters")
    return v
