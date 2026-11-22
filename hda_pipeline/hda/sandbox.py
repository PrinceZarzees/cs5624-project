"""Compile and run HDA C++ solutions under resource limits.

Limits per run: CPU time = time_limit_s (RLIMIT_CPU, plus a wall-clock kill at
2x + 1 s), address space 1 GiB, stack 8 MiB (Linux default, stated in the
prompt), output 1 MiB, no child processes. Each solution runs in an empty
temporary directory; test inputs are piped on stdin, never placed beside it.
"""
from __future__ import annotations
import os, resource, shutil, signal, subprocess, tempfile, threading, time
from dataclasses import dataclass, asdict
from pathlib import Path

GXX = os.environ.get("HDA_GXX", "g++")
CXXFLAGS = ["-std=c++17", "-O2", "-pipe"]
MEM_BYTES = 1 << 30
STACK_BYTES = 8 << 20
OUT_BYTES = 1 << 20


@dataclass
class RunResult:
    verdict: str          # AC, WA, TLE, RE, MLE, OLE (or OK when no expected answer)
    output: str
    stderr: str
    cpu_s: float
    wall_s: float
    returncode: int

    def to_dict(self):
        return asdict(self)


def compile_cpp(src: str, workdir: Path, timeout_s: int = 60) -> tuple[bool, str, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    cpp, exe = workdir / "solution.cpp", workdir / "solution"
    cpp.write_text(src)
    if exe.exists():
        exe.unlink()
    try:
        r = subprocess.run([GXX, *CXXFLAGS, "-o", str(exe), str(cpp)], capture_output=True,
                           text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "compilation timed out", exe
    return r.returncode == 0, (r.stderr or "")[-4000:], exe


def _limits(cpu_s: float, out_bytes: int = OUT_BYTES):
    def f():
        os.setsid()
        c = max(1, int(cpu_s + 0.999))
        resource.setrlimit(resource.RLIMIT_CPU, (c, c + 1))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_STACK, (STACK_BYTES, STACK_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (out_bytes, out_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64)) if os.getuid() != 0 else None
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


def normalize(ans: str) -> str:
    s = (ans or "").strip().splitlines()
    s = s[0].strip() if s else ""
    return s.upper() if s.upper() in ("YES", "NO") else s


def run_binary(exe: Path, input_text: str, time_limit_s: float, expected: str | None = None) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="hda_run_") as td:
        inp, out, err = Path(td) / "in.txt", Path(td) / "out.txt", Path(td) / "err.txt"
        inp.write_text(input_text)
        run_dir = Path(td) / "cwd"; run_dir.mkdir()
        with open(inp) as fi, open(out, "w") as fo, open(err, "w") as fe:
            t0 = time.perf_counter()
            p = subprocess.Popen([str(exe.resolve())], stdin=fi, stdout=fo, stderr=fe, cwd=run_dir,
                                 preexec_fn=_limits(time_limit_s))
            killed = {"wall": False}
            def killer():
                killed["wall"] = True
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            timer = threading.Timer(2 * time_limit_s + 1, killer); timer.start()
            _, status, ru = os.wait4(p.pid, 0)
            timer.cancel()
            p.returncode = os.waitstatus_to_exitcode(status)
            wall = time.perf_counter() - t0
        cpu = ru.ru_utime + ru.ru_stime
        output = out.read_text(errors="replace")[:OUT_BYTES]
        stderr = err.read_text(errors="replace")[-2000:]
    rc = p.returncode
    if killed["wall"] or cpu > time_limit_s or rc in (-signal.SIGXCPU, -signal.SIGKILL) and cpu >= time_limit_s * 0.95:
        v = "TLE"
    elif rc == -signal.SIGXFSZ or len(output) >= OUT_BYTES:
        v = "OLE"
    elif rc != 0:
        v = "MLE" if ("bad_alloc" in stderr or rc == -signal.SIGABRT and "alloc" in stderr) else "RE"
    elif expected is None:
        v = "OK"
    else:
        v = "AC" if normalize(output) == normalize(expected) else "WA"
    return RunResult(v, output[:2000], stderr, round(cpu, 4), round(wall, 4), rc)


def run_capture(exe: Path, time_limit_s: float, max_bytes: int = 64 << 20) -> str | None:
    """Run a (generator) binary with no stdin under the same limits; return full stdout or None."""
    lim = _limits(time_limit_s, max_bytes)
    with tempfile.TemporaryDirectory(prefix="hda_gen_") as td:
        try:
            p = subprocess.run([str(exe.resolve())], stdin=subprocess.DEVNULL, capture_output=True, cwd=td,
                               preexec_fn=lim, timeout=2 * time_limit_s + 1)
        except subprocess.TimeoutExpired:
            return None
    if p.returncode != 0 or len(p.stdout) > max_bytes:
        return None
    return p.stdout.decode(errors="replace")
