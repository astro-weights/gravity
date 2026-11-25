#!/usr/bin/env python3
"""Benchmark stock mmap vs MoE cache vs --gravity-stream on one GGUF.

Runs llama-completion once per (rep, cache mode, arm), cold and warm page cache,
and writes everything to one JSON file. Stdlib only, Python >= 3.8.
See docs/gravity-bench.md.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

ARMS = ("stock", "cache", "stream")
MODES = ("cold", "warm")
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

# -lv 4: the llama_moe_cache hit/miss and stream lines are logged at debug level.
# Same flag on every arm so the logging cost is shared.
FIXED_ARGS = ["--seed", "1", "--temp", "0", "-no-cnv", "-ngl", "99", "-cmoe", "-fit", "off", "-lv", "4"]

_TS = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ")
_LOAD = re.compile(r"common_perf_print:\s+load time\s*=\s*([\d.]+) ms")
_PROMPT = re.compile(r"common_perf_print:\s+prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?,\s*([\d.]+) tokens per second")
_EVAL = re.compile(r"common_perf_print:\s+eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs.*?,\s*([\d.]+) tokens per second")
_UBATCH = re.compile(r"llama_moe_cache: ubatch\s*(<=|>)\s*\d+: hits = (\d+), misses = (\d+) \((\d+) on the CPU\), "
                     r"hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB")
_STREAM = re.compile(r"gravity stream: read = ([\d.]+) MiB from file in ([\d.]+) ms, fallbacks = (\d+) \(([\d.]+) MiB from host memory\)")


def parse_stderr(text):
    """Pull the perf and cache numbers out of llama-completion stderr."""
    out = {"load_ms": None, "prompt_eval_ms": None, "prompt_tokens": None, "prompt_tok_s": None,
           "eval_ms": None, "eval_runs": None, "eval_tok_s": None, "ttft_ms": None,
           "moe_cache": {}, "stream": None}
    for line in text.splitlines():
        line = _TS.sub("", line)
        m = _LOAD.search(line)
        if m:
            out["load_ms"] = float(m.group(1))
            continue
        m = _PROMPT.search(line)
        if m:
            out["prompt_eval_ms"], out["prompt_tokens"], out["prompt_tok_s"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
            continue
        m = _EVAL.search(line)
        if m:
            out["eval_ms"], out["eval_runs"], out["eval_tok_s"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
            continue
        m = _UBATCH.search(line)
        if m:
            key = "small" if m.group(1) == "<=" else "large"
            out["moe_cache"][key] = {"hits": int(m.group(2)), "misses": int(m.group(3)), "cpu_misses": int(m.group(4)),
                                     "hit_rate_pct": float(m.group(5)), "uploaded_mib": float(m.group(6))}
            continue
        m = _STREAM.search(line)
        if m:
            out["stream"] = {"read_mib": float(m.group(1)), "read_ms": float(m.group(2)),
                             "fallbacks": int(m.group(3)), "fallback_mib": float(m.group(4))}
    # load-independent: prompt eval + the first generated token (mean eval step)
    if out["prompt_eval_ms"] is not None and out["eval_ms"] is not None and out["eval_runs"]:
        out["ttft_ms"] = out["prompt_eval_ms"] + out["eval_ms"] / out["eval_runs"]
    return out


def build_argv(args, arm):
    """argv for one arm; arms differ only in the cache / stream flags."""
    argv = [args.bin, "-m", args.model]
    argv += ["-f", args.prompt_file] if args.prompt_file else ["-p", args.prompt]
    argv += ["-n", str(args.n_predict)] + FIXED_ARGS
    if args.threads:
        argv += ["-t", str(args.threads)]
    if arm in ("cache", "stream"):
        argv += ["--moe-cache-mib", str(args.cache_mib)]
    if arm == "stream":
        argv += ["--gravity-stream"]
        if args.gravity_stream_threads:
            argv += ["--gravity-stream-threads", str(args.gravity_stream_threads)]
    if args.extra:
        argv += shlex.split(args.extra)
    return argv


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def meminfo(key):
    if not IS_LINUX:
        return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def top_procs(n=5):
    try:
        out = subprocess.run(["ps", "-Ao", "pid,pcpu,comm"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines()[1:]:
        p = line.split(None, 2)
        if len(p) == 3 and p[0].isdigit() and int(p[0]) != os.getpid():
            try:
                rows.append((float(p[1]), os.path.basename(p[2])))
            except ValueError:
                pass
    rows.sort(reverse=True)
    return [{"pcpu": c, "comm": k} for c, k in rows[:n]]


def drop_cache(args):
    """Make the page cache cold for the model file. Returns (ok, note, meminfo dict)."""
    mi = {"cached_kib_before_drop": meminfo("Cached")}
    if hasattr(os, "posix_fadvise"):
        try:
            fd = os.open(args.model, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as e:
            return False, "fadvise failed: %s" % e, mi
        mi["cached_kib_after_drop"] = meminfo("Cached")
        return True, "posix_fadvise(DONTNEED)", mi
    if IS_MAC and args.mac_purge:
        r = subprocess.run(["sudo", "-n", "purge"], capture_output=True, text=True)
        if r.returncode == 0:
            return True, "sudo -n purge", mi
        return False, "cold unavailable: sudo -n purge failed (%s)" % r.stderr.strip(), mi
    return False, "cold unavailable: no posix_fadvise on this OS (use --mac-purge on macOS)", mi


def read_proc_io(pid):
    try:
        with open("/proc/%d/io" % pid) as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def run_once(argv, timeout):
    """Run argv; returns dict with rusage, io, wall, stdout bytes, stderr text."""
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
        t0 = time.monotonic()
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe)
        io_last = None
        status = ru = None
        timed_out = False
        while True:
            pid, st, r = os.wait4(p.pid, os.WNOHANG)
            if pid:
                status, ru = st, r
                break
            if IS_LINUX:
                v = read_proc_io(p.pid)
                if v is not None:
                    io_last = v
            if timeout and time.monotonic() - t0 > timeout:
                timed_out = True
                p.kill()
                _, status, ru = os.wait4(p.pid, 0)
                break
            time.sleep(0.1)
        wall = time.monotonic() - t0
        p.returncode = 0  # reaped by wait4 above; stops Popen from waiting again
        fo.seek(0)
        fe.seek(0)
        stdout, stderr = fo.read(), fe.read().decode("utf-8", "replace")
    if os.WIFEXITED(status):
        code = os.WEXITSTATUS(status)
    else:
        code = -os.WTERMSIG(status)
    rss = ru.ru_maxrss if IS_MAC else ru.ru_maxrss * 1024  # Linux KiB, macOS bytes
    return {"exit_code": code, "timed_out": timed_out, "wall_s": wall, "stdout": stdout, "stderr": stderr,
            "max_rss_bytes": rss, "majflt": ru.ru_majflt, "minflt": ru.ru_minflt,
            "bytes_read_blocks": ru.ru_inblock * 512, "proc_io_read_bytes": io_last}


def one_run(args, arm, mode, rep, keep_stderr_dir):
    argv = build_argv(args, arm)
    rec = {"arm": arm, "cache_mode": mode, "rep": rep, "argv": argv}
    if mode == "cold":
        ok, note, mi = drop_cache(args)
        rec["cold_method"] = note
        rec["meminfo"] = mi
        if not ok:
            rec.update(skipped=True, skip_reason=note)
            return rec
    else:
        rec["warmup_exit_code"] = run_once(argv, args.timeout)["exit_code"]
    rec["load_before"] = os.getloadavg()
    rec["top_procs_before"] = top_procs()
    r = run_once(argv, args.timeout)
    rec["load_after"] = os.getloadavg()
    if mode == "cold" and IS_LINUX:
        rec["meminfo"]["cached_kib_after_run"] = meminfo("Cached")
    rec["stdout_sha256"] = hashlib.sha256(r["stdout"]).hexdigest()
    rec["stdout_bytes"] = len(r["stdout"])
    for k in ("exit_code", "timed_out", "wall_s", "max_rss_bytes", "majflt", "minflt", "bytes_read_blocks", "proc_io_read_bytes"):
        rec[k] = r[k]
    rec.update(parse_stderr(r["stderr"]))
    if keep_stderr_dir and (r["exit_code"] != 0 or args.keep_stderr):
        path = os.path.join(keep_stderr_dir, "%s-%s-%d.stderr.txt" % (arm, mode, rep))
        with open(path, "w") as f:
            f.write(r["stderr"])
        rec["stderr_file"] = path
    elif r["exit_code"] != 0:
        rec["stderr_tail"] = r["stderr"][-2000:]
    return rec


def summarize(runs):
    rows = []
    for mode in MODES:
        for arm in ARMS:
            rs = [r for r in runs if r["arm"] == arm and r["cache_mode"] == mode and not r.get("skipped") and r["exit_code"] == 0]
            if not rs:
                continue
            st = [r["stream"] for r in rs if r.get("stream")]
            rows.append({"arm": arm, "cache_mode": mode, "n": len(rs),
                         "median_eval_tok_s": median([r["eval_tok_s"] for r in rs]),
                         "median_ttft_ms": median([r["ttft_ms"] for r in rs]),
                         "median_max_rss_mib": median([r["max_rss_bytes"] / 1048576.0 for r in rs]),
                         "median_majflt": median([r["majflt"] for r in rs]),
                         "median_bytes_read": median([r["bytes_read_blocks"] for r in rs]),
                         "stream_read_mib_median": median([s["read_mib"] for s in st]),
                         "stream_fallbacks_total": sum(s["fallbacks"] for s in st) if st else None})
    return rows


def text_identity(runs):
    out = {}
    for mode in MODES:
        h = {}
        for arm in ("cache", "stream"):
            h[arm] = {r["stdout_sha256"] for r in runs if r["arm"] == arm and r["cache_mode"] == mode
                      and not r.get("skipped") and r["exit_code"] == 0}
        if h["cache"] and h["stream"]:
            out[mode] = len(h["cache"]) == 1 and h["cache"] == h["stream"]
    return out


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def print_summary(rows, ident):
    print("%-7s %-5s %2s %12s %10s %10s %9s %14s" % ("arm", "cache", "n", "eval tok/s", "ttft ms", "RSS MiB", "majflt", "bytes read"))
    for r in rows:
        print("%-7s %-5s %2d %12s %10s %10s %9s %14s" % (
            r["arm"], r["cache_mode"], r["n"], fmt(r["median_eval_tok_s"], ".2f"), fmt(r["median_ttft_ms"], ".1f"),
            fmt(r["median_max_rss_mib"], ".1f"), fmt(r["median_majflt"], ".0f"), fmt(r["median_bytes_read"], ".0f")))
    for r in rows:
        if r["arm"] == "stream":
            print("stream %s: median read %s MiB, fallbacks total %s" % (r["cache_mode"], fmt(r["stream_read_mib_median"], ".2f"), r["stream_fallbacks_total"]))
    for mode, same in ident.items():
        print("stream text == cache text (%s): %s" % (mode, "yes" if same else "no"))


def host_info():
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "mem_total_kib": meminfo("MemTotal") if IS_LINUX else mac_mem_kib()}


def mac_mem_kib():
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip()) // 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def git_sha(binpath):
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(binpath)), "..", ".."))
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def model_info(path, want_sha):
    st = os.stat(path)
    info = {"path": os.path.abspath(path), "size_bytes": st.st_size,
            "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat()}
    if want_sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", help="path to llama-completion")
    ap.add_argument("-m", "--model", help="GGUF file")
    ap.add_argument("--cache-mib", type=int, help="--moe-cache-mib for the cache and stream arms")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--prompt-file")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--cache-modes", default=None, help="cold,warm (macOS default: warm)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--gravity-stream-threads", type=int)
    ap.add_argument("--extra", default="", help="extra llama-completion args for every arm, one quoted string")
    ap.add_argument("--mac-purge", action="store_true", help="macOS: use 'sudo -n purge' for cold runs")
    ap.add_argument("--model-sha256", action="store_true", help="hash the model (slow on big files)")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per run, 0 = none")
    ap.add_argument("--keep-stderr", action="store_true", help="save every run's stderr next to --out")
    ap.add_argument("--out", help="JSON output path")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return a
    for req in ("bin", "model", "cache_mib", "out"):
        if getattr(a, req) is None:
            ap.error("--%s is required" % req.replace("_", "-"))
    a.arms = [x for x in a.arms.split(",") if x]
    if a.cache_modes is None:
        a.cache_modes = "warm" if IS_MAC and not a.mac_purge else "cold,warm"
    a.cache_modes = [x for x in a.cache_modes.split(",") if x]
    for x in a.arms:
        if x not in ARMS:
            ap.error("unknown arm %r" % x)
    for x in a.cache_modes:
        if x not in MODES:
            ap.error("unknown cache mode %r" % x)
    return a


def main(argv):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    notes = []
    if IS_MAC and "cold" not in args.cache_modes:
        notes.append("macOS: warm only (no posix_fadvise; pass --mac-purge for cold)")
    data = {"header": {"host": host_info(), "gravity_git_sha": git_sha(args.bin), "bin": os.path.abspath(args.bin),
                       "model": model_info(args.model, args.model_sha256), "args": vars(args), "notes": notes,
                       "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            "runs": [], "summary": [], "text_identity": {}}
    failed = 0
    try:
        for rep in range(1, args.reps + 1):
            for mode in args.cache_modes:
                for arm in args.arms:  # interleaved: every arm once per rep, so drift hits all arms
                    print("rep %d %s %s ..." % (rep, mode, arm), file=sys.stderr, flush=True)
                    rec = one_run(args, arm, mode, rep, out_dir)
                    if not rec.get("skipped") and rec["exit_code"] != 0:
                        failed += 1
                        print("  FAILED exit %s" % rec["exit_code"], file=sys.stderr)
                    data["runs"].append(rec)
    finally:
        data["summary"] = summarize(data["runs"])
        data["text_identity"] = text_identity(data["runs"])
        data["header"]["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(args.out, "w") as f:
            json.dump(data, f, indent=1)
    for n in notes:
        print("note:", n)
    skipped = {r["skip_reason"] for r in data["runs"] if r.get("skipped")}
    for s in sorted(skipped):
        print("skipped:", s)
    print_summary(data["summary"], data["text_identity"])
    print("wrote", args.out)
    return 1 if failed else 0


CANNED = """0.00.052.640 I llama_model_loader: loaded meta data
0.00.062.600 I impl:       MTL0 MoE cache size =     7.59 MiB for 30.38 MiB of host experts, 0% of the misses on the CPU
0.00.184.674 I llama_moe_cache: ubatch <= 8: hits = 56, misses = 734 (0 on the CPU), hit rate = 7.09%, uploaded = 928.97 MiB
0.00.184.675 I llama_moe_cache: gravity stream: read = 928.97 MiB from file in 48.3 ms, fallbacks = 0 (0.00 MiB from host memory)
0.00.296.932 I common_perf_print:    samplers time =       0.32 ms /    21 tokens
0.00.296.933 I common_perf_print:        load time =      23.53 ms
0.00.296.934 I common_perf_print: prompt eval time =       8.88 ms /     5 tokens (    1.78 ms per token,   562.87 tokens per second)
0.00.296.934 I common_perf_print:        eval time =      82.34 ms /    15 runs   (    5.49 ms per token,   182.18 tokens per second)
0.00.296.935 I common_perf_print:       total time =     110.60 ms /    20 tokens
"""


def self_test():
    p = parse_stderr(CANNED)
    assert p["load_ms"] == 23.53, p
    assert (p["prompt_eval_ms"], p["prompt_tokens"], p["prompt_tok_s"]) == (8.88, 5, 562.87), p
    assert (p["eval_ms"], p["eval_runs"], p["eval_tok_s"]) == (82.34, 15, 182.18), p
    assert abs(p["ttft_ms"] - (8.88 + 82.34 / 15)) < 1e-9, p
    assert p["moe_cache"]["small"] == {"hits": 56, "misses": 734, "cpu_misses": 0, "hit_rate_pct": 7.09, "uploaded_mib": 928.97}, p
    assert "large" not in p["moe_cache"]
    assert p["stream"] == {"read_mib": 928.97, "read_ms": 48.3, "fallbacks": 0, "fallback_mib": 0.0}, p
    # no stream line -> None; the "ubatch  > 8" line (two spaces) is the large bucket
    q = parse_stderr("0.00.1.2 I llama_moe_cache: ubatch  > 8: hits = 1, misses = 2 (0 on the CPU), hit rate = 33.33%, uploaded = 1.50 MiB\n")
    assert q["stream"] is None and q["moe_cache"]["large"]["misses"] == 2 and q["ttft_ms"] is None, q

    a = parse_args(["--bin", "/b/llama-completion", "-m", "m.gguf", "--cache-mib", "64", "--n-predict", "32",
                    "--threads", "8", "--gravity-stream-threads", "6", "--extra", "-c 512 --no-warmup", "--out", "o.json"])
    stock, cache, stream = (build_argv(a, x) for x in ARMS)
    assert stock[:3] == ["/b/llama-completion", "-m", "m.gguf"] and "--moe-cache-mib" not in stock and "--gravity-stream" not in stock
    assert cache[cache.index("--moe-cache-mib") + 1] == "64" and "--gravity-stream" not in cache
    assert "--gravity-stream" in stream and stream[stream.index("--gravity-stream-threads") + 1] == "6"
    for argv in (stock, cache, stream):
        assert argv[argv.index("-n") + 1] == "32" and argv[argv.index("-t") + 1] == "8"
        assert argv[argv.index("-p") + 1] == "Once upon a time"
        assert all(f in argv for f in ("-cmoe", "-no-cnv", "--seed", "-ngl")) and argv[-3:] == ["-c", "512", "--no-warmup"]
    a.prompt_file = "p.txt"
    assert "-f" in build_argv(a, "stock") and "-p" not in build_argv(a, "stock")

    runs = [{"arm": "cache", "cache_mode": "warm", "exit_code": 0, "stdout_sha256": "x"},
            {"arm": "stream", "cache_mode": "warm", "exit_code": 0, "stdout_sha256": "x"},
            {"arm": "cache", "cache_mode": "cold", "exit_code": 0, "stdout_sha256": "x"},
            {"arm": "stream", "cache_mode": "cold", "exit_code": 0, "stdout_sha256": "y"}]
    assert text_identity(runs) == {"cold": False, "warm": True}
    assert median([3, 1, None, 2]) == 2 and median([None]) is None
    print("self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
