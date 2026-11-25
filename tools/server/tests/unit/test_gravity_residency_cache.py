import glob
import os
import re
import shutil
import sys
import tempfile
import pytest
from utils import *

# > 32 tokens: the prompt runs in one ubatch too large for the MoE cache, so the experts are computed from the host mmap
PROMPT = ("Once upon a time there was a little girl named Lily. She loved to play outside in the park with her friends. "
          "One day she saw a big red ball in the tall green grass near the old tree, and she ran to get it before her brother could")

BUDGET_MIB = 8


def require_gpu():
    # the MoE cache lives on a GPU: without one the server refuses --moe-cache-mib and never starts
    server = ServerPreset.stories15m_moe()
    server.start()
    try:
        devices = server.make_request("GET", "/props").body.get("gravity", {}).get("devices_available", [])
    finally:
        server.stop()
    if not [d for d in devices if d != "CPU"]:
        pytest.skip("no GPU device registered, no MoE cache")


def private_model(tmp_path) -> str:
    # mincore sees the page cache of the file, which other servers using the shared model file fill:
    # a private copy (own inode) counts only the pages of this server
    src = glob.glob(os.path.join(TMP_DIR, "models--ggml-org-dev--stories15M_MOE", "snapshots", "*", "stories15M_MOE-F16.gguf"))
    if not src:
        pytest.skip("stories15M_MOE-F16.gguf not in the test model cache")
    dst = str(tmp_path / "stories15M_MOE-F16-private.gguf")
    shutil.copyfile(src[0], dst)
    return dst


def run_server(monkeypatch, on: bool, model_file: str | None = None):
    for name in ("LLAMA_ARG_CPU_MOE", "LLAMA_ARG_MOE_CACHE_MIB", "LLAMA_ARG_GRAVITY_STREAM", "LLAMA_ARG_GRAVITY_RESIDENCY_MIB",
                 "LLAMA_ARG_GRAVITY_RESIDENCY_DIAG", "LLAMA_GRAVITY_STREAM_IO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLAMA_ARG_FIT", "off")
    if on:
        monkeypatch.setenv("LLAMA_ARG_CPU_MOE", "1")
        monkeypatch.setenv("LLAMA_ARG_MOE_CACHE_MIB", "8")
        monkeypatch.setenv("LLAMA_ARG_GRAVITY_STREAM", "1")
        monkeypatch.setenv("LLAMA_ARG_GRAVITY_RESIDENCY_MIB", str(BUDGET_MIB))
        monkeypatch.setenv("LLAMA_ARG_GRAVITY_RESIDENCY_DIAG", "1")
        # page cache reads of the misses would bring the dropped file pages back into core (mincore counts them)
        monkeypatch.setenv("LLAMA_GRAVITY_STREAM_IO", "direct")

    server = ServerPreset.stories15m_moe()
    if model_file:
        server.model_file = model_file
        server.model_hf_repo = None
        server.model_hf_file = None
    server.n_gpu_layer = 99
    server.debug = True # the residency stats are INFO lines
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)

    try:
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": PROMPT,
            "n_predict": 64,
            "temperature": 0.0,
            "seed": 42,
        })
        assert res.status_code == 200
        content = res.body["content"]
    finally:
        # the residency prints its stats when the context is freed
        server.stop()
    with open(server.log_path, errors="replace") as f:
        log = f.read()
    os.remove(server.log_path)
    return content, log


def test_gravity_residency_cache(monkeypatch, tmp_path):
    if not (sys.platform == "darwin" or sys.platform.startswith("linux")):
        pytest.skip("no page drop on this OS")
    require_gpu()
    off, _      = run_server(monkeypatch, on=False)
    on,  log_on = run_server(monkeypatch, on=True, model_file=private_model(tmp_path))

    stats = [l for l in log_on.splitlines() if "gravity residency" in l or "MoE cache size" in l or "gravity stream:" in l]
    info  = "\n".join(stats)
    assert on == off, f"text differs, residency on:\n{on!r}\nall off:\n{off!r}\nlog:\n{info}"

    if "MoE cache size" not in log_on:
        pytest.skip("no MoE cache on this host (no GPU)")
    m = re.search(r"gravity residency: budget [0-9.]+ MiB, est resident [0-9.]+ MiB of ([0-9.]+) MiB, mincore ([0-9.-]+) MiB, "
                  r"steps [0-9]+ \(([0-9]+) host", log_on)
    assert m is not None, f"no gravity residency stats line, log:\n{info}"
    total, mincore, n_host = float(m.group(1)), float(m.group(2)), int(m.group(3))
    d = re.search(r"gravity residency diag: dropped ranges had ([0-9]+) pages in core before drop, ([0-9]+) right after", log_on)
    assert d is not None, f"no gravity residency diag line, log:\n{info}"
    before, after = int(d.group(1)), int(d.group(2))
    nums = f"mincore {mincore} MiB, total {total} MiB, host steps {n_host}, diag pages {before} -> {after}, log:\n{info}"
    assert n_host >= 1, f"the prompt did not run the experts from the host mmap: {nums}"
    if before > 0 and after >= before:
        # Linux MADV_DONTNEED unmaps the pages but the page cache keeps them, so mincore does not change
        pytest.skip(f"the page drop does not change mincore on this OS: {nums}")
    # one layer of slack: expert slices are not page aligned, and edge pages stay
    n_layer = 6
    assert 0 <= mincore <= BUDGET_MIB + total / n_layer, f"over budget {BUDGET_MIB} MiB + one layer: {nums}"
