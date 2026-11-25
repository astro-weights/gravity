import os
import re
import tempfile
import pytest
from utils import *

LORA_FILE_URL = "https://huggingface.co/ggml-org-dev/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

PROMPT = "Once upon a time"


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


# experts on the host, a small MoE cache on the device, LoRA on attn and on the *_exps tensors
def run_server(monkeypatch, stream: bool):
    monkeypatch.setenv("LLAMA_ARG_CPU_MOE", "1")
    monkeypatch.setenv("LLAMA_ARG_MOE_CACHE_MIB", "8")
    monkeypatch.setenv("LLAMA_ARG_FIT", "off")
    if stream:
        monkeypatch.setenv("LLAMA_ARG_GRAVITY_STREAM", "1")
    else:
        monkeypatch.delenv("LLAMA_ARG_GRAVITY_STREAM", raising=False)

    server = ServerPreset.stories15m_moe()
    server.n_gpu_layer = 99
    server.lora_files = [download_file(LORA_FILE_URL)]
    server.debug = True # the cache stats are INFO lines
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)

    contents = {}
    try:
        server.start()
        for scale in (0.0, 1.0):
            res = server.make_request("POST", "/completion", data={
                "prompt": PROMPT,
                "n_predict": 32,
                "temperature": 0.0,
                "seed": 42,
                "n_probs": 4,
                "lora": [{"id": 0, "scale": scale}],
            })
            assert res.status_code == 200
            # the probs are floats: equal text and probs is a near bit-level check
            contents[scale] = (res.body["content"], res.body["completion_probabilities"])
    finally:
        # the cache prints its stats when the context is freed
        server.stop()
    with open(server.log_path, errors="replace") as f:
        log = f.read()
    os.remove(server.log_path)
    return contents, log


def test_gravity_stream_lora_identical(monkeypatch):
    require_gpu()
    off, log_off = run_server(monkeypatch, stream=False)
    on,  log_on  = run_server(monkeypatch, stream=True)

    for scale in (0.0, 1.0):
        assert on[scale] == off[scale], f"scale {scale}: stream on {on[scale]!r} != off {off[scale]!r}"
    assert on[1.0][0] != on[0.0][0], "the adapter did not change the output"

    if "MoE cache size" not in log_off:
        pytest.skip("no MoE cache on this host (no GPU)")

    # the cache must be used with the LoRA on the experts, and the stream must read the misses from the file
    for log in (log_off, log_on):
        m = re.search(r"hits = (\d+), misses = (\d+)", log)
        assert m is not None, "MoE cache had no traffic"
        assert int(m.group(1)) + int(m.group(2)) > 0
    m = re.search(r"gravity stream: read = ([0-9.]+) MiB", log_on)
    assert m is not None and float(m.group(1)) > 0, "gravity stream read nothing"
    assert "gravity stream: read" not in log_off
