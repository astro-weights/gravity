import hashlib
import subprocess
import time

import requests

import pytest
from utils import *

# the line server_context::load_model() logs first; a refused pin must never get there
LOAD_MARKER = "loading model '"

LORA_FILE_URL = "https://huggingface.co/ggml-org-dev/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

ZERO_HASH = "0" * 64


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_path(preset: ServerProcess) -> str:
    # the path the server actually loads (-hf resolves into the cache)
    preset.start()
    res = preset.make_request("GET", "/props")
    assert res.status_code == 200
    path = res.body["model_path"]
    preset.stop()
    assert os.path.isfile(path)
    return path


@pytest.fixture(scope="module")
def model_path():
    return resolve_model_path(ServerPreset.tinyllama2())


@pytest.fixture(scope="module")
def moe_paths():
    lora = download_file(LORA_FILE_URL)
    return resolve_model_path(ServerPreset.stories15m_moe()), lora


def server_bin() -> str:
    return os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")


def run_failing(args: list[str], timeout: float = 60) -> tuple[int, str]:
    """run llama-server, expect it to exit by itself; returns (exit code, output)"""
    port = int(os.environ.get("PORT", "8080"))
    env = {**os.environ, "LLAMA_CACHE": os.environ.get("LLAMA_CACHE", "tmp")}
    cmd = [server_bin(), "--port", str(port), "--host", "127.0.0.1", *args]
    print(f"tests: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    out = proc.stdout.decode("utf-8", errors="replace")
    print(out)
    return proc.returncode, out


class ReadyServer:
    """llama-server started with the given args and waited for until /health is ok; keeps its output"""

    def __init__(self, args: list[str], log_path: str, timeout: float = DEFAULT_HTTP_TIMEOUT):
        port = int(os.environ.get("PORT", "8080"))
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        env = {**os.environ, "LLAMA_CACHE": os.environ.get("LLAMA_CACHE", "tmp")}
        cmd = [server_bin(), "--port", str(port), "--host", "127.0.0.1", "-c", "512", "-np", "1", *args]
        print(f"tests: starting {' '.join(cmd)}")
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}:\n{self.log()}")
            try:
                if requests.get(self.url + "/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.05)
        self.stop()
        raise TimeoutError("server did not become ready")

    def get(self, path: str):
        return requests.get(self.url + path, timeout=DEFAULT_REQUEST_TIMEOUT)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if not self._log.closed:
            self._log.close()

    def log(self) -> str:
        if not self._log.closed:
            self._log.flush()
        with open(self.log_path, "r", errors="replace") as f:
            return f.read()


@pytest.fixture
def ready_server(tmp_path):
    servers: list[ReadyServer] = []

    def start(args: list[str]) -> ReadyServer:
        s = ReadyServer(args, str(tmp_path / f"server{len(servers)}.log"))
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.stop()


def test_model_pin_correct(ready_server, model_path):
    server = ready_server(["--offline", "-m", model_path, "--model-sha256", sha256_of(model_path)])
    assert server.get("/health").status_code == 200
    server.stop()
    log = server.log()
    assert LOAD_MARKER in log
    assert "sha256 of 1 pinned file(s) ok" in log
    assert "model not pinned" not in log


def test_model_pin_upper_case(ready_server, model_path):
    server = ready_server(["--offline", "-m", model_path, "--model-sha256", sha256_of(model_path).upper()])
    assert server.get("/health").status_code == 200
    assert "sha256 of 1 pinned file(s) ok" in server.log()


def test_model_pin_from_env(model_path):
    os.environ["LLAMA_ARG_MODEL_SHA256"] = ZERO_HASH
    try:
        code, out = run_failing(["--offline", "-m", model_path])
    finally:
        del os.environ["LLAMA_ARG_MODEL_SHA256"]
    assert code != 0
    assert "sha256 mismatch" in out


def test_model_pin_wrong(model_path):
    actual = sha256_of(model_path)
    wrong = actual[:-1] + ("0" if actual[-1] != "0" else "1")
    code, out = run_failing(["--offline", "-m", model_path, "--model-sha256", wrong])
    assert code != 0
    assert "sha256 mismatch" in out
    assert model_path in out
    assert f"expected {wrong}" in out
    assert f"actual {actual}" in out
    assert LOAD_MARKER not in out


@pytest.mark.parametrize("value", [
    "0" * 63,
    "0" * 65,
    "g" + "0" * 63,
    "",
])
def test_model_pin_bad_value(model_path, value):
    code, out = run_failing(["--offline", "-m", model_path, "--model-sha256", value])
    assert code != 0
    assert "invalid sha256" in out
    assert LOAD_MARKER not in out


def test_model_pin_split_model(tmp_path, model_path):
    # the check is on the path: the loader only follows the other shards for a "-0000k-of-0000n.gguf" name
    split = tmp_path / "model-00001-of-00002.gguf"
    split.write_bytes(open(model_path, "rb").read())
    code, out = run_failing(["--offline", "-m", str(split), "--model-sha256", sha256_of(str(split))])
    assert code != 0
    assert "not supported for split models" in out
    assert LOAD_MARKER not in out


def test_pin_router_mode():
    code, out = run_failing(["--offline", "--model-sha256", ZERO_HASH])
    assert code != 0
    assert "not supported in router mode" in out


def test_lora_pin_count_mismatch(moe_paths):
    model, lora = moe_paths
    pin = sha256_of(lora)
    code, out = run_failing(["--offline", "-m", model, "--lora", lora, "--lora-sha256", pin, "--lora-sha256", pin])
    assert code != 0
    assert "got 2 --lora-sha256 values for 1 LoRA adapters" in out
    assert LOAD_MARKER not in out


def test_lora_pin_without_lora(model_path):
    code, out = run_failing(["--offline", "-m", model_path, "--lora-sha256", ZERO_HASH])
    assert code != 0
    assert "got 1 --lora-sha256 values for 0 LoRA adapters" in out
    assert LOAD_MARKER not in out


def test_lora_pin_correct(ready_server, moe_paths):
    model, lora = moe_paths
    server = ready_server([
        "--offline", "-m", model, "--model-sha256", sha256_of(model),
        "--lora", lora, "--lora-sha256", sha256_of(lora),
    ])
    assert server.get("/health").status_code == 200
    res = server.get("/lora-adapters")
    assert res.status_code == 200
    assert len(res.json()) == 1
    server.stop()
    log = server.log()
    assert "sha256 of 2 pinned file(s) ok" in log
    assert "not pinned" not in log


def test_lora_pin_wrong(moe_paths):
    model, lora = moe_paths
    actual = sha256_of(lora)
    code, out = run_failing(["--offline", "-m", model, "--lora", lora, "--lora-sha256", ZERO_HASH])
    assert code != 0
    assert "sha256 mismatch for LoRA adapter" in out
    assert lora in out
    assert f"expected {ZERO_HASH}" in out
    assert f"actual {actual}" in out
    assert LOAD_MARKER not in out


def test_unpinned_warns(ready_server, moe_paths):
    model, lora = moe_paths
    server = ready_server(["--offline", "-m", model, "--lora", lora])
    assert server.get("/health").status_code == 200
    server.stop()
    log = server.log()
    assert "model not pinned (--model-sha256)" in log
    assert "LoRA adapter not pinned (--lora-sha256)" in log
