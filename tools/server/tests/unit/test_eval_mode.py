import json

import pytest
import requests
from utils import *
from test_hash_pins import ReadyServer, resolve_model_path, sha256_of, LORA_FILE_URL

MODEL_ARGS = ["--offline", "--hf-repo", "ggml-org-dev/stories15M_MOE", "--hf-file", "stories15M_MOE-F16.gguf"]
PROMPT = "Look in thy glass"
BASE = {"prompt": PROMPT, "n_predict": 16}
ALL = {"prompt": PROMPT, "n_predict": 16, "seed": 1, "temperature": 0, "samplers": ["top_k", "temperature"],
       "top_k": 1, "lora": [{"id": 0, "scale": 0}]}
DEFAULT_FILLED = {"seed", "temperature", "samplers", "top_k", "lora"}


@pytest.fixture(scope="module")
def lora_file():
    return download_file(LORA_FILE_URL)


@pytest.fixture
def start(tmp_path, lora_file):
    servers: list[ReadyServer] = []

    def _start(*extra: str) -> ReadyServer:
        s = ReadyServer([*MODEL_ARGS, "--lora", lora_file, *extra], str(tmp_path / f"server{len(servers)}.log"))
        servers.append(s)
        return s

    yield _start
    for s in servers:
        s.stop()


def post(s: ReadyServer, path: str, body: dict):
    return requests.post(s.url + path, json=body, timeout=DEFAULT_REQUEST_TIMEOUT)


def text(res) -> str:
    assert res.status_code == 200, res.text
    return res.json()["content"]


def test_flag_fix_without_eval_mode(start):
    s = start("--lora-init-without-apply")
    assert s.get("/lora-adapters").json()[0]["scale"] == 0
    # first request of a fresh server, no `lora`: must be the base model
    no_lora = text(post(s, "/completion", {**BASE, "seed": 42, "temperature": 0}))
    base = text(post(s, "/completion", {**BASE, "seed": 42, "temperature": 0, "lora": [{"id": 0, "scale": 0}]}))
    edit = text(post(s, "/completion", {**BASE, "seed": 42, "temperature": 0, "lora": [{"id": 0, "scale": 1}]}))
    assert no_lora == base
    assert no_lora != edit


def test_no_flag_keeps_scale_1(start):
    s = start()
    assert s.get("/lora-adapters").json()[0]["scale"] == 1
    # outside eval mode the response is upstream's: no gravity object; /props still has it
    assert "gravity" not in post(s, "/completion", BASE).json()
    assert s.get("/props").json()["gravity"]["eval_mode"] is False


def test_eval_mode_object_survives_response_fields(start):
    s = start("--eval-mode")
    res = post(s, "/completion", {**BASE, "response_fields": ["content"]}).json()
    assert set(res) == {"content", "gravity"}
    assert set(res["gravity"]["filled"]) == DEFAULT_FILLED


def test_eval_mode_fresh_server(start):
    s = start("--eval-mode", "-np", "4", "--cache-reuse", "256")
    props = s.get("/props").json()
    assert props["total_slots"] == 1
    assert props["gravity"]["eval_mode"] is True
    assert props["gravity"]["lora"] == [{"id": 0, "scale": 0.0, "sha256": None}]
    assert s.get("/lora-adapters").json()[0]["scale"] == 0
    assert "overrides" in s.log()

    no_lora = text(post(s, "/completion", BASE))  # first request, nothing sent but the prompt
    base = text(post(s, "/completion", {**BASE, "seed": 0, "temperature": 0, "lora": [{"id": 0, "scale": 0}]}))
    edit = text(post(s, "/completion", {**BASE, "seed": 0, "temperature": 0, "lora": [{"id": 0, "scale": 1}]}))
    assert no_lora == base
    assert no_lora != edit


def test_eval_mode_filled_and_echoed(start):
    s = start("--eval-mode")
    g = post(s, "/completion", BASE).json()["gravity"]
    assert set(g["filled"]) == DEFAULT_FILLED
    assert g["overridden"] == []
    assert (g["seed"], g["temperature"], g["samplers"]) == (0, 0, ["top_k", "temperature"])
    assert g["lora"] == [{"id": 0, "scale": 0.0, "sha256": None}]
    assert g["version"] == "0.1.0"
    assert g["eval_mode"] is True and g["build"] and g["devices_available"] and g["model_sha256"] is None

    g = post(s, "/completion", ALL).json()["gravity"]
    assert g["filled"] == [] and g["overridden"] == []
    assert g["seed"] == 1  # given fields are never changed

    g = post(s, "/completion", {**BASE, "temperature": 0.5, "cache_prompt": True}).json()["gravity"]
    assert g["overridden"] == ["cache_prompt"]
    assert "temperature" not in g["filled"] and g["temperature"] == 0.5

    res = post(s, "/completion", {**BASE, "cache_prompt": None})
    assert res.status_code == 200 and res.json()["gravity"]["overridden"] == []
    assert post(s, "/completion", {**BASE, "cache_prompt": "yes"}).status_code == 400

    g = post(s, "/completion", {**BASE, "lora": []}).json()["gravity"]
    assert "lora" in g["filled"] and g["lora"][0]["scale"] == 0

    g = post(s, "/completion", {**BASE, "lora": [{"id": 0, "scale": 1}]}).json()["gravity"]
    assert "lora" not in g["filled"] and g["lora"][0]["scale"] == 1


def test_eval_mode_repeatable_and_strict(start):
    s = start("--eval-mode")
    assert text(post(s, "/completion", BASE)) == text(post(s, "/completion", BASE))
    res = post(s, "/completion", {**BASE, "lora": [{"id": 3, "scale": 1}]})
    assert res.status_code == 400 and "lora" in res.json()["error"]["message"]


def test_eval_mode_endpoints(start):
    s = start("--eval-mode")
    chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    res = post(s, "/v1/chat/completions", chat)
    assert res.status_code == 200
    assert set(res.json()["gravity"]["filled"]) == DEFAULT_FILLED
    res = post(s, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4})
    assert set(res.json()["gravity"]["filled"]) == DEFAULT_FILLED

    for path, body in [("/completion", {**BASE, "stream": True}), ("/v1/chat/completions", {**chat, "stream": True})]:
        chunks = []
        with requests.post(s.url + path, json=body, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT) as r:
            for line in r.iter_lines():
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    chunks.append(json.loads(line[6:]))
        assert "gravity" in chunks[-1], path
        assert all("gravity" not in c for c in chunks[:-1]), path
        assert set(chunks[-1]["gravity"]["filled"]) == DEFAULT_FILLED


def test_model_sha256_echo(start):
    model = resolve_model_path(ServerPreset.stories15m_moe())
    pin = sha256_of(model)
    s = start("--eval-mode", "--model-sha256", pin)
    assert post(s, "/completion", BASE).json()["gravity"]["model_sha256"] == pin
    assert s.get("/props").json()["gravity"]["model_sha256"] == pin
    s2 = start("--eval-mode")
    assert post(s2, "/completion", BASE).json()["gravity"]["model_sha256"] is None


def check_devices(g: dict, n_layer: int):
    assert sum(g["layers_on_device"].values()) == n_layer
    assert set(g["layers_on_device"]) <= set(g["devices_used"])
    assert g["output_device"] in g["devices_used"]
    assert g["devices_used"][0] in g["layers_on_device"]
    assert [d["name"] for d in g["device_info"]] == g["devices_used"]
    for d in g["device_info"]:
        assert d["description"]
        assert d["memory_total_mib"] > 0


def test_devices_used_cpu(start):
    s = start("--eval-mode", "-ngl", "0")
    n_layer = 6  # stories15M_MOE
    for g in (post(s, "/completion", BASE).json()["gravity"], s.get("/props").json()["gravity"]):
        assert g["devices_used"] == ["CPU"]
        assert g["layers_on_device"] == {"CPU": n_layer}
        check_devices(g, n_layer)


def test_devices_used_gpu(start):
    s = start("--eval-mode", "-ngl", "99")
    props = s.get("/props").json()["gravity"]
    gpu = [d for d in props["devices_available"] if d != "CPU"]
    if not gpu:
        pytest.skip("no GPU device registered")
    for g in (post(s, "/completion", BASE).json()["gravity"], props):
        assert set(g["devices_used"]) & set(gpu)
        check_devices(g, sum(g["layers_on_device"].values()))


def test_devices_used_props_without_eval_mode(start):
    s = start("-ngl", "0")
    assert s.get("/props").json()["gravity"]["devices_used"] == ["CPU"]
    assert "gravity" not in post(s, "/completion", BASE).json()


REFUSED = [
    "/lora-adapters", "/slots/0?action=restore", "/infill", "/v1/responses", "/v1/messages",
    "/embedding", "/v1/embeddings", "/rerank", "/v1/systemone", "/v1/audio/transcriptions", "/tts", "/v1/audio/speech",
]


def test_eval_mode_refuses_unechoed_routes(start):
    s = start("--eval-mode")
    for path in REFUSED:
        res = post(s, path, {"input": "hi", "prompt": "hi", "messages": [{"role": "user", "content": "hi"}]})
        assert res.status_code == 400, path
        assert "--eval-mode" in res.json()["error"]["message"], path
    assert s.get("/lora-adapters").json()[0]["scale"] == 0  # GET still answers
    assert post(s, "/tokenize", {"content": "hi"}).status_code == 200


def test_routes_open_without_eval_mode(start):
    s = start()
    assert post(s, "/lora-adapters", [{"id": 0, "scale": 0}]).status_code == 200
