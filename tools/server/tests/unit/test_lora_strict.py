import pytest
from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org-dev/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.server_port = 18481  # own port, other suites run in parallel on 8080
    server.lora_files = [download_file(LORA_FILE_URL)]


# one adapter is loaded, so valid ids are only 0
BAD_LORA = [
    pytest.param([{"scale": 1.0}], id="missing-id"),
    pytest.param([{"id": -1, "scale": 1.0}], id="id-minus-1"),
    pytest.param([{"id": 1, "scale": 1.0}], id="id-eq-n-adapters"),
    pytest.param([{"id": "0", "scale": 1.0}], id="id-string"),
    pytest.param([{"id": 0.5, "scale": 1.0}], id="id-fraction"),
    pytest.param([{"id": True, "scale": 1.0}], id="id-bool"),
    pytest.param([{"id": 0}], id="missing-scale"),
    pytest.param([{"id": 0, "scale": "1"}], id="scale-string"),
    pytest.param([{"id": 0, "scale": 1.0}, {"id": 0, "scale": 0.5}], id="duplicate-id"),
    pytest.param([{"id": 0, "scal": 1.0, "scale": 1.0}], id="extra-key"),
    pytest.param(["0"], id="entry-not-object"),
    pytest.param({"id": 0, "scale": 1.0}, id="lora-not-array"),
]

GOOD_LORA = [
    pytest.param([{"id": 0, "scale": 0}], id="scale-int-0"),
    pytest.param([{"id": 0, "scale": 1}], id="scale-int-1"),
    pytest.param([{"id": 0, "scale": 0.0}], id="scale-0"),
    pytest.param([{"id": 0, "scale": 1.0}], id="scale-1"),
    pytest.param([{"id": 0, "scale": 0.5}], id="scale-half"),
    pytest.param([], id="empty"),
    pytest.param([{"id": 0.0, "scale": 1}], id="id-integral-float"),
]


def request_lora(endpoint: str, lora):
    if endpoint == "/v1/chat/completions":
        return server.make_request("POST", endpoint, data={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "lora": lora,
        })
    if endpoint == "/completion":
        return server.make_request("POST", endpoint, data={"prompt": "hi", "n_predict": 2, "lora": lora})
    return server.make_request("POST", endpoint, data=lora)


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/completion", "/lora-adapters"])
@pytest.mark.parametrize("lora", BAD_LORA)
def test_lora_strict_400(endpoint: str, lora):
    server.start()
    res = request_lora(endpoint, lora)
    assert res.status_code == 400
    assert res.body["error"]["type"] == "invalid_request_error"
    assert "lora" in res.body["error"]["message"]


def test_lora_strict_message():
    server.start()
    res = request_lora("/completion", [{"id": 0, "scale": 1.0}, {"id": 3, "scale": 1.0}])
    assert res.status_code == 400
    assert "lora[1]: id 3 out of range (1 adapters loaded)" in res.body["error"]["message"]


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/completion", "/lora-adapters"])
@pytest.mark.parametrize("lora", GOOD_LORA)
def test_lora_strict_200(endpoint: str, lora):
    server.start()
    res = request_lora(endpoint, lora)
    assert res.status_code == 200
