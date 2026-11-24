# gravity

**Run an open model with an adapter, and know which adapter, at what strength, on every answer.**

gravity is our fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for running the LoRA adapters that [astro-weights](https://github.com/astro-weights/astro-weights) makes. An adapter is a small course correction to a model, and when you measure one you need to know it was actually on, or actually off, for every request. Stock llama.cpp makes that easy to get wrong.

Two things we hit on stock llama.cpp (build b12359, Qwen3.5-9B Q4_K_M, Apple M3 Pro):
- A request without a `lora` field ran the adapter at full strength, even on a fresh server started with `--lora-init-without-apply` (5 of 5 test prompts).
- With 4 parallel slots, only 3 of 20 answers matched the single-slot run byte for byte. With 1 slot, 20 of 20 matched.

## What gravity adds
- **Pinned files.** `--model-sha256` and `--lora-sha256` check the base model and each adapter before loading. A wrong file stops the server and names both hashes.
- **Eval mode.** `--eval-mode` turns on one slot and no prompt cache, starts every adapter at strength 0, and fills any missing sampling field with a fixed default. On our test model the base arm reran byte-identical, 20 of 20 answers, on a Mac (Metal) and on Linux (CUDA).
- **Strict `lora` lists.** Bad `lora` entries get HTTP 400 instead of being ignored.
- **A receipt on every answer in eval mode.** Each answer reports the strength and sha256 of every adapter, the model's sha256, which fields gravity filled or overrode, and the device that ran it.

## Not built yet
These are on the roadmap, not in the code:
- Running models larger than memory by streaming MoE experts from disk (Linux first, then macOS).
- LoRA adapters on streamed experts.
- One-command install and Windows builds.
- A public page of measurements.

## What gravity is not
- **Not a speed race.** We don't compete on tokens per second. If an adapter result can't be checked, it doesn't count.
- **No telemetry.** gravity sends nothing anywhere by default.
- **Adapters first.** Features that make adapters safer to run and easier to measure come before everything else.

gravity follows upstream llama.cpp releases (currently v1.0.0) and keeps its MIT licence.

[![CI](https://github.com/astro-weights/gravity/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/astro-weights/gravity/actions/workflows/ci.yml?query=branch%3Amain)
[![CUDA build](https://github.com/astro-weights/gravity/actions/workflows/cuda.yml/badge.svg?event=schedule)](https://github.com/astro-weights/gravity/actions/workflows/cuda.yml)
[![macOS release build](https://github.com/astro-weights/gravity/actions/workflows/macos.yml/badge.svg)](https://github.com/astro-weights/gravity/actions/workflows/macos.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Based on llama.cpp v1.0.0, see [UPSTREAM.md](UPSTREAM.md). Upstream README: [README.upstream.md](README.upstream.md).

## Details: what gravity adds over llama.cpp v1.0.0

Everything below is built and tested. Full server docs: [tools/server/README.md](tools/server/README.md).

- **File hash pins** (`--model-sha256`, `--lora-sha256`). The server hashes the files at startup. On a mismatch it prints `sha256 mismatch for ...: expected X, actual Y` and exits before it loads anything. Why: a result is only tied to an adapter if the file is the one you think it is. Not for split models and not in router mode. Without pins the server warns `not pinned` and starts as before.
- **Strict per-request `lora`.** Malformed lists return HTTP 400 (not an array, entry not an object, missing or non-integer `id`, `id` outside the loaded adapters, missing or non-numeric `scale`, unknown key, duplicate `id`). The same check runs on `POST /lora-adapters`. Why: upstream silently ignores a bad entry, so an adapter you thought was on could be off.
- **Eval mode** (`--eval-mode`). One slot, no prompt cache, no continuous batching, every global adapter scale starts at 0. On `/completion`, `/completions`, `/v1/completions` and `/v1/chat/completions`, each absent field is filled with a fixed default (`seed` 0, `temperature` 0, `samplers` `["top_k","temperature"]`, `top_k` 1, `lora` = every adapter at scale 0) and its name is listed in the response. `cache_prompt: true` is forced to `false`. Routes that could run the model without these defaults (for example `POST /lora-adapters`, `/infill`, `/v1/messages`, embeddings) return HTTP 400. Why: both arms of a comparison run under the same fixed settings, and the arm is always explicit.
- **`gravity` provenance object.** In eval mode every response from the four routes above carries `gravity` (in the final chunk when streaming). Fields: `eval_mode`, `build`, `devices_available`, `devices_used`, `device_info`, `layers_on_device`, `output_device`, `model_sha256`, `lora` (`id`, `scale`, `sha256` per adapter), `seed`, `temperature`, `samplers`, `filled`, `overridden`. Outside eval mode responses stay exactly as upstream; `GET /props` always has the object. Why: the answer itself says which files, which strengths and which settings made it.
- **Device info.** `devices_used` (device names the model's layers run on), `layers_on_device` (layer count per device), `output_device`, and `device_info` (ggml description and total memory in MiB per used device, so you can tell which card ran). Why: `devices_available` alone does not show where the model ran.
- **Slot cache clear on global lora change.** A request without `lora` after one with `lora` no longer reuses a prompt cache computed under the other adapter setting. This is upstream's fix (llama.cpp #30367, commit 58a4a6b49), cherry-picked onto v1.0.0.
- **`--lora-init-without-apply` scale fix.** On v1.0.0 the flag left every adapter scale at 1.0. In gravity the global scales really start at 0, so a request without `lora` runs the base model.

## Flags and fields

Only what gravity adds. Everything else is upstream's.

| Flag | Default | What |
|---|---|---|
| `--model-sha256 HEX` | off | sha256 of the model file (64 hex digits). Server refuses to start on a mismatch. Env `LLAMA_ARG_MODEL_SHA256`. |
| `--lora-sha256 HEX` | off | sha256 of one LoRA file. Give once per `--lora` / `--lora-scaled`, same order. Repeatable or comma-separated. |
| `--eval-mode` | off | Reproducible evaluation, see above. Env `LLAMA_ARG_EVAL_MODE`. |
| `--lora-init-without-apply` | off | Load adapters with every global scale at 0. Eval mode turns it on. |

| Response field | Where | What |
|---|---|---|
| `gravity` | eval-mode responses, `GET /props` | Provenance object, fields listed above. |
| `gravity.filled` | eval-mode responses | Request fields gravity filled with a default. |
| `gravity.overridden` | eval-mode responses | Request fields gravity forced (`cache_prompt`). |
| `gravity.devices_used`, `device_info`, `layers_on_device`, `output_device` | eval-mode responses, `GET /props` | Where the model runs. |

## Build

Linux and macOS, from source. Same commands as gravity's CI.

```bash
git clone https://github.com/astro-weights/gravity
cd gravity
```

Linux (CPU):

```bash
cmake -B build -DGGML_NATIVE=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --config Release -j "$(nproc)"
```

macOS (Apple silicon, Metal is on by default; the flag below is what CI sets):

```bash
cmake -B build -DGGML_NATIVE=OFF -DGGML_METAL=ON -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)"
```

The binaries are `build/bin/llama-server` and `build/bin/llama-cli`. For CUDA, Vulkan, HIP and other backends, see [docs/build.md](docs/build.md). `-DLLAMA_USE_PREBUILT_UI=OFF` skips the web UI download, so the server has no web UI.

## Evaluate an adapter with astro-weights

Serve the base GGUF and the adapter GGUF on one server. Pin both files. Eval mode forces one slot (`-np 1`), which is what gives byte-identical reruns.

```bash
astro-weights export ADAPTER_DIR --gguf adapter.gguf --arch qwen35   # prints the adapter sha256
shasum -a 256 base.gguf                                              # base sha256
build/bin/llama-server -m base.gguf --lora adapter.gguf \
  --model-sha256 BASE_SHA256 --lora-sha256 ADAPTER_SHA256 \
  --eval-mode -np 1 -ngl 99 --port 8080
```

Then run both arms with astro-weights' HTTP runner. The arm is the `lora` scale in `extra_body`. `arm_keys=["lora"]` tells the header check that the arms may differ only there. Declare the server flags in `serving_flags`, because the client cannot read them.

```python
from astro_weights.eval import check_headers, grade_exact, header_path, write_header
from astro_weights.runners.endpoint import generate_records_http, run_header_http

URL, NAME = "http://127.0.0.1:8080", "base.gguf"
FLAGS = {"--eval-mode": "on", "--parallel": "1"}
items = [{"item_id": "i0", "prompt": "Say w5.", "gold": "w5"}]
headers = {}
for arm, scale in [("base", 0), ("edit", 1)]:
    extra_body = {"lora": [{"id": 0, "scale": scale}]}  # explicit, base arm too
    headers[arm] = run_header_http(URL, NAME, 64, 0, extra_body, serving_flags=FLAGS)
    write_header(header_path(f"{arm}-http.jsonl"), headers[arm])
    generate_records_http(
        URL, NAME, items, arm, grade_exact, 64, extra_body=extra_body, out=f"{arm}-http.jsonl"
    )
check_headers(headers["base"], headers["edit"], arm_keys=["lora"])
```

Then `astro-weights eval base-http.jsonl edit-http.jsonl --ignore-header`. Each response also has `gravity.lora[].scale` and the sha256 values, so you can check after the fact that the arms ran as intended. See the astro-weights docs (`docs/how-to/run-eval.md`) for the rest of the runner.

## Measured

From GR1, our noise-floor test. Model: Qwen3.5-9B Q4_K_M. Machine: Apple M3 Pro (Metal). Base arm only, 20 MATH-500 confirm items, card sampling with seed 0, token cap 1024. This tests determinism, not accuracy.

| Server | Comparison | Byte-identical answers |
|---|---|---|
| stock llama.cpp b12359, 1 slot | run 1 vs run 2 | 20 of 20 |
| stock llama.cpp b12359, 4 slots | vs the 1-slot run | 3 of 20 |
| gravity `--eval-mode` (1 slot), Mac (Metal) | base arm rerun | 20 of 20 |
| gravity `--eval-mode` (1 slot), Linux (CUDA) | base arm rerun | 20 of 20 |

In the same test, a fresh stock server with `--lora-init-without-apply` ran the adapter at scale 1.0 for a request without `lora` (5 of 5 prompts).

## Upstream base and rebase policy

- Base: llama.cpp `v1.0.0`, commit `b1b770d90d94`.
- Move the base only to an upstream `v*` release, at most once a month, each move its own pull request.
- gravity's changes stay as separate commits on top of the base.

Details: [UPSTREAM.md](UPSTREAM.md).

## Licence

MIT. Copyright (c) 2023-2026 The ggml authors. See [LICENSE](LICENSE). gravity is built on llama.cpp by the ggml authors.

Dependencies:

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
