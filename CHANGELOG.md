# Changelog

All notable changes to gravity are listed here. gravity is a fork of llama.cpp; this file lists only what gravity adds or changes. Upstream changes are in upstream's release notes. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/); while the version is 0.x, any release may break things.

## [0.1.0] - unreleased

Based on llama.cpp v1.0.0 (`b1b770d90`).

### Added

- `--model-sha256` and `--lora-sha256` check the model and each adapter file at startup; a mismatch stops the server before it loads anything.
- `--eval-mode`: one slot, no prompt cache, every adapter at scale 0 at start, fixed defaults for absent sampling fields, refusal of routes that could run without them.
- The `gravity` provenance object in eval-mode responses and `GET /props`: gravity `version`, upstream `build`, devices available and used, `device_info`, `layers_on_device`, `output_device`, model and adapter sha256, adapter scales, seed, temperature, samplers, and which fields were filled or overridden.
- Experimental, off by default: `--gravity-stream` (with `--moe-cache-mib`) reads MoE cache misses from the model file, output bit-identical to the cache without it; `--gravity-stream-threads` sets the read threads.
- Expert LoRA tensors of cached experts are copied to the cache device once instead of every token; an adapter on streamed experts gives bit-identical output with streaming on or off (server test).
- Prototype, off by default: `--gravity-residency-mib` drops cold expert pages of the mmap after each decode. It turns itself off with the MoE cache, under GPU-wrapped experts (full Metal offload) and where the OS cannot drop pages.
- `llama-gravity-repack` rewrites a MoE GGUF with 4096-byte alignment and a per-expert offset index; stock llama.cpp still loads the result.
- `scripts/gravity-bench.py` compares stock mmap, MoE cache and `--gravity-stream`, cold and warm, and writes JSON.

### Changed

- A bad per-request `lora` list (or `POST /lora-adapters` body) returns HTTP 400 instead of being partly ignored.

### Fixed

- `--lora-init-without-apply` now starts every global adapter scale at 0 (v1.0.0 left them at 1.0).
- A request without `lora` after one with `lora` no longer reuses a prompt cache computed under the other adapter setting (upstream #30367, cherry-picked).
- The GGUF writer, `gguf-split` and `llama-quantize` keep a file's `general.alignment`.
