# Upstream

gravity is built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, Copyright (c) 2023-2026 The ggml authors). Its full history is kept in this repository.

## Base

| | |
|---|---|
| Upstream tag | `v1.0.0` (upstream build `b12359`) |
| Upstream commit | `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (2026-11-13) |
| Imported | 2026-11-19 |

## What changed at import

- `.github/workflows/` removed. Upstream's workflows target upstream's runners, release channels and self-hosted machines, and would spend this repository's Actions minutes. gravity's own workflows replace them.
- `UPSTREAM.md` added.
- Later (2026-11-20): upstream's `.github/actions/` removed (used only by its workflows). Kept and adapted: the three bug templates (upstream links and AI-output rule removed, backends trimmed, upstream-check field added) and `.github/labeler.yml` (gravity's backends and area labels, run by `labeler.yml` workflow). Replaced: research/refactor/enhancement templates and the PR template (they pointed at upstream's discussions and guidelines). `SECURITY.md` gained a gravity reporting note at the top.

Nothing else differs from the upstream tag. `LICENSE`, `AUTHORS`, `licenses/` and the licence files under `vendor/` and `gguf-py/` are kept unchanged, and must stay so in every file that comes from upstream.

## Rebase policy

- Move the base only to an upstream `v*` release, never to an hourly `b*` build.
- At most once a month, or sooner for a security fix.
- Each move is its own pull request: rebase gravity's commits onto the new tag, update the Base table above, attach the latest drift report, and pass CI on Linux and macOS before merge.
- gravity's own changes stay as separate commits on top of the base, so each rebase shows exactly what gravity adds.

## Drift report

A weekly workflow reports how far the base is behind upstream's newest `v*` release and `master`, and which upstream changes touch files gravity has changed.

## Telemetry

gravity must not send telemetry. Check each rebase for upstream changes that add any, and remove or disable them.
