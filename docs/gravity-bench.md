# gravity-bench

`scripts/gravity-bench.py` runs the same GGUF and prompt through three setups, with a cold and a warm page cache, and writes one JSON file. Python 3 standard library only; Linux and macOS.

```bash
python3 scripts/gravity-bench.py --bin build/bin/llama-completion -m MODEL.gguf --cache-mib N \
    [--reps 3] [--n-predict 128] [--prompt TEXT | --prompt-file F] [--arms stock,cache,stream] \
    [--cache-modes cold,warm] [--threads T] [--gravity-stream-threads K] [--extra "ARGS"] \
    [--mac-purge] [--model-sha256] [--timeout S] [--keep-stderr] --out bench.json
python3 scripts/gravity-bench.py --self-test    # parser and argv builder, no binary needed
```

`--cache-mib` is the `--moe-cache-mib` of the `cache` and `stream` arms. Pick a size below the free device memory after the dense weights and KV cache; a size smaller than one expert layer fails at startup. The exit code is nonzero if any run failed; the JSON is still written with the runs that happened.

## Arms

All arms: `--seed 1 --temp 0 -no-cnv -ngl 99 -cmoe -fit off -n N -lv 4`, plus `-t` and `--extra` if given. `-lv 4` is needed because the `llama_moe_cache` lines are debug level; every arm gets it.

| Arm | Adds | Experts |
|---|---|---|
| `stock` | nothing | host mmap, no cache |
| `cache` | `--moe-cache-mib N` | MoE cache, misses uploaded from host memory |
| `stream` | `--moe-cache-mib N --gravity-stream` (`--gravity-stream-threads K`) | MoE cache, misses read from the model file |

Within each rep the arms run in turn (rep 1: stock, cache, stream; rep 2: stock, cache, stream; ...) so machine drift spreads over all arms.

## Cache modes

- **cold:** before each timed run the model file is dropped from the page cache with `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` (Linux, no root). `Cached` in `/proc/meminfo` is recorded before the drop, after it and after the run.
- **warm:** one untimed run of the same arm first, then the timed run.

Limits:

- `DONTNEED` drops clean pages that no process has mapped. Pages another process maps, or dirty pages, stay. Check `meminfo.cached_kib_before_drop` against `cached_kib_after_drop`.
- Only the file given to `-m` is dropped. For a split GGUF the other shards stay cached.
- macOS has no `posix_fadvise`. The default there is warm only, noted in `header.notes`. `--mac-purge` runs `sudo -n purge` before each cold run (needs passwordless sudo); if it fails the cold runs are recorded as `skipped` with `cold unavailable`. `purge` drops the whole system cache.
- Disk-level caches (the SSD controller, a RAID cache) are not touched.

## Per-run metrics

- `exit_code`, `wall_s`, `argv`, `timed_out`.
- From stderr `common_perf_print`: `load_ms`, `prompt_eval_ms`, `prompt_tokens`, `prompt_tok_s`, `eval_ms`, `eval_runs`, `eval_tok_s`.
- `ttft_ms = prompt_eval_ms + eval_ms / eval_runs`: the prompt evaluation plus the mean time of one generated token, standing in for the first one. Model load time is not included. It is derived, not measured per token.
- From `wait4` rusage: `max_rss_bytes` (Linux KiB and macOS bytes both normalized to bytes), `majflt`, `minflt`, `bytes_read_blocks = ru_inblock * 512`.
- Linux: `proc_io_read_bytes`, the last `read_bytes` of `/proc/PID/io` seen while polling every 100 ms (null on macOS or if the process was too short to poll).
- `moe_cache.small` / `moe_cache.large` (`ubatch <= 8` / `> 8`): `hits`, `misses`, `cpu_misses`, `hit_rate_pct`, `uploaded_mib`.
- `stream` (stream arm): `read_mib`, `read_ms`, `fallbacks`, `fallback_mib`. Fallbacks above 0 mean some experts came from host memory instead of the file.
- `stdout_sha256`: hash of the generated text. With the same seed and greedy sampling the `stream` and `cache` hashes should match; the summary prints `stream text == cache text` per cache mode.
- Noise: `load_before`, `load_after` (`os.getloadavg()`) and `top_procs_before` (top 5 other processes by `ps` `pcpu`, which is a lifetime average, not instantaneous). Check these before trusting a number.
- `-lv 4` makes stderr bigger than the default, so stderr is kept only for failed runs (or all runs with `--keep-stderr`, as `<out dir>/<arm>-<mode>-<rep>.stderr.txt`).

## Output schema

```
{
  "header": { host{platform,machine,cpu_count,mem_total_kib}, gravity_git_sha, bin,
             model{path,size_bytes,mtime_utc[,sha256]}, args{...all CLI args}, notes[],
             start_utc, end_utc },
  "runs": [ { arm, cache_mode, rep, argv, warmup_exit_code (warm), cold_method + meminfo (cold),
              skipped + skip_reason (cold unavailable), load_before, load_after, top_procs_before[{pcpu,comm}],
              exit_code, timed_out, wall_s, max_rss_bytes, majflt, minflt, bytes_read_blocks, proc_io_read_bytes,
              stdout_sha256, stdout_bytes, load_ms, prompt_eval_ms, prompt_tokens, prompt_tok_s,
              eval_ms, eval_runs, eval_tok_s, ttft_ms, moe_cache{small,large}, stream{...}|null,
              stderr_file | stderr_tail (failed runs) }, ... ],
  "summary": [ { arm, cache_mode, n, median_eval_tok_s, median_ttft_ms, median_max_rss_mib,
                 median_majflt, median_bytes_read, stream_read_mib_median, stream_fallbacks_total }, ... ],
  "text_identity": { "cold": bool, "warm": bool }   // stream text == cache text
}
```

Summary medians use only runs that exited 0. The model is not hashed unless `--model-sha256` is given (slow for large files).
