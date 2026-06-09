# HiCache One-in-All Benchmark Harness

`bench_all_in_one.py` can be used as the evaluation harness for an
AlphaEvolve/AIScientist-style loop that mutates KV-cache manager code, launches
an SGLang server, and asks for one final fitness score plus detailed feedback.

The harness does not launch the server. Start the candidate server first, then
run this benchmark against it.

## Server Setup

For KV-cache architecture search with GPU + host RAM + file/SSD L3 HiCache (all in one),
use a server command that exposes metrics and enables the file storage backend:

```bash
export SGLANG_ADMIN_API_KEY=1234567890
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/scratch/ows/alphacache/.hicache-file
mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --context-length 32768 \
  --mem-fraction-static 0.70 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --enable-metrics \
  --enable-mfu-metrics \
  --enable-cache-report \
  --enable-request-time-stats-logging \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy best_effort \
  --admin-api-key "$SGLANG_ADMIN_API_KEY"
```

`--hicache-ratio 2` sizes the host RAM HiCache pool as 2x the GPU KV cache
pool, not 2x system RAM. If the log says something like `Allocating 62.2 GB
host memory for hierarchical KV cache` on a 1 TB machine, that is expected when
the GPU KV pool is about 31 GB. To use a larger L2 host-memory cache, set
`--hicache-size <GB>`; this overrides `--hicache-ratio` and is applied per rank.
For example, `--hicache-size 64` with `--tp-size 8` can allocate roughly 512 GB
total host KV memory. Leave enough RAM for the OS, dataset loading, page cache,
and other jobs.



## Basic Evaluation

Run one benchmark session per candidate:

```bash
cd /scratch/ows/alphacache/extern/sglang/benchmark/hicache

python3 bench_all_in_one.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --candidate-id my_candidate_001 \
  --session-name my_candidate_001
```

For a few-minute feature check, use fast mode:

```bash
python3 bench_all_in_one.py \
  --fast \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --candidate-id smoke_001 \
  --session-name smoke_001
```

Fast mode uses one request rate, small prompt/client counts, shorter synthetic
requests, and a short warm-cache sweep. It still exercises LooGLE serving,
ShareGPT-style multi-round traffic, warm-cache behavior, and shared-prefix
serving.

To include L3 behavior in the fast check:

```bash
python3 bench_all_in_one.py \
  --fast \
  --l3-cache \
  --l3-replay \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

`--l3-replay` duplicates replayable workloads into a populate pass and a replay
pass. The runner flushes memory cache between runs but keeps L3 storage intact,
so the replay pass can expose file/Mooncake/HF3FS load-back behavior.

To use only part of the data outside fast mode:

```bash
# Scale all workload sample counts to 10% of their current settings.
python3 bench_all_in_one.py --data-fraction 0.1

# Or cap each workload family to at most 8 samples/clients/prompts.
python3 bench_all_in_one.py --sample-count 8
```

Dataset-backed serving workloads shuffle by default, so `--data-fraction` and
`--sample-count` act like a random subset unless `--serving-disable-shuffle` is
also passed.

By default, results are written under:

```text
bench_all_results/<session-name>/
```

Important files:

- `fitness.txt`: one numeric final score. Higher is better.
- `fitness_summary.json`: machine-readable session summary.
- `feedback.md`: human-readable diagnosis.
- `runs.jsonl`: one full record per workload run.
- `<run-name>/feedback_metrics.json`: compact metrics used by the scorer.
- `<run-name>/run_record.json`: full per-run record.
- `<run-name>/before_metrics.prom` and `after_metrics.prom`: raw Prometheus snapshots.
- `<run-name>/stdout.log` and `stderr.log`: child benchmark logs.
- `l3_config.json`: L3 setup/status when `--l3-cache` is enabled.

## Fitness Function

The easiest editable path is a Python file with `score(metrics)`:

```bash
python3 bench_all_in_one.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --fitness-file fitness_template.py \
  --candidate-id my_candidate_001
```

`metrics` is a `FitnessMetrics` object. It supports both dot access and
`metrics.get(name, default)`.

Common aliases:

- `ttft_ms`
- `e2e_ms`
- `itl_ms`
- `throughput_req_s`
- `throughput_out_tok_s`
- `cache_hit_rate`
- `gpu_kv_used_tokens`
- `gpu_kv_evictable_tokens`
- `gpu_kv_available_tokens`
- `host_kv_used_tokens`
- `host_kv_total_tokens`
- `cached_device_tokens`
- `cached_host_tokens`
- `cached_storage_file_tokens`
- `prefetched_tokens`
- `backuped_tokens`
- `evicted_tokens`
- `load_back_tokens`
- `cached_storage_mooncake_tokens`
- `cached_storage_hf3fs_tokens`
- `aborted_requests`

Prometheus names are also available through `metrics.get(...)`, for example:

```python
metrics.get("prom.delta.sglang:time_to_first_token_seconds_avg")
metrics.get("prom.after.sglang:hicache_host_used_tokens")
```

For quick experiments, a command-line expression also works:

```bash
python3 bench_all_in_one.py \
  --fitness-expr "1000*cache_hit_rate - 0.1*ttft_ms + 0.00002*cached_host_tokens"
```

The final session score combines per-run scores with `--fitness-aggregate`
(`mean`, `min`, `max`, or `sum`).

## Feedback Signals

The harness records both child benchmark metrics and SGLang server metrics.
Useful signals for KV-cache manager search include:

- Latency: TTFT, ITL, E2E request latency.
- Online serving throughput: request/s and output token/s.
- GPU cache pressure: KV used, evictable, and available tokens.
- Host HiCache pressure: host used/total tokens.
- Cache reuse source: device hits, host hits, file/mooncake/hf3fs storage hits.
- Storage movement: prefetched, backed up, evicted, and loaded-back tokens.
- Reliability: aborted requests and failed benchmark runs.

`feedback.md` gives a compact human summary. `fitness_summary.json` and each
`feedback_metrics.json` are the best inputs for an automated reflection loop.

## Profiling

SGLang exposes `/start_profile` and `/stop_profile`. The harness can wrap each
workload with those endpoints:

```bash
python3 bench_all_in_one.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --profile \
  --profile-start-step 5 \
  --profile-num-steps 10 \
  --profile-activities CPU,GPU \
  --profile-by-stage \
  --profile-merge
```

Trace output goes under `<run-name>/profile` by default. If
`--profile-output-dir /path/to/profiles` is set, each run writes to:

```text
/path/to/profiles/<run-name>/
```

Additional profiling controls:

- `--profile-stages prefill,decode` limits stage profiling to selected stages.
- `--profile-with-stack` / `--no-profile-with-stack` controls stack capture.
- `--profile-record-shapes` / `--no-profile-record-shapes` controls tensor shape capture.
- `--profile-activities CPU,GPU,MEM` can include memory profiling if supported.

Profiling is expensive. Use it for a small number of candidate runs after the
fitness signal shows something interesting.

## Suggested Loop

1. Mutate or generate a KV-cache manager change.
2. Build/install SGLang for that candidate.
3. Launch the server with metrics enabled.
4. Run `bench_all_in_one.py --candidate-id <id> --session-name <id>`.
5. Read `fitness.txt` for selection.
6. Read `fitness_summary.json` and `feedback.md` for reflection.
7. Preserve the full `bench_all_results/<id>/` folder with the candidate patch.

Keep the workload fixed while comparing candidates. Change the fitness function
only between search campaigns, not halfway through a population, unless the
optimizer is explicitly aware of the objective change.
