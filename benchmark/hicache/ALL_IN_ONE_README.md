# HiCache All-in-One Benchmark Harness

`bench_all_in_one.py` can be used as the evaluation harness for an
AlphaEvolve/AIScientist-style loop that mutates KV-cache manager code, launches
an SGLang server, and asks for one final fitness score plus detailed feedback.

The harness does not launch the server. Start the candidate server first, then
run this benchmark against it.

## Dataset Preparation

Run the downloader from this folder. By default it prepares all datasets under
`./data`, skips files that already validate, and retries partially downloaded
files through `.part` files or incomplete checkout recovery.

```bash
cd /scratch/ows/alphacache/extern/sglang/benchmark/hicache

./download_all_in_one.sh
```

Expected layout:

```text
data/
  LooGLE/data/longdep_qa.jsonl
  ShareGPT_V3_unfiltered_cleaned_split.json
  narrativeqa_long_context.json
  reviewmt_sharegpt.json
```

Use another location with:

```bash
./download_all_in_one.sh --data-dir /scratch/ows/alphacache/hicache-data
```

To download only some datasets:

```bash
./download_all_in_one.sh --datasets loogle,sharegpt
```

Notes:

- LooGLE uses Git LFS. If Hugging Face returns an auth error, run
  `hf auth login --force --add-to-git-credential`, then rerun the script.
- NarrativeQA conversion uses the Python `datasets` package. If missing, install
  it in the active benchmark env with `uv pip install datasets`.
- ReviewMT is discovered from the upstream GitHub releases and converted to a
  ShareGPT-style JSON file for the multi-round workload.
- `benchmark/hicache/data/` is ignored by git.

## Server Setup

For Strata-style long-document evaluation, use a model/server configuration
whose accepted input length covers the dataset. A 32K server is useful for
debugging, but it is not a faithful Strata long-context setup for NarrativeQA
or the longest LooGLE samples.

For KV-cache architecture search with GPU + host RAM + file/SSD L3 HiCache
(all in one), use a server command that exposes metrics and enables the file
storage backend. The model/backend must be compatible with radix cache, because
HiCache depends on it.

```bash
export SGLANG_ADMIN_API_KEY=1234567890
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/scratch/ows/alphacache/.hicache-file
mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --tp-size 8 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --enable-metrics \
  --enable-mfu-metrics \
  --enable-cache-report \
  --enable-request-time-stats-logging \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy best_effort \
  --admin-api-key "$SGLANG_ADMIN_API_KEY"
```

The LMSYS HiCache blog benchmark uses the same core HiCache knobs, plus a
longer explicit context length, chunked prefill, and a higher static GPU-memory
budget. Their published long-context command uses DeepSeek-R1 with a 3FS
backend, so the closest single-node local-file version is:

```bash
export SGLANG_ADMIN_API_KEY=1234567890
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/scratch/ows/alphacache/.hicache-file
mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --page-size 64 \
  --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --enable-metrics \
  --enable-mfu-metrics \
  --enable-cache-report \
  --enable-request-time-stats-logging \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --admin-api-key "$SGLANG_ADMIN_API_KEY"
```

  <!-- --context-length 262144 \ -->


Use this command when the goal is to reproduce the public HiCache blog's
long-context stress shape on local SSD/file storage. Use `--tp-size 4` or
`--tp-size 8` for larger models such as `meta-llama/Llama-3.1-70B-Instruct`.
Use `--context-length 262144` only with a model that accepts that window, such
as the Qwen 9B command above. Use `--context-length 131072` for 128K
Strata-style runs, or `65536` for a faster blog-style 64K run. If
`wait_complete` makes TTFT too conservative for your workload, change it to
`best_effort` or `timeout` and record that choice in the candidate report.

The command keeps only the main non-default choices visible. Useful optional
knobs:

- `--port 30000`: default is already `30000`; include it only for clarity or if changing ports.
- `--mem-fraction-static 0.70`: optional GPU memory budget cap; omit it to let SGLang choose.
- `--hicache-ratio 2`: default is already `2`; use `--hicache-size <GB>` when you want a fixed host-cache size.
- `--hicache-io-backend kernel`: default is already `kernel`.
- `--hicache-write-policy write_through`: default is already `write_through`.
- `--hicache-storage-prefetch-policy best_effort`: non-default; keep it if you want aggressive L3 prefetching.
- `--page-size 64` and `--hicache-mem-layout page_first`: intentional for this Strata-style/HiCache setup.


If SGLang's default context length for the selected model is not what you want,
add `--context-length <tokens>` explicitly. Set it to the largest prompt length
you want the server to accept, rounded up with room for generation tokens and
within the model/server limit. Do not use server-side auto truncation for Strata
comparisons; an over-length request should fail loudly rather than become a
different workload.

`--hicache-ratio 2` sizes the host RAM HiCache pool as 2x the GPU KV cache
pool, not 2x system RAM. If the log says something like `Allocating 62.2 GB
host memory for hierarchical KV cache` on a 1 TB machine, that is expected when
the GPU KV pool is about 31 GB. To use a larger L2 host-memory cache, set
`--hicache-size <GB>`; this overrides `--hicache-ratio` and is applied per rank.
For example, `--hicache-size 64` with `--tp-size 8` can allocate roughly 512 GB
total host KV memory. Leave enough RAM for the OS, dataset loading, page cache,
and other jobs.



## Basic Evaluation

Use this as the default all-in-one evaluation command. By default the harness
runs the four Strata workload families:

- Long-document QA: LooGLE and NarrativeQA
- Multi-round dialogue: ShareGPT and ReviewMT

Fast mode keeps the same per-sample workload shape and shortens runtime by
sampling each workload family with a dataset-aware default. By default it uses
5% for large LooGLE/warm-cache sample counts, 50% for NarrativeQA, and 25% for
dataset-backed ShareGPT/ReviewMT clients. It keeps the Strata-style prompt
lengths, conversation rounds, warm-cache token lengths, and prefix percentages.
It runs one request-rate point. It also captures metrics and L3 storage status,
writes a candidate-scoped result folder, keeps going after individual workload
failures, records the full suite terminal transcript in `terminal.log`, and
uses the editable `fitness_template.py` scorer.

With the default fast fraction, the local defaults become small smoke-test
counts:

```text
NarrativeQA: 24 * 0.50 => 12 requests
ShareGPT/ReviewMT: 80 * 0.25 => 20 clients, 20 * 10 rounds => 200 total turns
LooGLE serving: 64 * 0.05 => 4 sampled conversations
warm-cache: 64 * 0.05 => 4 prompts
```

Those are not full Strata-sized counts; they are meant to catch broken runs in a
few minutes.

```bash
export SGLANG_ADMIN_API_KEY=1234567890
export CANDIDATE_ID=smoke_001

cd /scratch/ows/alphacache/extern/sglang/benchmark/hicache

python3 bench_all_in_one.py \
  --data-dir data \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --workloads strata,cache-extra \
  --request-rates "16" \
  --port 30000 \
  --admin-api-key "$SGLANG_ADMIN_API_KEY" \
  --candidate-id "$CANDIDATE_ID" \
  --session-name "$CANDIDATE_ID" \
  --output-dir "bench_all_results/$CANDIDATE_ID" \
  --flush-cache-timeout 120 \
  --fitness-file fitness_template.py \
  --continue-on-error \
  --fast
```

For the full run, remove `--fast` and change `CANDIDATE_ID` to a new session
name. If you explicitly set `--data-fraction 0.10` or `--data-fraction 0.25`,
that value applies to serving, long-context, warm-cache, and ShareGPT/ReviewMT
multi-round client counts. Use `--sample-count 16` to cap those
sample-count/client-count workloads for a middle-sized run. Dataset-backed
workloads sample randomly by default; pass `--serving-disable-shuffle` or
`--synthetic-disable-random-sample` only when you need deterministic first-N
behavior.

If you are debugging on a smaller-context server, keep the result labeled as a
smoke test and pass explicit caps such as `--long-context-max-prompt-len` or
`--serving-max-prompt-len`. The harness does not automatically filter or
truncate Strata workloads.

For a full local Strata-style run at request rate 16, remove `--fast` and pin
the count/load knobs explicitly:

```bash
python3 bench_all_in_one.py \
  --data-dir data \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --workloads strata,cache-extra \
  --request-rates "16" \
  --serving-num-prompts 64 \
  --synthetic-clients 80 \
  --synthetic-rounds 10 \
  --synthetic-request-length 2048 \
  --synthetic-output-length 1 \
  --synthetic-max-parallel 4 \
  --long-context-clients 24 \
  --warm-num-prompts 64 \
  --warm-total-tokens 32768 \
  --port 30000 \
  --admin-api-key "$SGLANG_ADMIN_API_KEY" \
  --candidate-id "$CANDIDATE_ID" \
  --session-name "$CANDIDATE_ID" \
  --output-dir "bench_all_results/$CANDIDATE_ID" \
  --flush-cache-timeout 120 \
  --fitness-file fitness_template.py \
  --continue-on-error
```

The default workload set is `--workloads strata`, which is the four Strata
dataset families:

- `strata-longdoc-loogle`
- `strata-longdoc-narrativeqa`
- `strata-multiround-sharegpt`
- `strata-multiround-reviewmt`

To run every supported workload one by one, use `--workloads all`. This includes
the Strata workloads plus controlled warm-cache, shared-prefix, synthetic
multi-turn, serving multi-turn, and generic long-context runs when their inputs
are available.

```bash
python3 bench_all_in_one.py \
  --workloads all \
  --data-dir data \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --admin-api-key "$SGLANG_ADMIN_API_KEY" \
  --continue-on-error
```

Dataset-backed workloads use `--data-dir` as the only dataset root. The harness
does not accept per-dataset path overrides; put converted files in the standard
layout instead:

```text
data/
  LooGLE/data/longdep_qa.jsonl
  ShareGPT_V3_unfiltered_cleaned_split.json
  narrativeqa_long_context.json
  reviewmt_sharegpt.json
```

Dataset flag:

- `--data-dir`: root scanned for `LooGLE`, `ShareGPT`, `NarrativeQA`, and `ReviewMT` files. Defaults to `./data`.
- `--long-context-max-prompt-len`: optional max NarrativeQA/long-context prompt tokens to send. This changes the workload and should not be used for Strata-faithful results.

If NarrativeQA shows 0 prompt tokens and aborted requests, the server rejected
the selected prompts before prefill, usually because they exceeded
`max_req_input_len`. In strict mode, that means the server/model is not suitable
for that Strata workload. Use a larger-context server rather than truncating or
filtering the dataset.

Shared-prefix and controlled warm-cache sweeps are not part of the four Strata
dataset experiments, but they are useful cache-manager stress tests. Add them
with:

```bash
python3 bench_all_in_one.py \
  --workloads strata,cache-extra \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

The harness always records L3/storage-backend status in `l3_config.json`.
For L3/cache pressure, prefer changing the real load shape: request rates,
concurrency, sample counts, prompt lengths, and `--workloads strata,cache-extra`.

If a run fails with `401 Client Error: Unauthorized for url: .../flush_cache`,
the server was launched with `--admin-api-key` but the benchmark process does
not have the same key. Export `SGLANG_ADMIN_API_KEY` before running the
benchmark, or pass `--admin-api-key "$SGLANG_ADMIN_API_KEY"` to
`bench_all_in_one.py`.

If a run fails with `400 Client Error: Bad Request for url: .../flush_cache`,
SGLang probably still had a request running or draining when the next phase
started. The harness now calls `/flush_cache?timeout=60` by default; increase it
with `--flush-cache-timeout 120` if the server is slow to become idle.

To scale or cap workload sizes:

```bash
# Scale serving, long-context, warm-cache, and dataset-backed multi-round
# client counts to 25%.
python3 bench_all_in_one.py --data-fraction 0.25

# Or cap those sample-count workloads to at most 16 prompts/requests/clients.
python3 bench_all_in_one.py --sample-count 16

# Fast mode is just a ratio-scaled run; override the ratio directly.
python3 bench_all_in_one.py --fast --data-fraction 0.10
```

Dataset-backed serving workloads shuffle by default, so `--data-fraction` and
`--sample-count` act like a random subset unless `--serving-disable-shuffle` is
also passed. For ShareGPT/ReviewMT, `--data-fraction` and `--sample-count`
scale/cap the number of multi-round clients. Total requests are
`clients * --synthetic-rounds`, so the Strata blog-style setting
`--synthetic-clients 80 --synthetic-rounds 10` creates 800 requests. Generic
`synthetic-multiturn` is not dataset-backed; set `--synthetic-clients`
explicitly when you want that generated workload to be smaller.

## Load Scenarios

The load shape matters. A KV-cache manager can win under low pressure, lose
under high pressure, or only shine when shared-prefix reuse is high. Treat the
benchmark setup as part of the claim: report it, keep it fixed for candidate
comparisons, and only let a search loop optimize it when the result is framed as
"best under this scenario".

The main load knobs are:

- `--request-rates`: unified request-rate sweep for LooGLE serving, shared-prefix serving, ShareGPT/ReviewMT multi-round, and NarrativeQA/long-context workloads, in requests/sec.
- `--serving-request-rates`: advanced override for LooGLE serving and shared-prefix serving only.
- `--serving-max-concurrency`: cap concurrent serving requests.
- `--synthetic-request-rates`: advanced override for ShareGPT/ReviewMT/synthetic multi-round workloads only.
- `--synthetic-clients`: number of multi-round clients.
- `--synthetic-max-parallel`: cap in-flight multi-round requests.
- `--synthetic-rounds`: number of turns per client.
- `--synthetic-request-length`: target token length for sampled multi-round prompts.
- `--long-context-request-rate`: advanced single-rate override for NarrativeQA/long-context when `--request-rates` is not set.
- `--long-context-clients`: number of NarrativeQA/long-context requests.
- `--long-context-max-prompt-len`: prompt-token cap for NarrativeQA/long-context; useful only for smoke/debug runs.
- `--warm-max-concurrency`: concurrency for the controlled warm-cache sweep.
- `--warm-pcts`: shared-prefix percentages for the warm-cache sweep.
- `--warm-total-tokens`: total prompt tokens per warm-cache request. In strict mode this is not capped automatically.

Rate lists accept either explicit comma-separated values or `start,stop,step`.
For example, `2,16,2` expands to `2,4,6,8,10,12,14,16`, while `2,4,8`
is treated as the explicit list `2,4,8`.

The local Strata-style reference scenario in this harness is:

```text
workloads: strata
serving_request_rates: 1,2,4,8,12,16,24
serving_num_prompts: 64
synthetic_request_rates: 1,2,3,4,5,6,7,8,9,10,12,14,16
synthetic_clients: 80
synthetic_rounds: 10
synthetic_request_length: 2048
synthetic_output_length: 1
synthetic_max_parallel: 4
long_context_clients: 24
long_context_request_rate: 8
warm_total_tokens: 32768 for cache-extra, unless explicitly overridden
```

For repeatable scenario control, keep the whole benchmark command in a shell
script or job file. That makes the load shape explicit and avoids hidden Python
layers mutating benchmark arguments.

```bash
python3 bench_all_in_one.py \
  --workloads strata,cache-extra \
  --request-rates "2,4,8,12" \
  --serving-num-prompts 64 \
  --synthetic-clients 80 \
  --synthetic-rounds 10 \
  --synthetic-request-length 2048 \
  --synthetic-output-length 1 \
  --synthetic-max-parallel 4 \
  --long-context-clients 24 \
  --warm-num-prompts 32 \
  --warm-total-tokens 32768 \
  --warm-max-concurrency 8 \
  --data-dir data \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

The selected setup is saved into `suite_args.json`, `fitness_summary.json`, and
`report.md`. For fair comparisons, freeze the benchmark command, server launch
args, model, datasets, and `fitness_template.py` for the whole candidate
population.

By default, results are written under:

```text
bench_all_results/<session-name>/
```

Important files:

- `fitness.txt`: one numeric final score. Higher is better.
- `fitness_summary.json`: machine-readable session summary.
- `report.md`: human-readable dashboard with aggregate stats, workload/dataset summaries, run matrix, per-run commands, and artifact paths.
- `terminal.log`: full parent harness terminal transcript, including child benchmark output streamed to the console.
- `runs.jsonl`: one full record per workload run.
- `<run-name>/feedback_metrics.json`: compact metrics used by the scorer.
- `<run-name>/run_record.json`: full per-run record.
- `<run-name>/before_metrics.prom` and `after_metrics.prom`: raw Prometheus snapshots.
- `<run-name>/stdout.log` and `stderr.log`: child benchmark logs.
- `l3_config.json`: L3 setup/status.

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

`report.md` gives a human dashboard. `fitness_summary.json` and each
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
6. Read `fitness_summary.json` and `report.md` for reflection.
7. Preserve the full `bench_all_results/<id>/` folder with the candidate patch.

Keep the workload fixed while comparing candidates. Change the fitness function
only between search campaigns, not halfway through a population, unless the
optimizer is explicitly aware of the objective change.
