#!/usr/bin/env python3
"""
Run a HiCache benchmark suite against an already-running SGLang server.

The default suite follows the Strata dataset split:

* strata-longdoc-loogle: LooGLE long-document QA workload.
* strata-longdoc-narrativeqa: NarrativeQA long-document workload, if converted.
* strata-multiround-sharegpt: ShareGPT-derived multi-round workload.
* strata-multiround-reviewmt: ReviewMT multi-round workload, if converted.

Additional cache-stress workloads are available explicitly:

* serving-multiturn: LooGLE long dependency QA via OpenAI-compatible serving.
* serving-shared-prefix: LooGLE shared-prefix serving pattern.
* synthetic-multiturn: generated multi-round token workload.
* warm-cache: controlled shared-prefix warm-cache sweep.
* long-context: optional Strata-style long-context JSON workload.

Start the server separately with the HiCache settings under test. For useful
memory/cache output, launch the server with --enable-metrics and
--enable-cache-report.
"""

import argparse
import contextlib
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from bench_all_report import build_session_summary, write_report
except ImportError:  # pragma: no cover - used when run as a package module.
    from .bench_all_report import build_session_summary, write_report


DEFAULT_WORKLOADS = (
    "strata-longdoc-loogle",
    "strata-longdoc-narrativeqa",
    "strata-multiround-sharegpt",
    "strata-multiround-reviewmt",
)

EXTRA_CACHE_WORKLOADS = (
    "warm-cache",
    "serving-shared-prefix",
)

ALL_WORKLOADS = DEFAULT_WORKLOADS + EXTRA_CACHE_WORKLOADS + (
    "serving-multiturn",
    "synthetic-multiturn",
    "long-context",
)

DEFAULT_SERVING_REQUEST_RATES = "1,2,4,8,12,16,24"
DEFAULT_SYNTHETIC_REQUEST_RATES = "1,2,3,4,5,6,7,8,9,10,12,14,16"
DEFAULT_SERVING_NUM_PROMPTS = 64
DEFAULT_SYNTHETIC_CLIENTS = 80
DEFAULT_SYNTHETIC_ROUNDS = 10
DEFAULT_SYNTHETIC_REQUEST_LENGTH = 2048
DEFAULT_SYNTHETIC_OUTPUT_LENGTH = 1
DEFAULT_SYNTHETIC_MAX_PARALLEL = 4
DEFAULT_WARM_NUM_PROMPTS = 64
DEFAULT_WARM_TOTAL_TOKENS = 32768
DEFAULT_WARM_OUTPUT_LEN = 32
DEFAULT_WARM_MAX_CONCURRENCY = 4
DEFAULT_WARM_PCTS = "0,50,80,90,95,99"
DEFAULT_LONG_CONTEXT_CLIENTS = 24
DEFAULT_LONG_CONTEXT_REQUEST_RATE = 8.0
DEFAULT_DATA_DIR = "data"

FAST_DATA_FRACTION = 0.05
FAST_LONG_CONTEXT_FRACTION = 0.50
FAST_DATASET_MULTITURN_FRACTION = 0.25
FAST_MIN_SERVING_PROMPTS = 4
FAST_MIN_WARM_PROMPTS = 4
FAST_MIN_LONG_CONTEXT_CLIENTS = 8
FAST_MIN_DATASET_MULTITURN_CLIENTS = 8
FAST_REQUEST_RATES = "16"

DATA_DIR_CANDIDATES = {
    "loogle": (
        "LooGLE/data/longdep_qa.jsonl",
        "loogle/longdep_qa.jsonl",
        "longdep_qa.jsonl",
    ),
    "sharegpt": (
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        "sharegpt.json",
    ),
    "reviewmt": (
        "reviewmt_sharegpt.json",
        "ReviewMT/reviewmt_sharegpt.json",
        "reviewmt/reviewmt_sharegpt.json",
        "reviewmt_test_sharegpt.json",
    ),
    "narrativeqa": (
        "narrativeqa_long_context.json",
        "NarrativeQA/narrativeqa_long_context.json",
        "narrativeqa/narrativeqa_long_context.json",
    ),
}

DEFAULT_METRIC_PATTERNS = (
    "sglang:token_usage",
    "sglang:full_token_usage",
    "sglang:swa_token_usage",
    "sglang:mamba_usage",
    "sglang:num_used_tokens",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:cache_hit_rate",
    "sglang:gen_throughput",
    "sglang:cached_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:uncached_prompt_tokens_histogram",
    "sglang:num_requests_total",
    "sglang:num_aborted_requests_total",
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
    "sglang:num_used_reqs",
    "sglang:num_waiting_reqs",
    "sglang:time_to_first_token_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang:prefetched_tokens_total",
    "sglang:backuped_tokens_total",
    "sglang:prefetch_pgs",
    "sglang:backup_pgs",
    "sglang:prefetch_bandwidth",
    "sglang:backup_bandwidth",
    "sglang:evicted_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:eviction_duration_seconds",
    "sglang:load_back_duration_seconds",
    "sglang:max_total_num_tokens",
    "sglang:page_size",
    "sglang:num_pages",
    "sglang:context_len",
    "sglang:startup_available_gpu_memory_gb",
    "sglang:realtime_tokens_total",
    "sglang:forward_execution_seconds_total",
    "sglang:estimated_flops_per_gpu_total",
    "sglang:estimated_read_bytes_per_gpu_total",
    "sglang:estimated_write_bytes_per_gpu_total",
)

GAUGE_METRICS = {
    "sglang:token_usage",
    "sglang:full_token_usage",
    "sglang:swa_token_usage",
    "sglang:mamba_usage",
    "sglang:num_used_tokens",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:cache_hit_rate",
    "sglang:gen_throughput",
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
    "sglang:num_used_reqs",
    "sglang:num_waiting_reqs",
    "sglang:num_paused_reqs",
    "sglang:max_total_num_tokens",
    "sglang:page_size",
    "sglang:num_pages",
    "sglang:context_len",
    "sglang:startup_available_gpu_memory_gb",
}

COUNTER_METRICS = {
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_requests_total",
    "sglang:num_aborted_requests_total",
    "sglang:prefetched_tokens_total",
    "sglang:backuped_tokens_total",
    "sglang:evicted_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:num_retracted_requests_total",
    "sglang:num_retracted_input_tokens_total",
    "sglang:num_retracted_output_tokens_total",
    "sglang:realtime_tokens_total",
    "sglang:forward_execution_seconds_total",
    "sglang:estimated_flops_per_gpu_total",
    "sglang:estimated_read_bytes_per_gpu_total",
    "sglang:estimated_write_bytes_per_gpu_total",
}

HISTOGRAM_METRICS = {
    "sglang:prompt_tokens_histogram",
    "sglang:uncached_prompt_tokens_histogram",
    "sglang:generation_tokens_histogram",
    "sglang:time_to_first_token_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang:prefetch_pgs",
    "sglang:backup_pgs",
    "sglang:prefetch_bandwidth",
    "sglang:backup_bandwidth",
    "sglang:eviction_duration_seconds",
    "sglang:load_back_duration_seconds",
    "sglang:kv_transfer_speed_gb_s",
    "sglang:kv_transfer_latency_ms",
}

METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@dataclass
class RunSpec:
    name: str
    workload: str
    cmd: List[str]
    output_path: Optional[Path] = None


@dataclass
class FitnessMetrics:
    """Small helper object intentionally kept easy to edit for search loops."""

    values: Dict[str, float]

    def get(self, name: str, default: float = 0.0) -> float:
        value = self.values.get(name, default)
        return default if value is None else float(value)

    def __getattr__(self, name: str) -> float:
        return self.get(name)


def default_fitness(metrics: FitnessMetrics) -> float:
    """Higher is better. Edit this function for your research objective.

    Examples:
      fitness = -metrics.ttft_ms
      fitness = 0.9 * -metrics.ttft_ms + 0.3 * metrics.cache_hit_rate
      fitness = metrics.throughput_out_tok_s - 0.1 * metrics.e2e_ms
    """

    ttft_ms = metrics.get("ttft_ms", 100000.0)
    e2e_ms = metrics.get("e2e_ms", 100000.0)
    output_throughput = metrics.get("throughput_out_tok_s", 0.0)
    cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
    host_tokens = metrics.get("host_kv_used_tokens", 0.0)
    storage_hits = metrics.get("cached_storage_file_tokens", 0.0)
    aborted = metrics.get("aborted_requests", 0.0)

    return (
        1000.0 * cache_hit_rate
        + 0.05 * output_throughput
        + 0.00001 * host_tokens
        + 0.00001 * storage_hits
        - 0.10 * ttft_ms
        - 0.02 * e2e_ms
        - 1000.0 * aborted
    )


def load_fitness_function(path: str):
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("hicache_fitness_plugin", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load fitness file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    module.FitnessMetrics = FitnessMetrics
    module.default_fitness = default_fitness
    spec.loader.exec_module(module)

    func = getattr(module, "score", None) or getattr(module, "fitness", None)
    if not callable(func):
        raise ValueError(
            f"{module_path} must define a callable score(metrics) or fitness(metrics)"
        )
    return func


def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_rates(value: str) -> List[float]:
    pieces = parse_csv(value)
    if len(pieces) == 3:
        try:
            start = float(pieces[0])
            stop = float(pieces[1])
            step = float(pieces[2])
        except ValueError:
            pass
        else:
            if step > 0 and start + step <= stop + 1e-9:
                rates = []
                current = start
                while current <= stop + 1e-9:
                    rates.append(round(current, 10))
                    current += step
                return rates
    return [float(piece) for piece in pieces]


def serving_rates(args: argparse.Namespace) -> List[float]:
    if args.request_rates:
        return parse_rates(args.request_rates)
    if args.mode == "fast" and args.serving_request_rates == DEFAULT_SERVING_REQUEST_RATES:
        return parse_rates(FAST_REQUEST_RATES)
    return parse_rates(args.serving_request_rates)


def synthetic_rates(args: argparse.Namespace) -> List[float]:
    if args.request_rates:
        return parse_rates(args.request_rates)
    if (
        args.mode == "fast"
        and args.synthetic_request_rates == DEFAULT_SYNTHETIC_REQUEST_RATES
    ):
        return parse_rates(FAST_REQUEST_RATES)
    return parse_rates(args.synthetic_request_rates)


def long_context_rates(args: argparse.Namespace) -> List[float]:
    if args.request_rates:
        return parse_rates(args.request_rates)
    if (
        args.mode == "fast"
        and args.long_context_request_rate == DEFAULT_LONG_CONTEXT_REQUEST_RATE
    ):
        return parse_rates(FAST_REQUEST_RATES)
    return [float(args.long_context_request_rate)]


def split_extra(value: Optional[str]) -> List[str]:
    return shlex.split(value) if value else []


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    return f"http://{args.host}:{args.port}"


def merge_headers(headers: Optional[dict] = None) -> dict:
    merged = {"User-Agent": "bench-all-in-one"}
    if headers:
        merged.update(headers)
    return merged


def admin_headers(args: argparse.Namespace, json_body: bool = False) -> dict:
    headers = {}
    token = args.admin_api_key or os.environ.get("SGLANG_ADMIN_API_KEY", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def request_text(
    url: str, timeout: float = 10.0, headers: Optional[dict] = None
) -> Optional[str]:
    try:
        req = Request(url, headers=merge_headers(headers))
        with urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, URLError):
        return None


def post(url: str, timeout: float = 30.0, headers: Optional[dict] = None) -> bool:
    try:
        req = Request(url, method="POST", headers=merge_headers(headers))
        with urlopen(req, timeout=timeout) as response:
            response.read()
        return True
    except (OSError, URLError):
        return False


def post_json(
    url: str, payload: dict, timeout: float = 30.0, headers: Optional[dict] = None
) -> bool:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=body,
            method="POST",
            headers=merge_headers({"Content-Type": "application/json", **(headers or {})}),
        )
        with urlopen(req, timeout=timeout) as response:
            response.read()
        return True
    except (OSError, URLError):
        return False


def put_json(
    url: str, payload: dict, timeout: float = 30.0, headers: Optional[dict] = None
) -> bool:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=body,
            method="PUT",
            headers=merge_headers({"Content-Type": "application/json", **(headers or {})}),
        )
        with urlopen(req, timeout=timeout) as response:
            response.read()
        return True
    except (OSError, URLError):
        return False


def write_text(path: Path, text: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else "", encoding="utf-8")


def find_data_file(data_dir: str, dataset_key: str) -> str:
    if not data_dir:
        return ""
    root = Path(data_dir).expanduser()
    candidates = DATA_DIR_CANDIDATES.get(dataset_key, ())
    for rel_path in candidates:
        path = root / rel_path
        try:
            if path.is_file() and path.stat().st_size > 0:
                return str(path.resolve())
        except OSError:
            continue
    return ""


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(obj, sort_keys=True) + "\n")


class TeeBuffer:
    def __init__(self, primary, log_buffer):
        self.primary = primary
        self.log_buffer = log_buffer

    def write(self, data: bytes) -> int:
        self.primary.write(data)
        self.log_buffer.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_buffer.flush()


class TeeText:
    def __init__(self, primary: TextIO, log_file: TextIO):
        self.primary = primary
        self.log_file = log_file
        self.buffer = TeeBuffer(primary.buffer, log_file.buffer)
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


def read_jsonl(path: Optional[Path]) -> List[dict]:
    if not path or not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def numeric_items(record: dict) -> Iterable[Tuple[str, float]]:
    for key, value in record.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield key, float(value)


def filter_metrics(metrics_text: Optional[str], patterns: Iterable[str]) -> str:
    if not metrics_text:
        return ""
    selected = []
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        if any(pattern in line for pattern in patterns):
            selected.append(line)
    return "\n".join(selected) + ("\n" if selected else "")


def parse_metric_labels(label_text: str) -> Dict[str, str]:
    labels = {}
    if not label_text:
        return labels
    body = label_text.strip("{}")
    for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', body):
        labels[match.group(1)] = match.group(2)
    return labels


def summarize_metrics(metrics_text: Optional[str]) -> Dict[str, float]:
    """Return compact numeric metrics useful for comparing HiCache pressure."""
    summary: Dict[str, float] = {}
    if not metrics_text:
        return summary

    cached_by_source: Dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        if raw_line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(raw_line.strip())
        if not match:
            continue

        name = match.group("name")
        value = float(match.group("value"))
        labels = parse_metric_labels(match.group("labels") or "")

        if name in GAUGE_METRICS:
            summary[name] = max(value, summary.get(name, value))
        elif name == "sglang:cached_tokens_total":
            source = labels.get("cache_source", "total")
            cached_by_source[source] = cached_by_source.get(source, 0.0) + value
        elif name in COUNTER_METRICS:
            summary[name] = summary.get(name, 0.0) + value
        else:
            for base_name in HISTOGRAM_METRICS:
                if name in {f"{base_name}_sum", f"{base_name}_count"}:
                    summary[name] = summary.get(name, 0.0) + value
                    break

    for source, value in cached_by_source.items():
        summary[f"sglang:cached_tokens_total[{source}]"] = value

    for base_name in HISTOGRAM_METRICS:
        total = summary.get(f"{base_name}_sum")
        count = summary.get(f"{base_name}_count")
        if total is not None and count:
            summary[f"{base_name}_avg"] = total / count

    return summary


def diff_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    keys = {key for key in set(before) | set(after) if not key.endswith("_avg")}
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(keys)}
    for base_name in HISTOGRAM_METRICS:
        total = delta.get(f"{base_name}_sum")
        count = delta.get(f"{base_name}_count")
        if total is not None and count and count > 0:
            delta[f"{base_name}_avg"] = total / count
    return delta


def flatten_run_metrics(
    spec: RunSpec,
    rc: int,
    elapsed: float,
    benchmark_records: List[dict],
    metrics_after: Dict[str, float],
    metrics_delta: Dict[str, float],
) -> Dict[str, float]:
    values: Dict[str, float] = {
        "returncode": float(rc),
        "elapsed_sec": elapsed,
        "success": 1.0 if rc == 0 else 0.0,
    }

    if benchmark_records:
        last = benchmark_records[-1]
        for key, value in numeric_items(last):
            values[f"bench.{key}"] = value

        aliases = {
            "mean_ttft_ms": "ttft_ms",
            "median_ttft_ms": "median_ttft_ms",
            "p90_ttft_ms": "p90_ttft_ms",
            "p99_ttft_ms": "p99_ttft_ms",
            "mean_itl_ms": "itl_ms",
            "mean_e2e_latency_ms": "e2e_ms",
            "request_throughput": "throughput_req_s",
            "input_throughput": "throughput_in_tok_s",
            "output_throughput": "throughput_out_tok_s",
            "completed": "completed_requests",
            "total_input_tokens": "input_tokens",
            "total_output_tokens": "output_tokens",
        }
        for source, target in aliases.items():
            if isinstance(last.get(source), (int, float)):
                values[target] = float(last[source])

    for key, value in metrics_after.items():
        values[f"prom.after.{key}"] = value
    for key, value in metrics_delta.items():
        values[f"prom.delta.{key}"] = value

    metric_aliases = {
        "sglang:cache_hit_rate": ("cache_hit_rate", "after"),
        "sglang:token_usage": ("gpu_token_usage", "after"),
        "sglang:kv_used_tokens": ("gpu_kv_used_tokens", "after"),
        "sglang:kv_evictable_tokens": ("gpu_kv_evictable_tokens", "after"),
        "sglang:kv_available_tokens": ("gpu_kv_available_tokens", "after"),
        "sglang:hicache_host_used_tokens": ("host_kv_used_tokens", "after"),
        "sglang:hicache_host_total_tokens": ("host_kv_total_tokens", "after"),
        "sglang:cached_tokens_total[device]": ("cached_device_tokens", "delta"),
        "sglang:cached_tokens_total[host]": ("cached_host_tokens", "delta"),
        "sglang:cached_tokens_total[storage_file]": (
            "cached_storage_file_tokens",
            "delta",
        ),
        "sglang:cached_tokens_total[storage_mooncake]": (
            "cached_storage_mooncake_tokens",
            "delta",
        ),
        "sglang:cached_tokens_total[storage_hf3fs]": (
            "cached_storage_hf3fs_tokens",
            "delta",
        ),
        "sglang:prompt_tokens_total": ("prompt_tokens", "delta"),
        "sglang:generation_tokens_total": ("generation_tokens", "delta"),
        "sglang:num_requests_total": ("requests", "delta"),
        "sglang:num_aborted_requests_total": ("aborted_requests", "delta"),
        "sglang:prefetched_tokens_total": ("prefetched_tokens", "delta"),
        "sglang:backuped_tokens_total": ("backuped_tokens", "delta"),
        "sglang:evicted_tokens_total": ("evicted_tokens", "delta"),
        "sglang:load_back_tokens_total": ("load_back_tokens", "delta"),
        "sglang:time_to_first_token_seconds_avg": ("prom_ttft_sec", "delta"),
        "sglang:e2e_request_latency_seconds_avg": ("prom_e2e_sec", "delta"),
        "sglang:inter_token_latency_seconds_avg": ("prom_itl_sec", "delta"),
    }
    for source, (target, scope) in metric_aliases.items():
        source_map = metrics_after if scope == "after" else metrics_delta
        if source in source_map:
            values[target] = source_map[source]

    # Fallback to Prometheus latency if a workload does not emit bench_serving JSON.
    if "ttft_ms" not in values and "prom_ttft_sec" in values:
        values["ttft_ms"] = values["prom_ttft_sec"] * 1000.0
    if "e2e_ms" not in values and "prom_e2e_sec" in values:
        values["e2e_ms"] = values["prom_e2e_sec"] * 1000.0
    if "itl_ms" not in values and "prom_itl_sec" in values:
        values["itl_ms"] = values["prom_itl_sec"] * 1000.0

    values["workload_hash"] = float(
        sum((idx + 1) * ord(ch) for idx, ch in enumerate(spec.workload))
    )
    return values


def compute_fitness(args: argparse.Namespace, metrics: Dict[str, float]) -> float:
    if metrics.get("success", 0.0) <= 0:
        return args.fitness_fail_score

    view = FitnessMetrics(metrics)
    if args.fitness_file:
        return float(load_fitness_function(args.fitness_file)(view))
    if args.fitness_expr:
        allowed = {key: value for key, value in metrics.items() if key.isidentifier()}
        allowed.update({"m": view, "metrics": view, "math": math})
        return float(eval(args.fitness_expr, {"__builtins__": {}}, allowed))
    return float(default_fitness(view))


def feedback_for_run(metrics: Dict[str, float]) -> List[str]:
    feedback = []
    if metrics.get("success", 0.0) <= 0:
        feedback.append("run failed; inspect stdout.log and stderr.log")
    if metrics.get("aborted_requests", 0.0) > 0:
        feedback.append(f"aborted_requests={metrics['aborted_requests']:.0f}")
    if "cache_hit_rate" in metrics and metrics["cache_hit_rate"] < 0.80:
        feedback.append(f"low cache_hit_rate={metrics['cache_hit_rate']:.3f}")
    if metrics.get("host_kv_total_tokens", 0.0) > 0:
        ratio = metrics.get("host_kv_used_tokens", 0.0) / metrics["host_kv_total_tokens"]
        feedback.append(f"host_cache_usage={ratio:.3f}")
    if metrics.get("cached_host_tokens", 0.0) > 0:
        feedback.append(f"host_cache_hits={metrics['cached_host_tokens']:.0f} tokens")
    if metrics.get("cached_storage_file_tokens", 0.0) > 0:
        feedback.append(
            f"file_storage_hits={metrics['cached_storage_file_tokens']:.0f} tokens"
        )
    if metrics.get("cached_storage_mooncake_tokens", 0.0) > 0:
        feedback.append(
            f"mooncake_storage_hits={metrics['cached_storage_mooncake_tokens']:.0f} tokens"
        )
    if metrics.get("cached_storage_hf3fs_tokens", 0.0) > 0:
        feedback.append(
            f"hf3fs_storage_hits={metrics['cached_storage_hf3fs_tokens']:.0f} tokens"
        )
    if metrics.get("load_back_tokens", 0.0) > 0:
        feedback.append(f"load_back_tokens={metrics['load_back_tokens']:.0f}")
    if metrics.get("evicted_tokens", 0.0) > 0:
        feedback.append(f"evicted_tokens={metrics['evicted_tokens']:.0f}")
    if not feedback:
        feedback.append("no obvious cache pressure feedback; increase workload size/rate")
    return feedback


def capture_server_snapshot(
    args: argparse.Namespace, run_dir: Path, label: str
) -> Dict[str, float]:
    if args.no_capture_metrics:
        return {}

    url = base_url(args)
    metrics = request_text(f"{url}/metrics", timeout=args.metrics_timeout)
    server_info = request_text(f"{url}/server_info", timeout=args.metrics_timeout)

    write_text(run_dir / f"{label}_metrics.prom", metrics)
    write_text(
        run_dir / f"{label}_metrics_selected.prom",
        filter_metrics(metrics, DEFAULT_METRIC_PATTERNS),
    )
    write_text(run_dir / f"{label}_server_info.json", server_info)
    return summarize_metrics(metrics)


def l3_storage_payload(args: argparse.Namespace) -> dict:
    return {
        "hicache_storage_backend": args.l3_storage_backend,
        "hicache_storage_backend_extra_config_json": args.l3_extra_config or "{}",
        "hicache_storage_prefetch_policy": args.l3_prefetch_policy,
        "hicache_write_policy": args.l3_write_policy,
    }


def get_l3_status(args: argparse.Namespace) -> Optional[dict]:
    text = request_text(
        f"{base_url(args)}/hicache/storage-backend",
        timeout=args.metrics_timeout,
        headers=admin_headers(args),
    )
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def clear_l3_storage(args: argparse.Namespace, reason: str) -> bool:
    ok = post(
        f"{base_url(args)}/hicache/storage-backend/clear",
        timeout=120.0,
        headers=admin_headers(args),
    )
    print(f"[l3] clear_storage({reason})={'ok' if ok else 'failed'}")
    return ok


def prepare_l3_cache(args: argparse.Namespace, suite_dir: Path) -> dict:
    if args.l3_storage_backend == "file" and args.l3_storage_dir:
        Path(args.l3_storage_dir).mkdir(parents=True, exist_ok=True)

    config = {
        "enabled": True,
        "runtime_attach": args.l3_runtime_attach,
        "clear_before_suite": args.clear_l3_cache,
        "clear_before_run": args.clear_l3_cache_before_run,
        "file_storage_dir": args.l3_storage_dir,
        "payload": l3_storage_payload(args),
        "status_before": None,
        "status_after": None,
        "attach_ok": None,
        "clear_ok": None,
        "notes": [],
    }

    if args.l3_storage_backend == "file":
        config["notes"].append(
            "The SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR environment variable "
            "must be set in the SGLang server process before launch for the "
            "file backend to use this directory."
        )

    if args.dry_run:
        write_text(suite_dir / "l3_config.json", json.dumps(config, indent=2))
        return config

    config["status_before"] = get_l3_status(args)

    if args.l3_runtime_attach:
        if not (args.admin_api_key or os.environ.get("SGLANG_ADMIN_API_KEY")):
            config["notes"].append(
                "Runtime L3 attach requested but no admin key was provided. "
                "Start the server with --admin-api-key and pass --admin-api-key "
                "or SGLANG_ADMIN_API_KEY."
            )
            print("[l3] runtime attach skipped: missing --admin-api-key")
        else:
            config["attach_ok"] = put_json(
                f"{base_url(args)}/hicache/storage-backend",
                l3_storage_payload(args),
                timeout=120.0,
                headers=admin_headers(args, json_body=True),
            )
            print(
                f"[l3] runtime_attach={ 'ok' if config['attach_ok'] else 'failed' } "
                f"backend={args.l3_storage_backend}"
            )

    config["status_after"] = get_l3_status(args)

    if args.clear_l3_cache:
        config["clear_ok"] = clear_l3_storage(args, "suite")

    write_text(suite_dir / "l3_config.json", json.dumps(config, indent=2))
    return config


def print_metric_summary(
    name: str,
    metrics: Dict[str, float],
    title: str,
    include_gauges: bool = True,
) -> None:
    if not metrics:
        print(f"[{name}] {title} unavailable; start server with --enable-metrics")
        return

    gauges = [
        "sglang:token_usage",
        "sglang:full_token_usage",
        "sglang:num_used_tokens",
        "sglang:kv_used_tokens",
        "sglang:kv_evictable_tokens",
        "sglang:kv_available_tokens",
        "sglang:hicache_host_used_tokens",
        "sglang:hicache_host_total_tokens",
        "sglang:cache_hit_rate",
        "sglang:gen_throughput",
        "sglang:num_running_reqs",
        "sglang:num_queue_reqs",
    ]
    counters_and_histograms = [
        "sglang:cached_tokens_total[device]",
        "sglang:cached_tokens_total[host]",
        "sglang:cached_tokens_total[storage_file]",
        "sglang:cached_tokens_total[storage_mooncake]",
        "sglang:cached_tokens_total[storage_hf3fs]",
        "sglang:cached_tokens_total[total]",
        "sglang:prompt_tokens_total",
        "sglang:generation_tokens_total",
        "sglang:num_requests_total",
        "sglang:num_aborted_requests_total",
        "sglang:prefetched_tokens_total",
        "sglang:backuped_tokens_total",
        "sglang:evicted_tokens_total",
        "sglang:load_back_tokens_total",
        "sglang:time_to_first_token_seconds_avg",
        "sglang:inter_token_latency_seconds_avg",
        "sglang:e2e_request_latency_seconds_avg",
        "sglang:prefetch_pgs_avg",
        "sglang:backup_pgs_avg",
        "sglang:prefetch_bandwidth_avg",
        "sglang:backup_bandwidth_avg",
        "sglang:eviction_duration_seconds_avg",
        "sglang:load_back_duration_seconds_avg",
        "sglang:realtime_tokens_total",
        "sglang:forward_execution_seconds_total",
        "sglang:estimated_flops_per_gpu_total",
        "sglang:estimated_read_bytes_per_gpu_total",
        "sglang:estimated_write_bytes_per_gpu_total",
    ]
    interesting = (gauges if include_gauges else []) + counters_and_histograms
    parts = []
    for key in interesting:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4g}")
    if parts:
        print(f"[{name}] {title}: " + ", ".join(parts))
    else:
        print(f"[{name}] {title} captured, but no selected HiCache metrics found")


def tee_stream(pipe, log_file, console_stream, quiet: bool) -> None:
    with pipe, log_file:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            log_file.write(chunk)
            log_file.flush()
            if not quiet:
                console_stream.buffer.write(chunk)
                console_stream.buffer.flush()


def run_child_process(args: argparse.Namespace, spec: RunSpec, run_dir: Path) -> int:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.admin_api_key:
        env["SGLANG_ADMIN_API_KEY"] = args.admin_api_key

    proc = subprocess.Popen(
        spec.cmd,
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    threads = [
        threading.Thread(
            target=tee_stream,
            args=(
                proc.stdout,
                stdout_path.open("wb"),
                sys.stdout,
                args.quiet_child_output,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=tee_stream,
            args=(
                proc.stderr,
                stderr_path.open("wb"),
                sys.stderr,
                args.quiet_child_output,
            ),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    rc = proc.wait()
    for thread in threads:
        thread.join()
    return rc


def preflight_server(args: argparse.Namespace) -> None:
    server_info_text = request_text(f"{base_url(args)}/server_info", timeout=5.0)
    if not server_info_text:
        print("[preflight] server_info unavailable; is the SGLang server running?")
        return

    try:
        info = json.loads(server_info_text)
    except json.JSONDecodeError:
        print("[preflight] server_info is not valid JSON")
        return

    max_req_input_len = info.get("max_req_input_len")
    print(
        "[preflight] "
        f"model={info.get('model_path')} "
        f"tp_size={info.get('tp_size')} "
        f"max_req_input_len={max_req_input_len} "
        f"enable_hicache={info.get('enable_hierarchical_cache')} "
        f"enable_metrics={info.get('enable_metrics')} "
        f"enable_cache_report={info.get('enable_cache_report')} "
        f"hicache_storage_backend={info.get('hicache_storage_backend')}"
    )

    if not args.no_capture_metrics and not info.get("enable_metrics"):
        print("[preflight] warning: restart server with --enable-metrics for memory metrics")
    if not info.get("enable_cache_report"):
        print(
            "[preflight] warning: restart server with --enable-cache-report "
            "for cached-token source details"
        )
    if max_req_input_len is not None and max_req_input_len < 65536:
        print(
            "[preflight] warning: this server has a small max input length for "
            "Strata-style long-context workloads. Strict mode will not filter "
            "or truncate prompts; over-length workloads should fail instead of "
            "being silently changed."
        )
    if max_req_input_len is not None and args.warm_total_tokens > max_req_input_len:
        print(
            f"[preflight] warning: warm-cache total_tokens={args.warm_total_tokens} "
            f"exceeds server max_req_input_len={max_req_input_len}."
        )


def start_profile(args: argparse.Namespace, spec: RunSpec, run_dir: Path) -> Optional[Path]:
    if not args.profile:
        return None

    if args.profile_output_dir:
        profile_dir = Path(args.profile_output_dir).resolve() / safe_name(spec.name)
    else:
        profile_dir = run_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(profile_dir),
        "activities": parse_csv(args.profile_activities),
        "merge_profiles": args.profile_merge,
        "profile_prefix": safe_name(spec.name),
    }
    if args.profile_by_stage:
        payload["profile_by_stage"] = True
    if args.profile_stages:
        payload["profile_stages"] = parse_csv(args.profile_stages)
    if args.profile_with_stack is not None:
        payload["with_stack"] = args.profile_with_stack
    if args.profile_record_shapes is not None:
        payload["record_shapes"] = args.profile_record_shapes
    if args.profile_start_step is not None:
        payload["start_step"] = args.profile_start_step
    if args.profile_num_steps is not None:
        payload["num_steps"] = args.profile_num_steps

    ok = post_json(
        f"{base_url(args)}/start_profile",
        payload,
        timeout=60.0,
        headers=admin_headers(args, json_body=True),
    )
    print(f"[{spec.name}] start_profile={'ok' if ok else 'failed'} dir={profile_dir}")
    write_text(run_dir / "profile_request.json", json.dumps(payload, indent=2))
    return profile_dir


def stop_profile(args: argparse.Namespace, spec: RunSpec) -> bool:
    if not args.profile:
        return False
    ok = post(
        f"{base_url(args)}/stop_profile",
        timeout=120.0,
        headers=admin_headers(args),
    )
    print(f"[{spec.name}] stop_profile={'ok' if ok else 'failed'}")
    return ok


def run_command(args: argparse.Namespace, spec: RunSpec, suite_dir: Path) -> dict:
    run_dir = suite_dir / safe_name(spec.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    meta_path = suite_dir / "runs.jsonl"

    if not args.dry_run and args.clear_l3_cache_before_run:
        clear_l3_storage(args, spec.name)

    if not args.dry_run and not args.no_flush_cache:
        flush_timeout = max(float(args.flush_cache_timeout), 0.0)
        flushed = post(
            f"{base_url(args)}/flush_cache?timeout={flush_timeout}",
            timeout=flush_timeout + 30.0,
            headers=admin_headers(args),
        )
        print(f"[{spec.name}] flush_cache={'ok' if flushed else 'failed'}")
        time.sleep(args.post_flush_sleep)

    before_metrics = (
        {} if args.dry_run else capture_server_snapshot(args, run_dir, "before")
    )

    profile_dir = None if args.dry_run else start_profile(args, spec, run_dir)

    if spec.output_path and spec.output_path.exists():
        spec.output_path.unlink()

    print(f"[{spec.name}] {' '.join(shlex.quote(part) for part in spec.cmd)}")
    start = time.time()
    if args.dry_run:
        rc = 0
    else:
        rc = run_child_process(args, spec, run_dir)
    elapsed = time.time() - start
    if not args.dry_run:
        stop_profile(args, spec)

    after_metrics = (
        {} if args.dry_run else capture_server_snapshot(args, run_dir, "after")
    )
    metric_delta = diff_metrics(before_metrics, after_metrics)
    benchmark_records = read_jsonl(spec.output_path)
    feedback_metrics = flatten_run_metrics(
        spec,
        rc,
        elapsed,
        benchmark_records,
        after_metrics,
        metric_delta,
    )
    fitness = compute_fitness(args, feedback_metrics)
    feedback = feedback_for_run(feedback_metrics)

    record = {
        "name": spec.name,
        "workload": spec.workload,
        "returncode": rc,
        "elapsed_sec": elapsed,
        "fitness": fitness,
        "feedback": feedback,
        "feedback_metrics": feedback_metrics,
        "cmd": spec.cmd,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "benchmark_output": str(spec.output_path) if spec.output_path else None,
        "profile_dir": str(profile_dir) if profile_dir else None,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "metrics_delta": metric_delta,
    }
    write_text(
        run_dir / "feedback_metrics.json",
        json.dumps(feedback_metrics, indent=2, sort_keys=True),
    )
    write_text(
        run_dir / "run_record.json",
        json.dumps(record, indent=2, sort_keys=True),
    )
    append_jsonl(meta_path, record)

    if rc == 0:
        print(f"[{spec.name}] done in {elapsed:.1f}s fitness={fitness:.6g}")
    else:
        print(
            f"[{spec.name}] failed with return code {rc}; "
            f"fitness={fitness:.6g}; see {run_dir}"
        )
    if not args.no_capture_metrics:
        print_metric_summary(spec.name, after_metrics, "metrics after")
        print_metric_summary(
            spec.name,
            metric_delta,
            "metrics delta",
            include_gauges=False,
        )
    return record


def workload_dataset(
    args: argparse.Namespace,
    workload: str,
) -> tuple[str, str]:
    if workload in {"serving-multiturn", "strata-longdoc-loogle"}:
        return "loogle", args.loogle_dataset_path
    if workload == "serving-shared-prefix":
        return "loogle", args.loogle_dataset_path
    return "", ""


def build_serving_workload(
    args: argparse.Namespace,
    suite_dir: Path,
    workload: str,
    mode: str,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    dataset_name, dataset_path = workload_dataset(args, workload)
    if not dataset_path:
        print(
            f"[skip] {workload} requires LooGLE under --data-dir "
            "(expected data/LooGLE/data/longdep_qa.jsonl)"
        )
        return []
    for rate in serving_rates(args):
        name = f"{workload}_{dataset_name}_r{rate:g}"
        output_file = suite_dir / f"{safe_name(name)}.jsonl"
        cmd = [
            sys.executable,
            "bench_serving.py",
            "--backend",
            "sglang",
            "--base-url",
            base_url(args),
            "--model",
            args.model,
            "--dataset-name",
            dataset_name,
            "--dataset-path",
            dataset_path,
            "--request-rate",
            str(rate),
            "--num-prompts",
            str(args.serving_num_prompts),
            "--output-file",
            str(output_file),
            "--flush-cache-timeout",
            str(args.flush_cache_timeout),
        ]
        if args.disable_child_progress:
            cmd.append("--disable-tqdm")
        if mode == "multiturn":
            cmd.append("--enable-multiturn")
        elif mode == "shared-prefix":
            cmd.append("--enable-shared-prefix")
        if args.serving_max_concurrency is not None:
            cmd.extend(["--max-concurrency", str(args.serving_max_concurrency)])
        if args.serving_fixed_output_len is not None:
            cmd.extend(["--fixed-output-len", str(args.serving_fixed_output_len)])
        if args.serving_max_prompt_len is not None:
            cmd.extend(["--max-prompt-len", str(args.serving_max_prompt_len)])
        if args.serving_disable_shuffle:
            cmd.append("--disable-shuffle")
        cmd.extend(split_extra(args.extra_serving_args))
        runs.append(
            RunSpec(name=name, workload=workload, cmd=cmd, output_path=output_file)
        )
    return runs


def build_synthetic_multiturn_runs(
    args: argparse.Namespace,
    suite_dir: Path,
    workload: str = "synthetic-multiturn",
    dataset_path: Optional[str] = None,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    num_clients = args.synthetic_clients
    if dataset_path:
        num_clients = dataset_multiround_client_count(args)
    for rate in synthetic_rates(args):
        name = f"{workload.replace('-', '_')}_r{rate:g}"
        output_file = suite_dir / f"{safe_name(name)}.jsonl"
        cmd = [
            sys.executable,
            "bench_multiturn.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model-path",
            args.model,
            "--num-clients",
            str(num_clients),
            "--num-rounds",
            str(args.synthetic_rounds),
            "--request-length",
            str(args.synthetic_request_length),
            "--output-length",
            str(args.synthetic_output_length),
            "--max-parallel",
            str(args.synthetic_max_parallel),
            "--request-rate",
            str(rate),
            "--disable-auto-run",
            "--log-file",
            str(output_file),
            "--tag",
            args.tag or name,
            "--api-format",
            args.synthetic_api_format,
        ]
        if args.synthetic_round_barrier:
            cmd.append("--enable-round-barrier")
        if args.synthetic_disable_random_sample:
            cmd.append("--disable-random-sample")
        if dataset_path:
            cmd.extend(["--dataset-path", dataset_path])
        cmd.extend(split_extra(args.extra_synthetic_args))
        runs.append(
            RunSpec(name=name, workload=workload, cmd=cmd, output_path=output_file)
        )
    return runs


def build_warm_cache_runs(args: argparse.Namespace, suite_dir: Path) -> List[RunSpec]:
    name = "warm_cache_prefix_sweep"
    output_file = suite_dir / f"{safe_name(name)}.jsonl"
    cmd = [
        sys.executable,
        "bench_warm_cache.py",
        "--base-url",
        base_url(args),
        "--model",
        args.model,
        "--num-prompts",
        str(args.warm_num_prompts),
        "--total-tokens",
        str(args.warm_total_tokens),
        "--output-len",
        str(args.warm_output_len),
        "--max-concurrency",
        str(args.warm_max_concurrency),
        "--pcts",
        args.warm_pcts,
        "--output-file",
        str(output_file),
        "--flush-cache-timeout",
        str(args.flush_cache_timeout),
    ]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    cmd.extend(split_extra(args.extra_warm_args))
    return [RunSpec(name=name, workload="warm-cache", cmd=cmd, output_path=output_file)]


def build_long_context_runs(args: argparse.Namespace, suite_dir: Path) -> List[RunSpec]:
    if not args.narrativeqa_dataset_path:
        print(
            "[skip] long-context requires NarrativeQA under --data-dir "
            "(expected data/narrativeqa_long_context.json)"
        )
        return []

    rates = long_context_rates(args)
    runs: List[RunSpec] = []
    for rate in rates:
        name = "long_context"
        if len(rates) > 1 or args.request_rates:
            name = f"{name}_r{rate:g}"
        output_file = suite_dir / f"{safe_name(name)}.jsonl"
        cmd = [
            sys.executable,
            "bench_long_context.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model-path",
            args.model,
            "--dataset-path",
            args.narrativeqa_dataset_path,
            "--num-clients",
            str(args.long_context_clients),
            "--request-rate",
            str(rate),
            "--disable-auto-run",
            "--log-file",
            str(output_file),
            "--tag",
            args.tag or name,
        ]
        if args.long_context_max_prompt_len is not None:
            cmd.extend(["--max-prompt-len", str(args.long_context_max_prompt_len)])
        cmd.extend(split_extra(args.extra_long_context_args))
        runs.append(
            RunSpec(name=name, workload="long-context", cmd=cmd, output_path=output_file)
        )
    return runs


def build_strata_narrativeqa_runs(
    args: argparse.Namespace, suite_dir: Path
) -> List[RunSpec]:
    if not args.narrativeqa_dataset_path:
        print(
            "[skip] strata-longdoc-narrativeqa requires "
            "NarrativeQA under --data-dir (expected data/narrativeqa_long_context.json)"
        )
        return []

    runs = build_long_context_runs(args, suite_dir)
    for run in runs:
        old_name = run.name
        new_name = run.name.replace("long_context", "strata_longdoc_narrativeqa")
        if run.output_path:
            old_output = str(run.output_path)
            new_output = suite_dir / f"{safe_name(new_name)}.jsonl"
            for idx, part in enumerate(run.cmd):
                if part == old_output:
                    run.cmd[idx] = str(new_output)
                if idx > 0 and run.cmd[idx - 1] == "--tag" and part == old_name:
                    run.cmd[idx] = new_name
            run.output_path = new_output
        run.name = new_name
        run.workload = "strata-longdoc-narrativeqa"
    return runs


def build_strata_reviewmt_runs(
    args: argparse.Namespace, suite_dir: Path
) -> List[RunSpec]:
    if not args.reviewmt_dataset_path:
        print(
            "[skip] strata-multiround-reviewmt requires "
            "ReviewMT under --data-dir (expected data/reviewmt_sharegpt.json)"
        )
        return []
    return build_synthetic_multiturn_runs(
        args,
        suite_dir,
        workload="strata-multiround-reviewmt",
        dataset_path=args.reviewmt_dataset_path,
    )


def expand_workloads(value: str) -> List[str]:
    selected = parse_csv(value)
    if not selected or "default" in selected:
        return list(DEFAULT_WORKLOADS)
    if "all" in selected:
        return list(ALL_WORKLOADS)

    aliases = {
        "serving": ["serving-multiturn", "serving-shared-prefix"],
        "cache-extra": list(EXTRA_CACHE_WORKLOADS),
        "cache": list(EXTRA_CACHE_WORKLOADS),
        "extra": list(EXTRA_CACHE_WORKLOADS),
        "shared-prefix": ["serving-shared-prefix"],
        "multiturn": ["synthetic-multiturn"],
        "synthetic": ["synthetic-multiturn"],
        "strata": list(DEFAULT_WORKLOADS),
    }
    expanded: List[str] = []
    for item in selected:
        for expanded_item in aliases.get(item, [item]):
            expanded.append(expanded_item)
    return expanded


def build_runs(args: argparse.Namespace, suite_dir: Path) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for workload in expand_workloads(args.workloads):
        if workload in {"serving-multiturn", "strata-longdoc-loogle"}:
            runs.extend(build_serving_workload(args, suite_dir, workload, "multiturn"))
        elif workload == "serving-shared-prefix":
            runs.extend(
                build_serving_workload(args, suite_dir, workload, "shared-prefix")
            )
        elif workload == "synthetic-multiturn":
            runs.extend(build_synthetic_multiturn_runs(args, suite_dir))
        elif workload == "strata-multiround-sharegpt":
            if not args.sharegpt_dataset_path:
                print(
                    "[skip] strata-multiround-sharegpt requires ShareGPT under "
                    "--data-dir (expected data/ShareGPT_V3_unfiltered_cleaned_split.json)"
                )
                continue
            runs.extend(
                build_synthetic_multiturn_runs(
                    args,
                    suite_dir,
                    workload="strata-multiround-sharegpt",
                    dataset_path=args.sharegpt_dataset_path,
                )
            )
        elif workload == "strata-multiround-reviewmt":
            runs.extend(build_strata_reviewmt_runs(args, suite_dir))
        elif workload == "strata-longdoc-narrativeqa":
            runs.extend(build_strata_narrativeqa_runs(args, suite_dir))
        elif workload == "warm-cache":
            runs.extend(build_warm_cache_runs(args, suite_dir))
        elif workload == "long-context":
            runs.extend(build_long_context_runs(args, suite_dir))
        else:
            raise ValueError(f"Unknown workload type: {workload}")
    return runs


def scale_count(
    value: int, fraction: float, cap: Optional[int], minimum: int = 1
) -> int:
    scaled = value
    if fraction < 1.0:
        scaled = max(minimum, math.ceil(scaled * fraction))
    if cap is not None:
        scaled = min(scaled, cap)
    return max(1, scaled)


def dataset_multiround_client_count(args: argparse.Namespace) -> int:
    cap = args.sample_count if args.sample_count > 0 else None
    fraction = getattr(args, "dataset_multiround_fraction", args.data_fraction)
    minimum = getattr(args, "dataset_multiround_min_clients", 1)
    if fraction < 1.0 or cap is not None:
        return scale_count(args.synthetic_clients, fraction, cap, minimum)
    return args.synthetic_clients


def normalize_args(args: argparse.Namespace) -> None:
    """Fill defaults, resolve dataset paths, and apply explicit sample scaling."""
    if args.fast:
        args.mode = "fast"

    if args.workloads is None:
        args.workloads = "strata"

    args.loogle_dataset_path = find_data_file(args.data_dir, "loogle")
    args.sharegpt_dataset_path = find_data_file(args.data_dir, "sharegpt")
    args.reviewmt_dataset_path = find_data_file(args.data_dir, "reviewmt")
    args.narrativeqa_dataset_path = find_data_file(args.data_dir, "narrativeqa")

    fast_default_fraction = args.mode == "fast" and args.data_fraction is None
    if args.data_fraction is None:
        args.data_fraction = FAST_DATA_FRACTION if args.mode == "fast" else 1.0

    args.serving_fraction = args.data_fraction
    args.warm_fraction = args.data_fraction
    args.long_context_fraction = (
        FAST_LONG_CONTEXT_FRACTION if fast_default_fraction else args.data_fraction
    )
    args.dataset_multiround_fraction = (
        FAST_DATASET_MULTITURN_FRACTION
        if fast_default_fraction
        else args.data_fraction
    )
    args.serving_min_prompts = FAST_MIN_SERVING_PROMPTS if fast_default_fraction else 1
    args.warm_min_prompts = FAST_MIN_WARM_PROMPTS if fast_default_fraction else 1
    args.long_context_min_clients = (
        FAST_MIN_LONG_CONTEXT_CLIENTS if fast_default_fraction else 1
    )
    args.dataset_multiround_min_clients = (
        FAST_MIN_DATASET_MULTITURN_CLIENTS if fast_default_fraction else 1
    )

    sample_cap = args.sample_count if args.sample_count > 0 else None
    fractions = [
        args.serving_fraction,
        args.warm_fraction,
        args.long_context_fraction,
        args.dataset_multiround_fraction,
    ]
    if any(not (0 < fraction <= 1.0) for fraction in fractions):
        raise ValueError("all data fractions must be in (0, 1]")
    if any(fraction < 1.0 for fraction in fractions) or sample_cap is not None:
        args.serving_num_prompts = scale_count(
            args.serving_num_prompts,
            args.serving_fraction,
            sample_cap,
            args.serving_min_prompts,
        )
        args.warm_num_prompts = scale_count(
            args.warm_num_prompts,
            args.warm_fraction,
            sample_cap,
            args.warm_min_prompts,
        )
        args.long_context_clients = scale_count(
            args.long_context_clients,
            args.long_context_fraction,
            sample_cap,
            args.long_context_min_clients,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HiCache workload types and capture server memory/cache metrics. "
            "Start the SGLang server separately."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--mode",
        choices=["full", "fast"],
        default="full",
        help="Use full benchmark defaults or a small smoke/evolution preset.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for --mode fast. Targets a few-minute feature check.",
    )
    parser.add_argument(
        "--workloads",
        default=None,
        help=(
            "Comma-separated workload types: strata-longdoc-loogle, "
            "strata-longdoc-narrativeqa, strata-multiround-sharegpt, "
            "strata-multiround-reviewmt, serving-shared-prefix, warm-cache, "
            "long-context, serving, strata, all, default"
        ),
    )
    parser.add_argument(
        "--request-rates",
        default="",
        help=(
            "Unified Poisson-arrival request-rate list in requests/sec. "
            "When set, it drives serving, shared-prefix, ShareGPT/ReviewMT "
            "multi-round, and NarrativeQA/long-context workloads. Accepts "
            "comma-separated values or start,stop,step."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for run logs, metrics snapshots, and JSONL outputs.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "Dataset root scanned for the standard all-in-one layout. Defaults "
            "to ./data in the current working directory."
        ),
    )
    parser.add_argument(
        "--session-name",
        default="",
        help="Stable name for this benchmark session. Defaults to timestamp.",
    )
    parser.add_argument(
        "--candidate-id",
        default="",
        help="Identifier for the KV-cache candidate/variant being evaluated.",
    )
    parser.add_argument(
        "--candidate-notes",
        default="",
        help="Free-form note saved into fitness_summary.json.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help=(
            "Scale sample counts for all workload families. Defaults to 1.0. "
            "Fast mode uses this as the LooGLE/warm-cache fraction, but uses "
            "larger built-in fractions for smaller NarrativeQA and "
            "ShareGPT/ReviewMT client populations. If you pass --data-fraction "
            "explicitly, that value applies to every scaled workload family. "
            "Generic synthetic-multiturn clients are not scaled; set "
            "--synthetic-clients explicitly when you want that generated "
            "workload smaller. "
            "Dataset-backed workloads sample randomly unless deterministic/"
            "no-shuffle flags are used."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help=(
            "Cap serving prompts, warm-cache prompts, and long-context clients. "
            "Also caps ShareGPT/ReviewMT dataset-backed multi-round clients. "
            "Does not cap generic synthetic-multiturn clients. 0 means no cap."
        ),
    )
    parser.add_argument(
        "--admin-api-key",
        default=os.environ.get("SGLANG_ADMIN_API_KEY", ""),
        help="Admin key for runtime HiCache storage endpoints.",
    )
    parser.add_argument(
        "--quiet-child-output",
        action="store_true",
        help="Write child benchmark stdout/stderr only to log files.",
    )
    parser.add_argument(
        "--disable-child-progress",
        action="store_true",
        help="Disable child benchmark progress bars when supported.",
    )
    parser.add_argument(
        "--no-flush-cache",
        action="store_true",
        help="Do not flush server KV cache before each workload run.",
    )
    parser.add_argument(
        "--flush-cache-timeout",
        type=float,
        default=60.0,
        help=(
            "Seconds to let /flush_cache wait for the server to become idle. "
            "Prevents transient 400 errors between back-to-back benchmark phases."
        ),
    )
    parser.add_argument("--post-flush-sleep", type=float, default=1.0)
    parser.add_argument(
        "--no-capture-metrics",
        action="store_true",
        help="Disable /metrics and /server_info snapshots.",
    )
    parser.add_argument("--metrics-timeout", type=float, default=10.0)

    l3 = parser.add_argument_group("L3 / storage HiCache options")
    l3.add_argument(
        "--l3-runtime-attach",
        action="store_true",
        help=(
            "Attach the storage backend through /hicache/storage-backend before "
            "running. Requires the server to be started with --admin-api-key."
        ),
    )
    l3.add_argument(
        "--l3-storage-backend",
        default="file",
        choices=["file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic", "simm"],
    )
    l3.add_argument(
        "--l3-storage-dir",
        default="/scratch/ows/alphacache/.hicache-file",
        help=(
            "Local file-backend directory to create/document. The running "
            "server must also have SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR set."
        ),
    )
    l3.add_argument("--l3-extra-config", default="{}")
    l3.add_argument(
        "--l3-prefetch-policy",
        default="best_effort",
        choices=["best_effort", "wait_complete", "timeout"],
    )
    l3.add_argument(
        "--l3-write-policy",
        default="write_through",
        choices=["write_back", "write_through", "write_through_selective"],
    )
    l3.add_argument(
        "--clear-l3-cache",
        action="store_true",
        help="Clear the storage backend once before the suite starts.",
    )
    l3.add_argument(
        "--clear-l3-cache-before-run",
        action="store_true",
        help="Clear the storage backend before each run.",
    )

    fitness = parser.add_argument_group("fitness and feedback options")
    fitness.add_argument(
        "--fitness-file",
        default="",
        help=(
            "Python file defining score(metrics) or fitness(metrics). "
            "The metrics argument is a FitnessMetrics object with dot access."
        ),
    )
    fitness.add_argument(
        "--fitness-expr",
        default="",
        help=(
            "Optional Python expression for higher-is-better fitness. "
            "Use aliases like ttft_ms, e2e_ms, cache_hit_rate, "
            "throughput_out_tok_s, host_kv_used_tokens, or m.get('name')."
        ),
    )
    fitness.add_argument(
        "--fitness-aggregate",
        choices=["mean", "min", "max", "sum"],
        default="mean",
        help="How to combine per-run fitness into the final session score.",
    )
    fitness.add_argument(
        "--fitness-fail-score",
        type=float,
        default=-1.0e12,
        help="Fitness assigned to failed runs.",
    )

    profile = parser.add_argument_group("profiling options")
    profile.add_argument(
        "--profile",
        action="store_true",
        help="Use SGLang /start_profile and /stop_profile around each workload.",
    )
    profile.add_argument(
        "--profile-output-dir",
        default="",
        help=(
            "Profiler trace root. Defaults to <run_dir>/profile; when set, "
            "each workload writes under <profile-output-dir>/<run-name>/."
        ),
    )
    profile.add_argument("--profile-start-step", type=int, default=None)
    profile.add_argument("--profile-num-steps", type=int, default=None)
    profile.add_argument(
        "--profile-activities",
        default="CPU,GPU",
        help="Comma-separated profiler activities, e.g. CPU,GPU or CPU,GPU,MEM.",
    )
    profile.add_argument("--profile-merge", action="store_true")
    profile.add_argument(
        "--profile-by-stage",
        action="store_true",
        help="Ask SGLang to emit separate prefill/decode stage profiles.",
    )
    profile.add_argument(
        "--profile-stages",
        default="",
        help="Comma-separated stages to profile when stage profiling is enabled.",
    )
    profile.add_argument(
        "--profile-with-stack",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override profiler stack capture.",
    )
    profile.add_argument(
        "--profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override profiler input-shape capture.",
    )

    serving = parser.add_argument_group("serving workload options")
    serving.add_argument("--serving-request-rates", default=DEFAULT_SERVING_REQUEST_RATES)
    serving.add_argument("--serving-num-prompts", type=int, default=DEFAULT_SERVING_NUM_PROMPTS)
    serving.add_argument("--serving-max-concurrency", type=int, default=None)
    serving.add_argument("--serving-fixed-output-len", type=int, default=None)
    serving.add_argument("--serving-max-prompt-len", type=int, default=None)
    serving.add_argument("--serving-disable-shuffle", action="store_true")
    serving.add_argument("--extra-serving-args", default="")

    synthetic = parser.add_argument_group("synthetic-multiturn options")
    synthetic.add_argument("--synthetic-request-rates", default=DEFAULT_SYNTHETIC_REQUEST_RATES)
    synthetic.add_argument("--synthetic-clients", type=int, default=DEFAULT_SYNTHETIC_CLIENTS)
    synthetic.add_argument("--synthetic-rounds", type=int, default=DEFAULT_SYNTHETIC_ROUNDS)
    synthetic.add_argument("--synthetic-request-length", type=int, default=DEFAULT_SYNTHETIC_REQUEST_LENGTH)
    synthetic.add_argument("--synthetic-output-length", type=int, default=DEFAULT_SYNTHETIC_OUTPUT_LENGTH)
    synthetic.add_argument("--synthetic-max-parallel", type=int, default=DEFAULT_SYNTHETIC_MAX_PARALLEL)
    synthetic.add_argument("--synthetic-round-barrier", action="store_true")
    synthetic.add_argument("--synthetic-disable-random-sample", action="store_true")
    synthetic.add_argument(
        "--synthetic-api-format",
        choices=["sglang", "openai"],
        default="sglang",
    )
    synthetic.add_argument("--extra-synthetic-args", default="")

    warm = parser.add_argument_group("warm-cache options")
    warm.add_argument("--warm-num-prompts", type=int, default=DEFAULT_WARM_NUM_PROMPTS)
    warm.add_argument("--warm-total-tokens", type=int, default=DEFAULT_WARM_TOTAL_TOKENS)
    warm.add_argument("--warm-output-len", type=int, default=DEFAULT_WARM_OUTPUT_LEN)
    warm.add_argument("--warm-max-concurrency", type=int, default=DEFAULT_WARM_MAX_CONCURRENCY)
    warm.add_argument("--warm-pcts", default=DEFAULT_WARM_PCTS)
    warm.add_argument("--extra-warm-args", default="")

    long_context = parser.add_argument_group("long-context options")
    long_context.add_argument("--long-context-clients", type=int, default=DEFAULT_LONG_CONTEXT_CLIENTS)
    long_context.add_argument("--long-context-request-rate", type=float, default=DEFAULT_LONG_CONTEXT_REQUEST_RATE)
    long_context.add_argument("--long-context-max-prompt-len", type=int, default=None)
    long_context.add_argument("--extra-long-context-args", default="")

    return parser.parse_args()


def run_suite(args: argparse.Namespace, session_id: str, suite_dir: Path) -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("SGLANG_TORCH_PROFILER_DIR", str(suite_dir / "profiles"))
    print(f"Writing results under {suite_dir}")
    print(f"Session id: {session_id}")
    print(f"Target server: {base_url(args)}")
    print(f"Data dir: {Path(args.data_dir).resolve()}")
    print(f"Expanded workloads: {','.join(expand_workloads(args.workloads))}")
    if args.request_rates:
        print(f"Unified request rates: {args.request_rates}")
    if not args.no_capture_metrics:
        print("Metrics capture: enabled")
    if args.mode == "fast":
        dataset_multiround_clients = dataset_multiround_client_count(args)
        print(
            "Fast mode: enabled "
            f"(workloads={args.workloads}, data_fraction={args.data_fraction}, "
            f"serving_fraction={args.serving_fraction}, "
            f"long_context_fraction={args.long_context_fraction}, "
            f"dataset_multiround_fraction={args.dataset_multiround_fraction}, "
            f"serving_num_prompts={args.serving_num_prompts}, "
            f"serving_max_concurrency={args.serving_max_concurrency}, "
            f"serving_max_prompt_len={args.serving_max_prompt_len}, "
            f"synthetic_clients={args.synthetic_clients}, "
            f"dataset_multiround_clients={dataset_multiround_clients}, "
            f"synthetic_rounds={args.synthetic_rounds}, "
            f"synthetic_request_length={args.synthetic_request_length}, "
            f"warm_num_prompts={args.warm_num_prompts}, "
            f"warm_total_tokens={args.warm_total_tokens}, "
            f"long_context_clients={args.long_context_clients}, "
            f"long_context_max_prompt_len={args.long_context_max_prompt_len})"
        )
        print(
            "Fast mode is a scaled smoke run, not the full Strata-style run: "
            f"NarrativeQA requests={args.long_context_clients}, "
            f"ShareGPT/ReviewMT clients={dataset_multiround_clients}, "
            f"ShareGPT/ReviewMT total turns={dataset_multiround_clients * args.synthetic_rounds}. "
            "Use --data-fraction 1.0 or remove --fast for full local counts."
        )
    print(f"L3 cache observation: enabled (backend={args.l3_storage_backend})")
    if args.profile:
        print(f"Profiling: enabled under {args.profile_output_dir or suite_dir / 'profiles'}")
    if not args.dry_run:
        preflight_server(args)
    prepare_l3_cache(args, suite_dir)

    runs = build_runs(args, suite_dir)
    if not runs:
        print("No benchmark runs selected.")
        return 1

    (suite_dir / "suite_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )

    attempted = 0
    succeeded = 0
    failed = 0
    records: List[dict] = []
    for spec in runs:
        attempted += 1
        print(f"\nStarting run {attempted}/{len(runs)}: {spec.name}")
        record = run_command(args, spec, suite_dir)
        records.append(record)
        rc = int(record["returncode"])
        if rc != 0:
            failed += 1
            if not args.continue_on_error:
                break
        else:
            succeeded += 1

    summary = build_session_summary(args, suite_dir, session_id, records)
    (suite_dir / "fitness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (suite_dir / "fitness.txt").write_text(
        f"{summary['fitness_score']:.12g}\n", encoding="utf-8"
    )
    write_report(summary, suite_dir / "report.md")

    print(f"Completed {succeeded}/{attempted} attempted runs ({len(runs)} planned).")
    print(f"Run index: {suite_dir / 'runs.jsonl'}")
    print(f"Fitness summary: {suite_dir / 'fitness_summary.json'}")
    print(f"Report: {suite_dir / 'report.md'}")
    print(f"Final fitness ({args.fitness_aggregate}): {summary['fitness_score']:.6g}")
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    normalize_args(args)
    session_id = args.session_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = Path(args.output_dir or f"bench_all_results/{session_id}").resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("runs.jsonl",):
        stale_path = suite_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    terminal_log_path = suite_dir / "terminal.log"
    args.terminal_log = str(terminal_log_path)
    with terminal_log_path.open(
        "w", encoding="utf-8", errors="replace", buffering=1
    ) as terminal_log:
        tee_stdout = TeeText(sys.stdout, terminal_log)
        tee_stderr = TeeText(sys.stderr, terminal_log)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(
            tee_stderr
        ):
            print(f"Terminal log: {terminal_log_path}")
            return run_suite(args, session_id, suite_dir)


if __name__ == "__main__":
    raise SystemExit(main())
