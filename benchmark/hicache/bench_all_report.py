"""Session summaries and Markdown reports for bench_all_in_one.py."""

import argparse
import json
import math
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def aggregate_fitness(args: argparse.Namespace, records: List[dict]) -> float:
    scores = [
        float(record["fitness"])
        for record in records
        if isinstance(record.get("fitness"), (int, float))
    ]
    if not scores:
        return args.fitness_fail_score
    if args.fitness_aggregate == "min":
        return min(scores)
    if args.fitness_aggregate == "max":
        return max(scores)
    if args.fitness_aggregate == "sum":
        return sum(scores)
    return sum(scores) / len(scores)


def collect_top_metrics(records: List[dict]) -> Dict[str, float]:
    """Aggregate a compact set of feedback metrics across successful runs."""
    totals: Dict[str, float] = {}
    maxes: Dict[str, float] = {}
    successful = [
        record
        for record in records
        if record.get("returncode") == 0
        and isinstance(record.get("feedback_metrics"), dict)
    ]
    for record in successful:
        metrics = record["feedback_metrics"]
        for key in [
            "prompt_tokens",
            "generation_tokens",
            "requests",
            "aborted_requests",
            "cached_device_tokens",
            "cached_host_tokens",
            "cached_storage_file_tokens",
            "cached_storage_mooncake_tokens",
            "cached_storage_hf3fs_tokens",
            "prefetched_tokens",
            "backuped_tokens",
            "evicted_tokens",
            "load_back_tokens",
        ]:
            if key in metrics:
                totals[key] = totals.get(key, 0.0) + float(metrics[key])
        for key in [
            "ttft_ms",
            "e2e_ms",
            "itl_ms",
            "throughput_req_s",
            "throughput_out_tok_s",
            "cache_hit_rate",
            "gpu_kv_used_tokens",
            "host_kv_used_tokens",
        ]:
            if key in metrics:
                maxes[key] = max(maxes.get(key, float("-inf")), float(metrics[key]))

    merged = {f"sum.{key}": value for key, value in totals.items()}
    merged.update({f"max.{key}": value for key, value in maxes.items()})
    return merged


def read_l3_config(suite_dir: Path) -> dict:
    path = suite_dir / "l3_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def cmd_option(cmd: List[str], option: str, default: str = "") -> str:
    for idx, part in enumerate(cmd):
        if part == option and idx + 1 < len(cmd):
            return str(cmd[idx + 1])
        prefix = f"{option}="
        if part.startswith(prefix):
            return part[len(prefix) :]
    return default


def run_config(record: dict) -> Dict[str, str]:
    cmd = record.get("cmd") or []
    workload = str(record.get("workload") or "")
    dataset_name = cmd_option(cmd, "--dataset-name")
    dataset_path = cmd_option(cmd, "--dataset-path")
    if not dataset_name:
        if dataset_path:
            dataset_name = Path(dataset_path).stem
        elif workload == "warm-cache":
            dataset_name = "synthetic-token-prefix"
        elif "multiround" in workload or "synthetic" in workload:
            dataset_name = "synthetic-or-sharegpt"
        else:
            dataset_name = "none"

    config = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "request_rate": cmd_option(
            cmd, "--request-rate", "inf" if workload == "warm-cache" else ""
        ),
        "num_prompts": cmd_option(cmd, "--num-prompts"),
        "num_clients": cmd_option(cmd, "--num-clients"),
        "num_rounds": cmd_option(cmd, "--num-rounds"),
        "request_length": cmd_option(cmd, "--request-length"),
        "output_length": cmd_option(cmd, "--output-length")
        or cmd_option(cmd, "--output-len"),
        "total_tokens": cmd_option(cmd, "--total-tokens"),
        "max_concurrency": cmd_option(cmd, "--max-concurrency")
        or cmd_option(cmd, "--max-parallel"),
        "shared_prefix_pcts": cmd_option(cmd, "--pcts"),
    }
    return {key: value for key, value in config.items() if value}


def metric_value(record: dict, key: str) -> Optional[float]:
    metrics = record.get("feedback_metrics") or {}
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarize_records(records: List[dict]) -> Dict[str, float]:
    successful = [record for record in records if record.get("returncode") == 0]
    summary: Dict[str, float] = {
        "runs": float(len(records)),
        "succeeded": float(len(successful)),
        "failed": float(len(records) - len(successful)),
    }

    average_keys = [
        "fitness",
        "elapsed_sec",
        "ttft_ms",
        "e2e_ms",
        "itl_ms",
        "throughput_req_s",
        "throughput_out_tok_s",
        "cache_hit_rate",
        "gpu_kv_used_tokens",
        "host_kv_used_tokens",
        "host_kv_total_tokens",
    ]
    for key in average_keys:
        values = []
        for record in successful:
            if key == "fitness":
                value = record.get("fitness")
            elif key == "elapsed_sec":
                value = record.get("elapsed_sec")
            else:
                value = metric_value(record, key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        avg = mean(values)
        if avg is not None:
            summary[f"avg.{key}"] = avg
            summary[f"min.{key}"] = min(values)
            summary[f"max.{key}"] = max(values)

    sum_keys = [
        "prompt_tokens",
        "generation_tokens",
        "requests",
        "aborted_requests",
        "cached_device_tokens",
        "cached_host_tokens",
        "cached_storage_file_tokens",
        "cached_storage_mooncake_tokens",
        "cached_storage_hf3fs_tokens",
        "prefetched_tokens",
        "backuped_tokens",
        "evicted_tokens",
        "load_back_tokens",
    ]
    for key in sum_keys:
        values = [
            metric_value(record, key)
            for record in successful
            if metric_value(record, key) is not None
        ]
        if values:
            summary[f"sum.{key}"] = float(sum(values))

    return summary


def group_summaries(records: List[dict], group_by) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[dict]] = {}
    for record in records:
        key = group_by(record) or "unknown"
        groups.setdefault(str(key), []).append(record)
    return {
        key: summarize_records(group_records)
        for key, group_records in sorted(groups.items())
    }


def dashboard_summary(records: List[dict]) -> dict:
    return {
        "overall": summarize_records(records),
        "by_workload": group_summaries(
            records, lambda record: record.get("workload") or "unknown"
        ),
        "by_dataset": group_summaries(
            records, lambda record: run_config(record).get("dataset_name", "none")
        ),
        "by_workload_and_dataset": group_summaries(
            records,
            lambda record: (
                f"{record.get('workload') or 'unknown'}::"
                f"{run_config(record).get('dataset_name', 'none')}"
            ),
        ),
    }


def build_session_summary(
    args: argparse.Namespace,
    suite_dir: Path,
    session_id: str,
    records: List[dict],
) -> dict:
    succeeded = sum(1 for record in records if record.get("returncode") == 0)
    failed = len(records) - succeeded
    best = max(records, key=lambda item: item.get("fitness", float("-inf")), default=None)
    summary = {
        "schema_version": 1,
        "session_id": session_id,
        "candidate_id": args.candidate_id,
        "candidate_notes": args.candidate_notes,
        "mode": args.mode,
        "workloads": args.workloads,
        "request_rates": args.request_rates,
        "data_dir": args.data_dir,
        "data_fraction": args.data_fraction,
        "sample_count": args.sample_count,
        "l3_storage_backend": args.l3_storage_backend,
        "l3_config": read_l3_config(suite_dir),
        "terminal_log": getattr(args, "terminal_log", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite_dir": str(suite_dir),
        "fitness_higher_is_better": True,
        "fitness_aggregate": args.fitness_aggregate,
        "fitness_file": args.fitness_file,
        "fitness_expr": args.fitness_expr or "",
        "fitness_function": args.fitness_file
        or args.fitness_expr
        or "default_fitness(metrics)",
        "fitness_score": aggregate_fitness(args, records),
        "num_runs": len(records),
        "succeeded": succeeded,
        "failed": failed,
        "dashboard": dashboard_summary(records),
        "best_run": {
            "name": best.get("name"),
            "workload": best.get("workload"),
            "fitness": best.get("fitness"),
        }
        if best
        else None,
        "aggregate_feedback_metrics": collect_top_metrics(records),
        "runs": [
            {
                "name": record.get("name"),
                "workload": record.get("workload"),
                "returncode": record.get("returncode"),
                "elapsed_sec": record.get("elapsed_sec"),
                "fitness": record.get("fitness"),
                "feedback": record.get("feedback"),
                "feedback_metrics": record.get("feedback_metrics"),
                "config": run_config(record),
                "cmd": record.get("cmd"),
                "benchmark_output": record.get("benchmark_output"),
                "profile_dir": record.get("profile_dir"),
                "stdout": record.get("stdout"),
                "stderr": record.get("stderr"),
            }
            for record in records
        ],
    }
    return summary


def fmt_number(value: object, precision: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return ""
    value = float(value)
    if abs(value) >= 10000 or (0 < abs(value) < 0.001):
        return f"{value:.{precision}g}"
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def fmt_bytes(value: object) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return ""
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return ""


def markdown_table(headers: List[str], rows: List[List[object]]) -> List[str]:
    if not rows:
        return ["No data.", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        rendered = []
        for value in row:
            text = fmt_number(value) if isinstance(value, (int, float)) else str(value or "")
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    lines.append("")
    return lines


def summary_row(name: str, stats: Dict[str, float]) -> List[object]:
    return [
        name,
        int(stats.get("runs", 0)),
        int(stats.get("succeeded", 0)),
        int(stats.get("failed", 0)),
        stats.get("avg.fitness", ""),
        stats.get("avg.ttft_ms", ""),
        stats.get("avg.e2e_ms", ""),
        stats.get("avg.throughput_req_s", ""),
        stats.get("avg.throughput_out_tok_s", ""),
        stats.get("avg.cache_hit_rate", ""),
        stats.get("sum.prompt_tokens", ""),
        stats.get("sum.cached_device_tokens", ""),
        stats.get("sum.cached_host_tokens", ""),
        stats.get("sum.cached_storage_file_tokens", ""),
        stats.get("sum.load_back_tokens", ""),
    ]


def write_report(summary: dict, path: Path) -> None:
    dashboard = summary.get("dashboard") or {}
    overall = dashboard.get("overall") or {}
    lines = [
        "# HiCache Benchmark Report",
        "",
        "## Overview",
        "",
        f"- session_id: `{summary['session_id']}`",
        f"- candidate_id: `{summary.get('candidate_id') or ''}`",
        f"- fitness_score: `{summary['fitness_score']:.6g}`",
        f"- succeeded/failed: `{summary['succeeded']}/{summary['failed']}`",
        f"- aggregate: `{summary['fitness_aggregate']}`",
        f"- mode: `{summary.get('mode')}`",
        f"- workloads: `{summary.get('workloads')}`",
        f"- request_rates: `{summary.get('request_rates') or ''}`",
        f"- data_dir: `{summary.get('data_dir')}`",
        f"- l3_storage_backend: `{summary.get('l3_storage_backend')}`",
        f"- data_fraction: `{summary.get('data_fraction')}`",
        f"- sample_count: `{summary.get('sample_count')}`",
        f"- output_dir: `{summary.get('suite_dir')}`",
        f"- terminal_log: `{summary.get('terminal_log') or ''}`",
        "",
        "## L3 Storage",
        "",
    ]

    l3_config = summary.get("l3_config") or {}
    l3_status = (
        l3_config.get("status_final")
        or l3_config.get("status_after")
        or l3_config.get("status_before")
        or {}
    )
    l3_stats = (
        l3_config.get("file_storage_stats_final")
        or l3_config.get("file_storage_stats_after_clear")
        or l3_config.get("file_storage_stats_before")
        or {}
    )
    if l3_config:
        lines.extend(
            [
                f"- backend: `{l3_status.get('hicache_storage_backend') or summary.get('l3_storage_backend')}`",
                f"- prefetch_policy: `{l3_status.get('hicache_storage_prefetch_policy') or ''}`",
                f"- write_policy: `{l3_status.get('hicache_write_policy') or ''}`",
                f"- file_storage_dir: `{l3_config.get('file_storage_dir') or l3_stats.get('path') or ''}`",
                f"- clear_before_suite: `{l3_config.get('clear_before_suite')}`",
                f"- clear_before_run: `{l3_config.get('clear_before_run')}`",
                f"- filesystem_used: `{fmt_bytes(l3_stats.get('fs_used_bytes'))}`",
                f"- filesystem_available: `{fmt_bytes(l3_stats.get('fs_available_bytes'))}`",
                f"- filesystem_used_pct: `{fmt_number((l3_stats.get('fs_used_pct') or 0) * 100)}%`",
                "",
            ]
        )
        notes = l3_config.get("notes") or []
        if notes:
            lines.extend(["Notes:"])
            lines.extend([f"- {note}" for note in notes])
            lines.append("")
    else:
        lines.extend(["No L3 config captured.", ""])

    lines.extend(
        [
        "## Best Run",
        "",
        ]
    )
    best = summary.get("best_run")
    if best:
        lines.extend(
            [
                f"- name: `{best['name']}`",
                f"- workload: `{best['workload']}`",
                f"- fitness: `{best['fitness']:.6g}`",
                "",
            ]
        )
    else:
        lines.extend(["No run records.", ""])

    lines.extend(["## Dashboard", "", "### Overall", ""])
    headers = [
        "scope",
        "runs",
        "ok",
        "fail",
        "avg fitness",
        "avg TTFT ms",
        "avg E2E ms",
        "avg req/s",
        "avg out tok/s",
        "avg cache hit",
        "sum prompt tok",
        "sum device hits",
        "sum host hits",
        "sum file hits",
        "sum load back",
    ]
    lines.extend(markdown_table(headers, [summary_row("all", overall)]))

    lines.extend(["### By Workload", ""])
    lines.extend(
        markdown_table(
            ["workload", *headers[1:]],
            [
                summary_row(name, stats)
                for name, stats in (dashboard.get("by_workload") or {}).items()
            ],
        )
    )

    lines.extend(["### By Dataset", ""])
    lines.extend(
        markdown_table(
            ["dataset", *headers[1:]],
            [
                summary_row(name, stats)
                for name, stats in (dashboard.get("by_dataset") or {}).items()
            ],
        )
    )

    lines.extend(["### By Workload And Dataset", ""])
    lines.extend(
        markdown_table(
            ["workload::dataset", *headers[1:]],
            [
                summary_row(name, stats)
                for name, stats in (
                    dashboard.get("by_workload_and_dataset") or {}
                ).items()
            ],
        )
    )

    lines.extend(["## Run Matrix", ""])
    run_rows = []
    for run in summary["runs"]:
        metrics = run.get("feedback_metrics") or {}
        config = run.get("config") or {}
        run_rows.append(
            [
                run.get("name"),
                run.get("workload"),
                config.get("dataset_name"),
                config.get("request_rate"),
                config.get("num_prompts") or config.get("num_clients"),
                run.get("returncode"),
                run.get("fitness"),
                metrics.get("ttft_ms"),
                metrics.get("e2e_ms"),
                metrics.get("throughput_req_s"),
                metrics.get("cache_hit_rate"),
                metrics.get("cached_device_tokens"),
                metrics.get("cached_host_tokens"),
                metrics.get("cached_storage_file_tokens"),
                metrics.get("load_back_tokens"),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "run",
                "workload",
                "dataset",
                "rate",
                "samples",
                "rc",
                "fitness",
                "TTFT ms",
                "E2E ms",
                "req/s",
                "cache hit",
                "device hits",
                "host hits",
                "file hits",
                "load back",
            ],
            run_rows,
        )
    )

    lines.extend(["## Aggregate Metrics", ""])
    for key, value in sorted(summary["aggregate_feedback_metrics"].items()):
        lines.append(f"- `{key}`: `{value:.6g}`")

    lines.extend(["", "## Run Details", ""])
    for run in summary["runs"]:
        lines.append(f"### {run['name']}  fitness=`{float(run['fitness']):.6g}`")
        lines.append(f"- workload: `{run.get('workload')}`")
        lines.append(f"- returncode: `{run.get('returncode')}`")
        lines.append(f"- elapsed_sec: `{fmt_number(run.get('elapsed_sec'))}`")
        config = run.get("config") or {}
        if config:
            lines.append("- config:")
            for key, value in config.items():
                lines.append(f"  - `{key}`: `{value}`")
        if run.get("benchmark_output"):
            lines.append(f"- benchmark_output: `{run['benchmark_output']}`")
        if run.get("stdout"):
            lines.append(f"- stdout: `{run['stdout']}`")
        if run.get("stderr"):
            lines.append(f"- stderr: `{run['stderr']}`")
        if run.get("profile_dir"):
            lines.append(f"- profile_dir: `{run['profile_dir']}`")
        if run.get("cmd"):
            rendered_cmd = " ".join(shlex.quote(str(part)) for part in run["cmd"])
            lines.extend(["- command:", "", "```bash", rendered_cmd, "```"])
        for item in run.get("feedback") or []:
            lines.append(f"- {item}")
        metrics = run.get("feedback_metrics") or {}
        compact_keys = [
            "ttft_ms",
            "e2e_ms",
            "throughput_req_s",
            "throughput_out_tok_s",
            "cache_hit_rate",
            "gpu_kv_used_tokens",
            "host_kv_used_tokens",
            "cached_device_tokens",
            "cached_host_tokens",
            "cached_storage_file_tokens",
            "cached_storage_mooncake_tokens",
            "cached_storage_hf3fs_tokens",
            "prefetched_tokens",
            "backuped_tokens",
            "evicted_tokens",
            "load_back_tokens",
            "aborted_requests",
        ]
        for key in compact_keys:
            if key in metrics:
                lines.append(f"- `{key}`: `{float(metrics[key]):.6g}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
