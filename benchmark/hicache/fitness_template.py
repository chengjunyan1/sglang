"""Example fitness function for bench_all_in_one.py.

The benchmark runner calls score(metrics) once per workload run. The final
session score is then aggregated with --fitness-aggregate.

Higher is better. Edit this file for the objective you want an evolution loop
to optimize.
"""


def score(metrics):
    # Latency metrics are milliseconds. Missing metrics return 0 by default
    # through the FitnessMetrics helper used by bench_all_in_one.py.
    ttft_ms = metrics.get("ttft_ms", 100000.0)
    e2e_ms = metrics.get("e2e_ms", 100000.0)
    throughput = metrics.get("throughput_out_tok_s", 0.0)
    cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
    host_hits = metrics.get("cached_host_tokens", 0.0)
    storage_hits = metrics.get("cached_storage_file_tokens", 0.0)
    evicted = metrics.get("evicted_tokens", 0.0)
    aborted = metrics.get("aborted_requests", 0.0)

    return (
        1000.0 * cache_hit_rate
        + 0.05 * throughput
        + 0.00002 * host_hits
        + 0.00002 * storage_hits
        - 0.10 * ttft_ms
        - 0.02 * e2e_ms
        - 0.00001 * evicted
        - 1000.0 * aborted
    )
