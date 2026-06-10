import json
import os
import queue
import random
import time

import requests
from bench_multiturn import (
    ReadyQueue,
    WorkloadGenerator,
    gen_payload,
    log_to_jsonl_file,
    parse_args,
)
from tqdm.asyncio import tqdm

from sglang.benchmark.utils import get_tokenizer
from sglang.test.kits.cache_hit_kit import async_request_sglang_generate


class ContextWorkloadGenerator(WorkloadGenerator):
    def __init__(self, args):
        self.url = f"http://{args.host}:{args.port}/generate"
        self.request_func = async_request_sglang_generate

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.sent_requests = 0
        self.completed_requests = 0

        with open(args.dataset_path, "r", encoding="utf-8") as fin:
            self.dataset = json.load(fin)
        query_indices = list(range(len(self.dataset["queries"])))
        if not args.disable_random_sample:
            random.shuffle(query_indices)

        init_requests = []
        prompt_lengths = []
        skipped_too_long = 0
        for query_idx in query_indices:
            query = self.dataset["queries"][query_idx]
            context_id = query["context"]
            # Tokenize the context + question to get input_ids
            prompt_text = self.dataset["contexts"][context_id] + query["question"]
            input_ids = self.tokenizer.encode(prompt_text)
            if args.max_prompt_len > 0 and len(input_ids) > args.max_prompt_len:
                skipped_too_long += 1
                continue
            output_len = max(
                1,
                len(self.tokenizer(query["reference_answer"])["input_ids"]),
            )
            request_id = len(init_requests)
            init_requests.append((request_id, gen_payload(input_ids, output_len)))
            prompt_lengths.append(len(input_ids))
            if len(init_requests) >= args.num_clients:
                break

        if not init_requests:
            limit_msg = (
                f" <= --max-prompt-len {args.max_prompt_len}"
                if args.max_prompt_len > 0
                else ""
            )
            raise ValueError(
                f"No valid long-context requests{limit_msg}. "
                f"Scanned {len(query_indices)} queries and skipped "
                f"{skipped_too_long} over-length prompts."
            )

        if len(init_requests) < args.num_clients:
            print(
                "WARNING: requested "
                f"{args.num_clients} long-context requests but only selected "
                f"{len(init_requests)} after filtering."
            )
        avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths)
        print(
            "Selected "
            f"{len(init_requests)} long-context requests; "
            f"skipped_too_long={skipped_too_long}; "
            f"prompt_tokens min/avg/max="
            f"{min(prompt_lengths)}/{avg_prompt_len:.0f}/{max(prompt_lengths)}"
        )
        self.ready_queue = ReadyQueue(init_requests=init_requests)

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=len(init_requests))
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "itl": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }
        self.failed_requests = 0

        self.max_parallel = args.max_parallel
        self.logfile = args.log_file
        self.enable_round_barrier = False

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    print(f"Request failed for client {client_id}: {response.error}")
                    self.failed_requests += 1
                    self.completed_requests += 1
                    continue
                if response.prompt_len <= 0:
                    print(
                        f"Request failed for client {client_id}: "
                        "server returned success with zero prompt tokens"
                    )
                    self.failed_requests += 1
                    self.completed_requests += 1
                    continue
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(response.itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.performance_metrics["generated_len"].append(response.generated_len)
                self.completed_requests += 1

            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break

    def run(self):
        performance_data = super().run()
        successful = performance_data["summary"]["total_requests"]
        performance_data["summary"]["failed_requests"] = self.failed_requests
        if successful == 0 or sum(self.performance_metrics["prompt_len"]) == 0:
            raise RuntimeError(
                "No successful long-context requests produced prompt tokens. "
                "The server probably rejected every prompt as over-length; "
                "check --max-prompt-len, --context-length, and server stdout."
            )
        return performance_data


def admin_headers():
    token = os.environ.get("SGLANG_ADMIN_API_KEY", "")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def flush_cache(url: str, timeout_s: float = 60.0) -> None:
    response = requests.post(
        url,
        params={"timeout": timeout_s},
        headers=admin_headers(),
        timeout=timeout_s + 30.0,
    )
    if response.status_code != 200:
        print(f"WARNING: flush_cache failed: {response.status_code} {response.text[:160]}")


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    args.num_rounds = 1
    args.max_parallel = 24
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    if args.disable_auto_run:
        request_rates = [args.request_rate]
    else:
        request_rates = [24, 16, 12, 8, 4, 2, 1]

    for request_rate in request_rates:
        args.request_rate = request_rate
        flush_cache(flush_cache_url)
        time.sleep(1)
        performance_data = ContextWorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, args.tag)
