#!/usr/bin/env python3
"""
Stress-test an Ollama evaluation endpoint.
Fires concurrent judge requests and reports latency & throughput stats.
"""

import argparse
import json
import statistics
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_chat_url(ip_or_host):
    ip_or_host = ip_or_host.strip()
    if ip_or_host.startswith("http://") or ip_or_host.startswith("https://"):
        return ip_or_host
    return f"http://{ip_or_host}:11434/api/chat"


def make_judge_request(prompt, model, url):
    """Send a single chat request to Ollama and return (duration_sec, success_bool, response_text)."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
        },
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    start = time.perf_counter()
    response_text = ""
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_text += response_json.get("message", {}).get("content", "")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return elapsed, False, str(exc)

    elapsed = time.perf_counter() - start
    return elapsed, True, response_text


def build_prompt_from_entry(entry):
    """Build a realistic judge prompt from an instruction-data entry."""
    instruction = entry["instruction"]
    reference = entry.get("output", "")
    model_answer = entry.get("model_response", "")

    rubric = (
        "You are a fair judge assistant. Score the model response on a scale of 1 to 5.\n"
        "1: Fails to address instructions.\n"
        "5: Fully adheres with clear, accurate answer.\n\n"
        "Provide ONLY the integer score."
    )

    prompt = (
        f"{rubric}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Reference Answer:\n{reference}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        "Score:"
    )
    return prompt


def run_stress_test(url, model, workers, total_requests, prompts):
    """Fire `total_requests` judge requests with `workers` concurrency."""
    latencies = []
    errors = 0
    results = []

    # Cycle through prompts if we have fewer than total_requests
    def get_prompt(i):
        return prompts[i % len(prompts)]

    start_global = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(make_judge_request, get_prompt(i), model, url): i
            for i in range(total_requests)
        }

        completed = 0
        for future in as_completed(futures):
            duration, ok, text = future.result()
            latencies.append(duration)
            if not ok:
                errors += 1
            results.append((duration, ok, text))
            completed += 1
            if completed % 5 == 0 or completed == total_requests:
                print(f"  Progress: {completed}/{total_requests} done")

    total_elapsed = time.perf_counter() - start_global
    return latencies, errors, total_elapsed, results


def print_report(latencies, errors, total_elapsed, total_requests, workers):
    successful = total_requests - errors
    print("\n" + "=" * 60)
    print("OLLAMA STRESS TEST REPORT")
    print("=" * 60)
    print(f"Total requests : {total_requests}")
    print(f"Successful     : {successful}")
    print(f"Errors         : {errors}")
    print(f"Concurrency    : {workers}")
    print(f"Total time     : {total_elapsed:.2f} s")
    print(f"Throughput     : {successful / total_elapsed:.2f} req/s")

    if not latencies:
        print("No successful requests to report.")
        return

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[int(n * 0.95)] if n > 1 else latencies[0]
    p99 = latencies[int(n * 0.99)] if n > 1 else latencies[0]

    print("\n--- Latency (seconds) ---")
    print(f"  Min  : {min(latencies):.3f}")
    print(f"  Max  : {max(latencies):.3f}")
    print(f"  Mean : {statistics.mean(latencies):.3f}")
    print(f"  Med  : {statistics.median(latencies):.3f}")
    print(f"  P95  : {p95:.3f}")
    print(f"  P99  : {p99:.3f}")
    print("=" * 60)


def main(args):
    url = build_chat_url(args.url)
    print(f"Target Ollama URL: {url}")
    print(f"Model            : {args.model}")
    print(f"Workers          : {args.workers}")
    print(f"Total requests   : {args.total}")
    print("-" * 60)

    # Load prompts from file if provided, otherwise use a default hard-coded prompt
    if args.file_path:
        with open(args.file_path, "r") as f:
            data = json.load(f)
        # Limit to first N unique prompts to keep memory low
        prompts = [build_prompt_from_entry(e) for e in data[: min(len(data), args.total)]]
        if not prompts:
            raise ValueError("No entries found in the provided JSON file.")
    else:
        prompts = [
            (
                "Score the following answer on a scale of 1 to 5.\n"
                "Question: What is the capital of France?\n"
                "Answer: Paris\n\n"
                "Provide ONLY the integer score.\nScore:"
            )
        ]

    latencies, errors, total_elapsed, _ = run_stress_test(
        url=url,
        model=args.model,
        workers=args.workers,
        total_requests=args.total,
        prompts=prompts,
    )

    print_report(latencies, errors, total_elapsed, args.total, args.workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stress-test an Ollama server with concurrent evaluation requests."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="10.16.76.75",
        help="IP or hostname of the Ollama server (default: 10.16.76.75)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="Judge model name served by Ollama",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=20,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to instruction-data JSON for realistic prompts (optional)",
    )
    args = parser.parse_args()
    main(args)
