"""Exercise real Redis/Celery workers using a frozen trip from the app.

python scripts/check_route_jobs.py --request-file /tmp/trip.json
"""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(base_url, path, payload=None, *, timeout=15):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(base_url.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), time.monotonic() - started
    except HTTPError as exc:
        raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode()}") from exc


def inspect(base_url, job_id):
    _, job, _ = request_json(base_url, f"/routes/jobs/{job_id}")
    if job["status"] == "failed":
        raise RuntimeError(f"Job {job_id} failed: {job.get('error')}")
    return job


def wait_for_result(base_url, job_id, timeout):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        job = inspect(base_url, job_id)
        _, health, elapsed = request_json(base_url, "/health", timeout=5)
        if health.get("status") != "ok":
            raise RuntimeError("API health check failed during route calculation")
        print(f"job={job_id} status={job['status']} health_seconds={elapsed:.3f}", flush=True)
        if job["status"] == "completed":
            return job["result"], time.monotonic() - started
        time.sleep(2)
    raise RuntimeError(f"Job {job_id} did not finish within {timeout}s; check worker logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--primary-timeout", type=int, default=180)
    parser.add_argument("--baseline-timeout", type=int, default=3900)
    args = parser.parse_args()
    payload = json.loads(args.request_file.read_text(encoding="utf-8"))

    code, baseline, elapsed = request_json(args.base_url, "/routes/jobs", {**payload, "model": "baseline"})
    if code != 202:
        raise RuntimeError(f"Baseline submission must return 202, got {code}")
    print(f"baseline_submit_seconds={elapsed:.3f} job={baseline['job_id']}", flush=True)

    # Wait briefly for a real baseline worker to start before submitting ANTRoute.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        baseline_state = inspect(args.base_url, baseline["job_id"])
        if baseline_state["status"] != "queued":
            break
        time.sleep(1)
    else:
        raise RuntimeError("Baseline stayed queued for 30s. Ensure its worker is running and the queue is empty.")

    code, primary, elapsed = request_json(args.base_url, "/routes/jobs", {**payload, "model": "antroute"})
    if code != 202:
        raise RuntimeError(f"ANTRoute submission must return 202, got {code}")
    print(f"antroute_submit_seconds={elapsed:.3f} job={primary['job_id']}", flush=True)
    primary_result, elapsed = wait_for_result(args.base_url, primary["job_id"], args.primary_timeout)
    print(f"antroute_poll_wall_seconds={elapsed:.3f}", flush=True)
    routes = primary_result["routes"]
    if not routes or routes[0].get("algorithm") != "antroute":
        raise RuntimeError("ANTRoute did not return an actual graph route. Check scored risk data; placeholders are not thesis evidence.")
    baseline_state = inspect(args.base_url, baseline["job_id"])
    print(f"baseline_status_when_antroute_ready={baseline_state['status']}", flush=True)
    baseline_result, elapsed = wait_for_result(args.base_url, baseline["job_id"], args.baseline_timeout)
    print(f"baseline_remaining_poll_wall_seconds={elapsed:.3f}", flush=True)
    print(json.dumps({"antroute": primary_result, "baseline": baseline_result}, indent=2))
    if baseline_result["routes"]:
        print(f"baseline_algorithm={baseline_result['routes'][0].get('algorithm')}")
    else:
        print(f"baseline_completed_without_route={baseline_result.get('notice')}")
    print("PASS: real workers responded, API stayed reachable, and both job lifecycles completed.")


if __name__ == "__main__":
    main()
