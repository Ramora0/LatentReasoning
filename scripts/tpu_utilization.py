#!/usr/bin/env python3
"""Show memory and compute utilization for all TPU VMs in a GCP project.

Uses the Cloud Monitoring API (via gcloud auth) to query TPU metrics,
and gcloud CLI to list TPU VMs. No extra pip packages required.

Usage:
    python scripts/tpu_utilization.py                          # all zones
    python scripts/tpu_utilization.py --zone us-central2-b     # specific zone
    python scripts/tpu_utilization.py --watch 5                # refresh every 5s
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

# TPU zones to scan if none specified
TPU_ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-c",
    "us-central2-b",
    "us-east1-d", "us-east5-a", "us-east5-b", "us-east5-c",
    "europe-west4-a", "europe-west4-b",
    "asia-east1-c",
]


def run_cmd(cmd, timeout=30):
    """Run a shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_project():
    return run_cmd(["gcloud", "config", "get-value", "project"])


def get_access_token():
    return run_cmd(["gcloud", "auth", "print-access-token"])


def list_tpus(zones=None):
    """List all TPU VMs across specified zones."""
    tpus = []
    zones = zones or TPU_ZONES
    for zone in zones:
        out = run_cmd([
            "gcloud", "compute", "tpus", "tpu-vm", "list",
            "--zone", zone, "--format=json"
        ])
        if out:
            try:
                for tpu in json.loads(out):
                    tpu["_zone"] = zone
                    tpus.append(tpu)
            except json.JSONDecodeError:
                pass
    return tpus


def query_monitoring(project, access_token, metric_type, minutes=5):
    """Query Cloud Monitoring API for a TPU metric."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    url = (
        f"https://monitoring.googleapis.com/v3/projects/{project}"
        f"/timeSeries"
        f"?filter=metric.type%3D%22{metric_type}%22"
        f"&interval.startTime={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&interval.endTime={now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&aggregation.alignmentPeriod=60s"
        f"&aggregation.perSeriesAligner=ALIGN_MEAN"
    )

    out = run_cmd([
        "curl", "-s", "-H", f"Authorization: Bearer {access_token}", url
    ], timeout=15)
    if out:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
    return None


def get_tpu_metrics_via_ssh(tpu_name, zone):
    """SSH into TPU VM and fetch metrics from local Prometheus endpoint."""
    out = run_cmd([
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name,
        "--zone", zone, "--worker=all",
        "--command", "curl -s http://localhost:8431/metrics 2>/dev/null || echo NO_METRICS",
    ], timeout=30)
    return out


def parse_prometheus_metrics(raw):
    """Parse Prometheus text format into a dict of metric_name -> value."""
    metrics = {}
    if not raw:
        return metrics
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].split("{")[0]  # strip labels
            try:
                val = float(parts[-1])
                if name in metrics:
                    if isinstance(metrics[name], list):
                        metrics[name].append(val)
                    else:
                        metrics[name] = [metrics[name], val]
                else:
                    metrics[name] = val
            except ValueError:
                pass
    return metrics


def fmt_bytes(b):
    """Format bytes to human-readable."""
    if b is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def aggregate(val):
    """If val is a list, return the mean; otherwise return val."""
    if isinstance(val, list):
        return sum(val) / len(val) if val else None
    return val


def display_tpu_info(tpus):
    """Display basic TPU info table."""
    print(f"\n{'='*80}")
    print(f"  TPU VMs — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"  {'Name':<25} {'Zone':<20} {'Type':<12} {'Status':<10}")
    print(f"  {'-'*25} {'-'*20} {'-'*12} {'-'*10}")
    for tpu in tpus:
        name = tpu.get("name", "?").split("/")[-1]
        zone = tpu.get("_zone", "?")
        accel = tpu.get("acceleratorType", "?")
        state = tpu.get("state", "?")
        print(f"  {name:<25} {zone:<20} {accel:<12} {state:<10}")
    print()


def display_monitoring_metrics(project, token, tpus):
    """Query and display Cloud Monitoring TPU metrics."""
    metrics_to_query = [
        ("tpu.googleapis.com/cpu/utilization", "CPU Utilization"),
        ("tpu.googleapis.com/memory/usage", "Host Memory Used"),
        ("tpu.googleapis.com/accelerator/duty_cycle", "TPU Duty Cycle"),
        ("tpu.googleapis.com/accelerator/memory_usage", "HBM Used (bytes)"),
        ("tpu.googleapis.com/accelerator/memory_total", "HBM Total (bytes)"),
    ]

    results = {}
    for metric_type, label in metrics_to_query:
        data = query_monitoring(project, token, metric_type)
        if data and "timeSeries" in data:
            for series in data["timeSeries"]:
                resource = series.get("resource", {}).get("labels", {})
                node_id = resource.get("node_id", "unknown")
                zone = resource.get("zone", "")
                key = f"{node_id} ({zone})"
                if key not in results:
                    results[key] = {}
                points = series.get("points", [])
                if points:
                    val = points[0].get("value", {})
                    v = (
                        val.get("doubleValue")
                        or val.get("int64Value")
                        or val.get("distribution", {}).get("mean")
                    )
                    if v is not None:
                        results[key][label] = float(v)

    if results:
        print("  Cloud Monitoring Metrics (last 5 min avg):")
        print(f"  {'-'*76}")
        for node, metrics in sorted(results.items()):
            print(f"\n  Node: {node}")
            cpu = metrics.get("CPU Utilization")
            duty = metrics.get("TPU Duty Cycle")
            mem_used = metrics.get("HBM Used (bytes)")
            mem_total = metrics.get("HBM Total (bytes)")
            host_mem = metrics.get("Host Memory Used")

            if cpu is not None:
                print(f"    CPU Utilization:    {fmt_pct(cpu * 100)}")
            if duty is not None:
                print(f"    TPU Duty Cycle:     {fmt_pct(duty)}")
            if mem_used is not None and mem_total is not None:
                pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                print(f"    HBM Memory:         {fmt_bytes(mem_used)} / {fmt_bytes(mem_total)} ({fmt_pct(pct)})")
            elif mem_used is not None:
                print(f"    HBM Memory Used:    {fmt_bytes(mem_used)}")
            if host_mem is not None:
                print(f"    Host Memory Used:   {fmt_bytes(host_mem)}")
    else:
        print("  No Cloud Monitoring metrics found.")
        print("  (Metrics may take a few minutes to populate after TPU starts.)")
    print()


def display_runtime_metrics(tpu_name, zone):
    """SSH into TPU and display runtime metrics from Prometheus endpoint."""
    print(f"  Runtime Metrics (via SSH → localhost:8431):")
    print(f"  {'-'*76}")

    raw = get_tpu_metrics_via_ssh(tpu_name, zone)
    if not raw or "NO_METRICS" in raw:
        print("    No runtime metrics available (no workload running or endpoint not active).")
        print()
        return

    m = parse_prometheus_metrics(raw)

    # Common TPU runtime metric names
    hbm_usage = aggregate(m.get("tpu_runtime_hbm_memory_usage_bytes"))
    hbm_total = aggregate(m.get("tpu_runtime_hbm_memory_total_bytes"))
    mem_usage = aggregate(m.get("tpu_runtime_memory_usage_bytes"))
    mem_limit = aggregate(m.get("tpu_runtime_memory_limit_bytes"))
    duty = aggregate(m.get("tpu_runtime_duty_cycle_percent"))
    mxu_util = aggregate(m.get("tpu_runtime_mxu_utilization_percent"))
    flops = aggregate(m.get("tpu_runtime_teraflops_per_second"))

    has_any = False
    if hbm_usage is not None:
        has_any = True
        if hbm_total:
            pct = hbm_usage / hbm_total * 100
            print(f"    HBM Memory:         {fmt_bytes(hbm_usage)} / {fmt_bytes(hbm_total)} ({fmt_pct(pct)})")
        else:
            print(f"    HBM Memory Used:    {fmt_bytes(hbm_usage)}")
    if mem_usage is not None:
        has_any = True
        if mem_limit:
            pct = mem_usage / mem_limit * 100
            print(f"    Host Memory:        {fmt_bytes(mem_usage)} / {fmt_bytes(mem_limit)} ({fmt_pct(pct)})")
        else:
            print(f"    Host Memory Used:   {fmt_bytes(mem_usage)}")
    if duty is not None:
        has_any = True
        print(f"    TPU Duty Cycle:     {fmt_pct(duty)}")
    if mxu_util is not None:
        has_any = True
        print(f"    MXU Utilization:    {fmt_pct(mxu_util)}")
    if flops is not None:
        has_any = True
        print(f"    TFLOPS:             {flops:.1f}")

    if not has_any:
        # Dump any metrics that look relevant
        interesting = {k: v for k, v in m.items()
                       if any(kw in k.lower() for kw in ["memory", "duty", "util", "flop", "hbm", "mxu", "tpu"])}
        if interesting:
            print("    Available TPU-related metrics:")
            for k, v in sorted(interesting.items()):
                v_agg = aggregate(v)
                print(f"      {k}: {v_agg}")
        else:
            print("    No recognized TPU metrics found in endpoint output.")
            print(f"    (Got {len(m)} total metrics)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show TPU memory & compute utilization")
    parser.add_argument("--zone", "-z", nargs="*", help="Zone(s) to scan (default: common TPU zones)")
    parser.add_argument("--watch", "-w", type=int, default=0, help="Refresh interval in seconds (0 = once)")
    parser.add_argument("--ssh", action="store_true", help="Also SSH into TPUs for runtime metrics")
    parser.add_argument("--project", "-p", help="GCP project (default: current gcloud config)")
    args = parser.parse_args()

    project = args.project or get_project()
    if not project:
        print("Error: Could not determine GCP project. Set with --project or gcloud config.", file=sys.stderr)
        sys.exit(1)

    zones = args.zone if args.zone else None

    while True:
        token = get_access_token()
        if not token:
            print("Error: Could not get access token. Run 'gcloud auth login'.", file=sys.stderr)
            sys.exit(1)

        tpus = list_tpus(zones)
        if not tpus:
            print(f"\nNo TPU VMs found" + (f" in zone(s): {', '.join(zones)}" if zones else " across common zones."))
            print("Scanned zones:", ", ".join(zones or TPU_ZONES))
            if args.watch:
                time.sleep(args.watch)
                continue
            sys.exit(0)

        display_tpu_info(tpus)
        display_monitoring_metrics(project, token, tpus)

        if args.ssh:
            for tpu in tpus:
                name = tpu.get("name", "").split("/")[-1]
                zone = tpu.get("_zone", "")
                state = tpu.get("state", "")
                if state == "READY":
                    print(f"  --- {name} runtime metrics ---")
                    display_runtime_metrics(name, zone)

        if args.watch:
            print(f"  Refreshing in {args.watch}s... (Ctrl+C to stop)")
            try:
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            # Clear screen for refresh
            print("\033[2J\033[H", end="")
        else:
            break


if __name__ == "__main__":
    main()
