#!/usr/bin/env python3
"""Show memory and compute utilization for all local TPU chips.

Run this directly on the TPU VM. Queries the libtpu Prometheus metrics
endpoint at localhost:8431, plus host memory from /proc/meminfo.

Usage:
    python scripts/tpu_utilization.py              # one-shot
    python scripts/tpu_utilization.py --watch 5    # refresh every 5s
    python scripts/tpu_utilization.py --raw        # dump all TPU-related metrics
"""

import argparse
import os
import re
import sys
import time
import urllib.request
from datetime import datetime


METRICS_URL = "http://localhost:8431/metrics"


def fetch_metrics():
    """Fetch Prometheus metrics from the local libtpu endpoint."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        return None


def parse_prometheus(raw):
    """Parse Prometheus text format into {metric_name: [(labels_dict, value), ...]}."""
    metrics = {}
    if not raw:
        return metrics
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse: metric_name{label="val",...} value  OR  metric_name value
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?(.*?)\}?\s+([\d.eE+\-]+|NaN|Inf|\+Inf|-Inf)$', line)
        if not m:
            # Try without labels
            m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.eE+\-]+|NaN|Inf|\+Inf|-Inf)$', line)
            if m:
                name, val_str = m.group(1), m.group(2)
                labels = {}
            else:
                continue
        else:
            name, labels_str, val_str = m.group(1), m.group(2), m.group(3)
            labels = {}
            if labels_str:
                for pair in re.findall(r'(\w+)="([^"]*)"', labels_str):
                    labels[pair[0]] = pair[1]

        try:
            val = float(val_str)
        except ValueError:
            continue

        metrics.setdefault(name, []).append((labels, val))
    return metrics


def get_values(metrics, name):
    """Get all values for a metric name, returning list of (labels, value)."""
    return metrics.get(name, [])


def avg_values(metrics, name):
    """Get mean of all values for a metric."""
    vals = get_values(metrics, name)
    if not vals:
        return None
    return sum(v for _, v in vals) / len(vals)


def sum_values(metrics, name):
    """Get sum of all values for a metric."""
    vals = get_values(metrics, name)
    if not vals:
        return None
    return sum(v for _, v in vals)


def per_chip_values(metrics, name):
    """Get per-chip values, keyed by chip_id or index."""
    vals = get_values(metrics, name)
    result = {}
    for i, (labels, val) in enumerate(vals):
        chip_id = labels.get("chip_id", labels.get("core", labels.get("device", str(i))))
        result[chip_id] = val
    return result


def fmt_bytes(b):
    if b is None:
        return "N/A"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PiB"


def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def bar(pct, width=30):
    """Render a simple progress bar."""
    if pct is None:
        return "[" + "?" * width + "]"
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def get_host_memory():
    """Read host memory from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024  # kB -> bytes
            total = info.get("MemTotal")
            available = info.get("MemAvailable")
            if total and available:
                used = total - available
                return used, total
    except (FileNotFoundError, ValueError):
        pass
    return None, None


def get_num_tpu_chips():
    """Detect number of TPU chips via /dev/accel* or /dev/vfio/*."""
    count = 0
    for path in ["/dev/accel0", "/dev/accel1", "/dev/accel2", "/dev/accel3",
                 "/dev/accel4", "/dev/accel5", "/dev/accel6", "/dev/accel7"]:
        if os.path.exists(path):
            count += 1
    if count == 0:
        # Try vfio for TPU v4+
        vfio = "/dev/vfio"
        if os.path.isdir(vfio):
            count = len([f for f in os.listdir(vfio) if f.isdigit()])
    return count


def display(metrics, show_per_chip=True):
    """Display formatted TPU utilization."""
    num_chips = get_num_tpu_chips()
    host_mem_used, host_mem_total = get_host_memory()

    print(f"\n{'='*70}")
    print(f"  TPU Utilization — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if num_chips:
        print(f"  TPU chips detected: {num_chips}")
    print()

    # --- Host Memory ---
    print("  Host Memory")
    print(f"  {'-'*66}")
    if host_mem_total:
        pct = host_mem_used / host_mem_total * 100
        print(f"    RAM:  {fmt_bytes(host_mem_used)} / {fmt_bytes(host_mem_total)}  {bar(pct)} {fmt_pct(pct)}")
    else:
        print("    RAM:  N/A (not on Linux or /proc/meminfo unavailable)")
    print()

    if not metrics:
        print("  TPU Metrics: No data from localhost:8431")
        print("  (Is a workload running? libtpu exposes metrics only during execution.)")
        print()
        return

    # --- HBM Memory ---
    # Try various known metric names
    hbm_usage_names = [
        "tpu_runtime_hbm_memory_usage_bytes",
        "jax_hbm_memory_usage_bytes",
        "memory_usage_bytes",
    ]
    hbm_total_names = [
        "tpu_runtime_hbm_memory_total_bytes",
        "jax_hbm_memory_total_bytes",
        "memory_total_bytes",
    ]

    hbm_used_per_chip = {}
    hbm_total_per_chip = {}
    for name in hbm_usage_names:
        hbm_used_per_chip = per_chip_values(metrics, name)
        if hbm_used_per_chip:
            break
    for name in hbm_total_names:
        hbm_total_per_chip = per_chip_values(metrics, name)
        if hbm_total_per_chip:
            break

    print("  TPU HBM Memory")
    print(f"  {'-'*66}")
    if hbm_used_per_chip:
        for chip_id in sorted(hbm_used_per_chip.keys()):
            used = hbm_used_per_chip[chip_id]
            total = hbm_total_per_chip.get(chip_id)
            if total and total > 0:
                pct = used / total * 100
                print(f"    Chip {chip_id}:  {fmt_bytes(used)} / {fmt_bytes(total)}  {bar(pct)} {fmt_pct(pct)}")
            else:
                print(f"    Chip {chip_id}:  {fmt_bytes(used)}")
    else:
        print("    No HBM memory metrics found.")
    print()

    # --- Compute Utilization ---
    duty_names = [
        "tpu_runtime_duty_cycle_percent",
        "jax_duty_cycle_percent",
        "duty_cycle",
    ]
    mxu_names = [
        "tpu_runtime_mxu_utilization_percent",
        "jax_mxu_utilization_percent",
    ]
    flops_names = [
        "tpu_runtime_teraflops_per_second",
        "jax_teraflops",
    ]

    duty_per_chip = {}
    mxu_per_chip = {}
    flops_per_chip = {}
    for name in duty_names:
        duty_per_chip = per_chip_values(metrics, name)
        if duty_per_chip:
            break
    for name in mxu_names:
        mxu_per_chip = per_chip_values(metrics, name)
        if mxu_per_chip:
            break
    for name in flops_names:
        flops_per_chip = per_chip_values(metrics, name)
        if flops_per_chip:
            break

    print("  TPU Compute")
    print(f"  {'-'*66}")
    has_compute = False
    if duty_per_chip:
        has_compute = True
        for chip_id in sorted(duty_per_chip.keys()):
            val = duty_per_chip[chip_id]
            print(f"    Chip {chip_id} Duty Cycle:     {bar(val)} {fmt_pct(val)}")
    if mxu_per_chip:
        has_compute = True
        for chip_id in sorted(mxu_per_chip.keys()):
            val = mxu_per_chip[chip_id]
            print(f"    Chip {chip_id} MXU Util:        {bar(val)} {fmt_pct(val)}")
    if flops_per_chip:
        has_compute = True
        for chip_id in sorted(flops_per_chip.keys()):
            val = flops_per_chip[chip_id]
            print(f"    Chip {chip_id} TFLOPS:          {val:.1f}")
    if not has_compute:
        print("    No compute utilization metrics found.")
    print()


def display_raw(metrics):
    """Dump all TPU/memory/compute related metrics."""
    keywords = ["memory", "hbm", "duty", "util", "flop", "mxu", "tpu", "jax",
                "xla", "megacore", "infeed", "outfeed"]
    print(f"\n{'='*70}")
    print(f"  Raw TPU-related metrics")
    print(f"{'='*70}")
    found = False
    for name in sorted(metrics.keys()):
        if any(kw in name.lower() for kw in keywords):
            found = True
            entries = metrics[name]
            if len(entries) == 1 and not entries[0][0]:
                print(f"  {name}: {entries[0][1]}")
            else:
                print(f"  {name}:")
                for labels, val in entries:
                    label_str = ", ".join(f'{k}="{v}"' for k, v in labels.items()) if labels else ""
                    print(f"    {{{label_str}}} = {val}")
    if not found:
        print("  No TPU-related metrics found.")
        print(f"  Total metrics available: {len(metrics)}")
        if metrics:
            print("  All metric names:")
            for name in sorted(metrics.keys()):
                print(f"    {name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show local TPU memory & compute utilization")
    parser.add_argument("--watch", "-w", type=int, default=0,
                        help="Refresh interval in seconds (0 = one-shot)")
    parser.add_argument("--raw", "-r", action="store_true",
                        help="Dump all TPU-related raw metrics")
    parser.add_argument("--url", default=METRICS_URL,
                        help=f"Metrics endpoint URL (default: {METRICS_URL})")
    args = parser.parse_args()

    global METRICS_URL
    METRICS_URL = args.url

    while True:
        raw = fetch_metrics()
        metrics = parse_prometheus(raw) if raw else {}

        if args.raw:
            display_raw(metrics)
        else:
            display(metrics)

        if args.watch:
            print(f"  Refreshing in {args.watch}s... (Ctrl+C to stop)")
            try:
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            print("\033[2J\033[H", end="", flush=True)
        else:
            break


if __name__ == "__main__":
    main()
