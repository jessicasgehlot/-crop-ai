#!/usr/bin/env python3
"""
monitor.py — Health & Performance Monitor
AI Crop Recommendation System

Usage:
    python monitoring/monitor.py                  # single check
    python monitoring/monitor.py --watch 30       # repeat every 30s
    python monitoring/monitor.py --url http://host:5000
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_URL    = os.environ.get("APP_URL", "http://localhost:5000")
LOG_FILE       = os.environ.get("MONITOR_LOG", "monitoring/monitor.log")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "crop-ai")

# ANSI colors
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ok(msg):    print(f"  {GREEN}[OK]{RESET}   {msg}")
def warn(msg):  print(f"  {YELLOW}[WARN]{RESET} {msg}")
def fail(msg):  print(f"  {RED}[FAIL]{RESET} {msg}")
def info(msg):  print(f"  {CYAN}[INFO]{RESET} {msg}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'-'*50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'-'*50}{RESET}")

# ── 1. API Health Check ───────────────────────────────────────────────────────
def check_api_health(base_url):
    section("API Health Check")
    if not REQUESTS_OK:
        warn("requests not installed — skipping API check")
        return False

    endpoints = [
        ("GET",  "/api/health",        None),
        ("GET",  "/api/clusters",      None),
        ("GET",  "/api/dataset-stats", None),
        ("POST", "/api/recommend",     {
            "temp":25,"humidity":65,"rainfall":100,
            "ph":6.5,"moisture":50,"N":80,"P":40,"K":40
        }),
    ]

    all_ok = True
    for method, path, payload in endpoints:
        url = base_url.rstrip("/") + path
        try:
            start = time.time()
            if method == "POST":
                r = requests.post(url, json=payload, timeout=10)
            else:
                r = requests.get(url, timeout=10)
            elapsed = round((time.time() - start) * 1000, 1)

            if r.status_code == 200:
                ok(f"{method} {path} -> {r.status_code} ({elapsed}ms)")
                if path == "/api/health":
                    d = r.json()
                    info(f"  Model loaded: {d.get('model_loaded')} | "
                         f"Prediction OK: {d.get('prediction_ok')} | "
                         f"Test crop: {d.get('test_crop')}")
            else:
                fail(f"{method} {path} -> {r.status_code}")
                all_ok = False
        except requests.exceptions.ConnectionError:
            fail(f"{method} {path} -> Connection refused (is the app running?)")
            all_ok = False
        except Exception as e:
            fail(f"{method} {path} → {e}")
            all_ok = False
    return all_ok

# ── 2. System Resources ───────────────────────────────────────────────────────
def check_system_resources():
    section("System Resources")
    if not PSUTIL_OK:
        warn("psutil not installed — skipping resource check")
        return

    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # CPU
    if cpu < 70:
        ok(f"CPU usage: {cpu}%")
    elif cpu < 90:
        warn(f"CPU usage: {cpu}% (elevated)")
    else:
        fail(f"CPU usage: {cpu}% (critical)")

    # Memory
    mem_pct = mem.percent
    mem_used = round(mem.used / 1024**3, 2)
    mem_total = round(mem.total / 1024**3, 2)
    if mem_pct < 75:
        ok(f"Memory: {mem_used}GB / {mem_total}GB ({mem_pct}%)")
    elif mem_pct < 90:
        warn(f"Memory: {mem_used}GB / {mem_total}GB ({mem_pct}%)")
    else:
        fail(f"Memory: {mem_used}GB / {mem_total}GB ({mem_pct}%) — critical")

    # Disk
    disk_pct = disk.percent
    disk_free = round(disk.free / 1024**3, 2)
    if disk_pct < 80:
        ok(f"Disk: {disk_pct}% used ({disk_free}GB free)")
    elif disk_pct < 95:
        warn(f"Disk: {disk_pct}% used ({disk_free}GB free)")
    else:
        fail(f"Disk: {disk_pct}% used — almost full!")

    # Network
    net = psutil.net_io_counters()
    info(f"Network — Sent: {round(net.bytes_sent/1024**2,1)}MB | "
         f"Recv: {round(net.bytes_recv/1024**2,1)}MB")

# ── 3. Docker Container Check ─────────────────────────────────────────────────
def check_docker_container(name):
    section(f"Docker Container: {name}")
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format",
             "{{.State.Status}}|{{.State.Health.Status}}|{{.Id}}", name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            warn(f"Container '{name}' not found or Docker not running")
            return

        parts = result.stdout.strip().split("|")
        state  = parts[0] if len(parts) > 0 else "unknown"
        health = parts[1] if len(parts) > 1 else "N/A"
        cid    = parts[2][:12] if len(parts) > 2 else "N/A"

        if state == "running":
            ok(f"Container state: {state} (ID: {cid})")
        else:
            fail(f"Container state: {state}")

        if health in ("healthy", "N/A", ""):
            ok(f"Health status: {health or 'no healthcheck'}")
        else:
            warn(f"Health status: {health}")

        # Stats
        stats = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
             "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}", name],
            capture_output=True, text=True, timeout=10
        )
        if stats.returncode == 0:
            s = stats.stdout.strip().split("|")
            info(f"CPU: {s[0]} | Mem: {s[1]} | Net: {s[2]}")

    except FileNotFoundError:
        warn("Docker not installed or not in PATH")
    except Exception as e:
        warn(f"Docker check error: {e}")

# ── 4. File & Model Integrity ─────────────────────────────────────────────────
def check_files():
    section("File & Model Integrity")
    required = [
        "app.py", "requirements.txt", "Dockerfile",
        "data/crop_data.csv", "data/kmeans_model.pkl",
        "ml/model.py", "ml/dataset.py",
    ]
    for f in required:
        if os.path.exists(f):
            size = round(os.path.getsize(f) / 1024, 1)
            ok(f"{f} ({size} KB)")
        else:
            fail(f"{f} — MISSING")

# ── 5. Application Logs ───────────────────────────────────────────────────────
def check_logs(log_path="monitoring/monitor.log"):
    section("Recent Application Logs")
    if not os.path.exists(log_path):
        info(f"No log file at {log_path}")
        return
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        last = lines[-20:] if len(lines) > 20 else lines
        for line in last:
            line = line.rstrip()
            if "ERROR" in line or "FAIL" in line:
                fail(line)
            elif "WARN" in line:
                warn(line)
            else:
                info(line)
    except Exception as e:
        warn(f"Could not read log: {e}")

# ── 6. Write Summary to Log ───────────────────────────────────────────────────
def write_log(summary: dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"timestamp": ts(), **summary}) + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_checks(base_url):
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}  AgriAI Monitor — {ts()}{RESET}")
    print(f"{BOLD}  Platform: {platform.system()} {platform.release()}{RESET}")
    print(f"{BOLD}{'='*50}{RESET}")

    api_ok = check_api_health(base_url)
    check_system_resources()
    check_docker_container(CONTAINER_NAME)
    check_files()
    check_logs()

    section("Summary")
    if api_ok:
        ok("All API endpoints healthy")
    else:
        fail("One or more API endpoints failed")

    write_log({"api_ok": api_ok, "url": base_url})
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriAI Monitor")
    parser.add_argument("--url",   default=DEFAULT_URL, help="Base URL of the app")
    parser.add_argument("--watch", type=int, default=0,
                        help="Repeat interval in seconds (0 = run once)")
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Watching every {args.watch}s — Ctrl+C to stop")
        while True:
            run_checks(args.url)
            time.sleep(args.watch)
    else:
        run_checks(args.url)
