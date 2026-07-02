#!/usr/bin/env python3
"""
run_stress_test.py — thin wrapper for the STRESS TEST.

Runs the split-screen clip (6 tiles = 6 clips at once), forcing all models to
detect in parallel, and records the performance metrics under load (throughput,
drop rate, per-detector latency, GPU/CPU/RAM resources).

Usage:
    venv/bin/python testing/run_stress_test.py /path/clip_stress.mp4

Any extra argument (--host, --duration, --loop, ...) is forwarded to
test_harness.py.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Utilizare: run_stress_test.py <clip.mp4> [extra args]")
        return 2
    video = sys.argv[1]
    extra = sys.argv[2:]
    cmd = [sys.executable, str(HERE / "test_harness.py"),
           "--mode", "stress", "--video", video] + extra
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
