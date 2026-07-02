#!/usr/bin/env python3
"""
run_sequential_test.py — thin wrapper for the SEQUENTIAL test.

Runs the clip where the objects of interest appear one at a time (known person,
unknowns, plates, vandalism, fights, fire, weapon) and automatically computes
accuracy based on the annotation file.

Usage:
    venv/bin/python testing/run_sequential_test.py /path/clip_sequential.mp4
    venv/bin/python testing/run_sequential_test.py /path/clip.mp4 --annotations testing/annotations.json

Any extra argument (--host, --duration, --no-ws, ...) is forwarded to
test_harness.py.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Utilizare: run_sequential_test.py <clip.mp4> [--annotations FILE] [extra args]")
        return 2
    video = sys.argv[1]
    extra = sys.argv[2:]
    # default annotations, if not given explicitly
    if "--annotations" not in extra:
        default_ann = HERE / "annotations.json"
        if not default_ann.is_file():
            print(f"⚠  Nu există {default_ann}. Copiază annotations_template.json în "
                  f"annotations.json și completează-l, sau dă --annotations explicit.")
        extra += ["--annotations", str(default_ann)]
    cmd = [sys.executable, str(HERE / "test_harness.py"),
           "--mode", "sequential", "--video", video] + extra
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
