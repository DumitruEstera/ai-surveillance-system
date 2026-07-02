#!/usr/bin/env python3
"""
test_harness.py — runs a video clip through the whole application pipeline, as if
it came from a surveillance camera, and automatically records the metrics needed
for the testing chapter of the thesis.

How it works (uses only the backend's public API — imports nothing from app.py):
  1. /api/login            -> obtain a JWT (admin)
  2. /api/*/toggle         -> enable all detectors
  3. /api/streams (file)   -> start playing the clip as source "CAM-TEST",
                             paced at the native FPS (see app.py, "file" source)
  4. collectors in parallel:
       - StatusPoller   (1 Hz) -> status_timeseries.csv  (throughput, FPS, drop,
                                  per-detector latency under load, queue depth)
       - ResourceSampler(1 Hz) -> resources_timeseries.csv (CPU, RAM, GPU, VRAM)
       - WSCollector    (/ws)  -> detections.jsonl (each detection: class,
                                  confidence, bbox, time)
  5. wait for EOF (clip finished) or --duration seconds
  6. /api/logs + /api/alarms -> db_detection_logs.csv, db_alarms.csv (the truth
                                persisted in PostgreSQL)
  7. aggregate everything -> summary.json + summary.tex (LaTeX tables) + run_meta.json
     (+ scoring.json/csv in "sequential" mode, if annotations exist)

RUN (from this directory, using the application's virtual environment):
    python testing/test_harness.py --mode stress \
        --video /path/clip_stress.mp4

    python testing/test_harness.py --mode sequential \
        --video /path/clip_sequential.mp4 \
        --annotations testing/annotations.json

The backend must already be running (python app.py, on :8000) with PostgreSQL up.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    import psutil
except Exception:
    psutil = None

# WS is optional: if missing, we only drop detections.jsonl and rely on
# db_detection_logs.csv (leaner, but enough for the functional check).
try:
    import asyncio
    import websockets
    _HAS_WS = True
except Exception:
    _HAS_WS = False


# --------------------------------------------------------------------------- #
#                            Minimal REST client                              #
# --------------------------------------------------------------------------- #
class Backend:
    """Thin wrapper over the backend's REST API."""

    DETECTORS = ("face", "plate", "demographics", "fire", "har", "weapon")

    def __init__(self, host, user, password):
        self.host = host.rstrip("/")
        self.session = requests.Session()
        self.token = None
        self._login(user, password)

    def _login(self, user, password):
        r = self.session.post(f"{self.host}/api/login",
                              json={"username": user, "password": password},
                              timeout=15)
        r.raise_for_status()
        self.token = r.json()["token"]
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def enable_all_detectors(self):
        for d in self.DETECTORS:
            try:
                self.session.post(f"{self.host}/api/{d}/toggle",
                                  json={"enabled": True}, timeout=10)
            except Exception as e:
                print(f"  ⚠  nu am putut activa {d}: {e}")

    def start_file_stream(self, path, camera_id, loop):
        r = self.session.post(f"{self.host}/api/streams",
                              json={"source": "file", "path": path,
                                    "camera_id": camera_id, "loop": loop},
                              timeout=60)
        r.raise_for_status()
        return r.json()

    def stop_stream(self, camera_id):
        try:
            self.session.delete(f"{self.host}/api/streams/{camera_id}", timeout=15)
        except Exception as e:
            print(f"  ⚠  nu am putut opri stream-ul: {e}")

    def get_status(self):
        r = self.session.get(f"{self.host}/api/status", timeout=10)
        r.raise_for_status()
        return r.json()

    def stream_finished(self, camera_id):
        """True if CAM-TEST reached EOF (or no longer exists)."""
        try:
            r = self.session.get(f"{self.host}/api/streams", timeout=10)
            r.raise_for_status()
            for cam in r.json().get("cameras", []):
                if cam.get("camera_id") == camera_id:
                    return cam.get("finished", False) or not cam.get("active", False)
            return True  # no longer in the list -> considered finished
        except Exception:
            return False

    def get_logs(self, camera_id, date_from):
        r = self.session.get(f"{self.host}/api/logs",
                             params={"camera_id": camera_id, "date_from": date_from,
                                     "limit": 5000},
                             timeout=30)
        r.raise_for_status()
        return r.json().get("logs", [])

    def get_alarms(self, camera_id):
        r = self.session.get(f"{self.host}/api/alarms",
                             params={"camera_id": camera_id, "limit": 1000},
                             timeout=30)
        r.raise_for_status()
        return r.json().get("alarms", [])


# --------------------------------------------------------------------------- #
#                        GPU sampling via nvidia-smi                          #
# --------------------------------------------------------------------------- #
def sample_gpu():
    """Return (util%, vram_used_MB, vram_total_MB, temp_C) or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        util, mem_used, mem_total, temp = [x.strip() for x in line.split(",")]
        return (float(util), float(mem_used), float(mem_total), float(temp))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#                               Collectors                                     #
# --------------------------------------------------------------------------- #
DETS = ("face", "plate", "fire", "har", "weapon")


class StatusPoller(threading.Thread):
    """Poll /api/status at 1 Hz and write a time series."""

    def __init__(self, backend, out_csv, stop_event, t0):
        super().__init__(daemon=True)
        self.backend, self.out_csv, self.stop = backend, out_csv, stop_event
        self.t0 = t0
        self.rows = []
        self._prev = None  # (t, frames_captured, frames_processed)

    def run(self):
        fields = (["t_rel", "frames_captured", "frames_processed", "frames_dropped",
                   "frames_skipped", "queue_skips", "face_detections", "plate_detections",
                   "fire_detections", "har_detections", "weapon_detections",
                   "capture_fps", "process_fps", "realtime_skip_pct", "queue_drop_pct"]
                  + [f"{d}_avg_ms" for d in DETS]
                  + [f"{d}_qdepth" for d in DETS])
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            while not self.stop.is_set():
                try:
                    st = self.backend.get_status()
                except Exception:
                    time.sleep(1.0)
                    continue
                t = time.monotonic() - self.t0
                perf = st.get("performance", {})
                lat = st.get("detector_latency_ms", {}) or {}
                qd = st.get("queue_depth", {}) or {}
                fc = perf.get("frames_captured", 0)
                fp = perf.get("frames_processed", 0)
                fd = perf.get("frames_dropped", 0)
                fsk = perf.get("frames_skipped", 0)
                cap_fps = proc_fps = 0.0
                if self._prev:
                    dt = t - self._prev[0]
                    if dt > 0:
                        cap_fps = (fc - self._prev[1]) / dt
                        proc_fps = (fp - self._prev[2]) / dt
                self._prev = (t, fc, fp)
                # frames ARRIVING at 30 fps = read + skipped in real time
                arrived = (fc + fsk) or 1
                row = {
                    "t_rel": round(t, 2),
                    "frames_captured": fc, "frames_processed": fp,
                    "frames_dropped": fd, "frames_skipped": fsk,
                    "queue_skips": perf.get("queue_skips", 0),
                    "face_detections": perf.get("face_detections", 0),
                    "plate_detections": perf.get("plate_detections", 0),
                    "fire_detections": perf.get("fire_detections", 0),
                    "har_detections": perf.get("har_detections", 0),
                    "weapon_detections": perf.get("weapon_detections", 0),
                    "capture_fps": round(cap_fps, 2),
                    "process_fps": round(proc_fps, 2),
                    # % of the 30 fps stream skipped to stay real-time
                    "realtime_skip_pct": round(100.0 * fsk / arrived, 2),
                    # % of read frames lost to full queues
                    "queue_drop_pct": round(100.0 * fd / (fc or 1), 2),
                }
                for d in DETS:
                    v = lat.get(d)
                    row[f"{d}_avg_ms"] = round(v, 2) if v is not None else ""
                    row[f"{d}_qdepth"] = qd.get(d, "")
                w.writerow(row)
                f.flush()
                self.rows.append(row)
                self.stop.wait(1.0)


class ResourceSampler(threading.Thread):
    """Sample CPU/RAM (psutil) + GPU (nvidia-smi) at 1 Hz."""

    def __init__(self, out_csv, stop_event, t0):
        super().__init__(daemon=True)
        self.out_csv, self.stop, self.t0 = out_csv, stop_event, t0
        self.rows = []

    def run(self):
        fields = ["t_rel", "cpu_pct", "ram_used_mb", "ram_pct",
                  "gpu_util_pct", "vram_used_mb", "vram_total_mb", "gpu_temp_c"]
        if psutil:
            psutil.cpu_percent(None)  # first call calibrates
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            while not self.stop.is_set():
                t = round(time.monotonic() - self.t0, 2)
                row = {k: "" for k in fields}
                row["t_rel"] = t
                if psutil:
                    row["cpu_pct"] = psutil.cpu_percent(None)
                    vm = psutil.virtual_memory()
                    row["ram_used_mb"] = round(vm.used / 1e6, 1)
                    row["ram_pct"] = vm.percent
                g = sample_gpu()
                if g:
                    row["gpu_util_pct"], row["vram_used_mb"], row["vram_total_mb"], row["gpu_temp_c"] = g
                w.writerow(row)
                f.flush()
                self.rows.append(row)
                self.stop.wait(1.0)


def _extract_detections(msg, t_rel):
    """Flatten a WS 'video_frame' message into a list of events."""
    out = []
    cam = msg.get("camera_id", "")
    fid = msg.get("frame_id")

    def emit(detector, label, conf, item):
        out.append({
            "t_rel": round(t_rel, 3), "camera_id": cam, "frame_id": fid,
            "detector": detector, "label": label,
            "confidence": round(float(conf), 4) if conf is not None else None,
            "bbox": item.get("bbox"),
            "extra": {k: item.get(k) for k in
                      ("age", "gender", "emotion", "severity", "is_authorized",
                       "owner", "confirmed", "action_label") if k in item},
        })

    for it in msg.get("face_results", []) or []:
        emit("face", it.get("name", "Unknown"), it.get("confidence"), it)
    for it in msg.get("plate_results", []) or []:
        emit("plate", it.get("plate_number") or it.get("plate", ""),
             it.get("confidence"), it)
    for it in msg.get("fire_results", []) or []:
        emit("fire", it.get("class", "fire"), it.get("confidence"), it)
    for it in msg.get("har_results", []) or []:
        emit("har", it.get("class", "normal"), it.get("confidence"), it)
    for it in msg.get("weapon_results", []) or []:
        emit("weapon", it.get("class", "weapon"), it.get("confidence"), it)
    return out


class WSCollector(threading.Thread):
    """Listen on /ws and write each detection to a JSONL."""

    def __init__(self, host, token, camera_id, out_jsonl, stop_event, t0):
        super().__init__(daemon=True)
        self.host = host.replace("http://", "ws://").replace("https://", "wss://")
        self.token, self.camera_id = token, camera_id
        self.out_jsonl, self.stop, self.t0 = out_jsonl, stop_event, t0
        self.events = []
        # t_rel of the FIRST frame actually broadcast for this camera. Used as the
        # time origin in scoring: t0 is fixed while the stream is not yet flowing
        # (stream start, WS connect, model warmup), so measuring latency straight
        # from t0 would add a fixed offset (~0.8 s) to every segment. Aligning to the
        # first frame makes latency reflect the real delay after the object appears.
        self.first_frame_t = None

    def run(self):
        if not _HAS_WS:
            print("  ⚠  websockets indisponibil — sar peste detections.jsonl")
            return
        asyncio.run(self._loop())

    async def _loop(self):
        url = f"{self.host}/ws?token={self.token}"
        f = open(self.out_jsonl, "w")
        try:
            async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
                while not self.stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("type") != "video_frame":
                        continue
                    if self.camera_id and msg.get("camera_id") != self.camera_id:
                        continue
                    t_rel = time.monotonic() - self.t0
                    if self.first_frame_t is None:
                        self.first_frame_t = t_rel
                    for ev in _extract_detections(msg, t_rel):
                        f.write(json.dumps(ev) + "\n")
                        self.events.append(ev)
                    f.flush()
        except Exception as e:
            print(f"  ⚠  WS închis: {e}")
        finally:
            f.close()


# --------------------------------------------------------------------------- #
#                    Functional scoring (sequential mode)                     #
# --------------------------------------------------------------------------- #
def _norm_plate(s):
    return "".join(c for c in (s or "").upper() if c.isalnum())


def _name_match(label, want):
    """Name match insensitive to order and case
    ('Estera Dumitru' == 'Dumitru Estera')."""
    return set((label or "").lower().split()) == set((want or "").lower().split())


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_sequential(annotations, events, har_tolerance_sec=5.0, frame_origin=0.0):
    """Compare WS detections against the expected timeline. events: list from WSCollector.

    HAR is a SCENE-level classifier: it runs inference about once every ~3s over a
    2s clip, so it localizes coarsely (~5s). That is why HAR windows are compared
    with a `har_tolerance_sec` tolerance, while per-object detectors
    (face/plate/fire/weapon) are compared strictly on the annotated window.

    `frame_origin` is the t_rel of the first frame actually broadcast (see
    WSCollector.first_frame_t). Annotations are in clip time (start_sec=0 = first
    frame), so we subtract this origin from t_rel so latency does not include the
    startup offset between t0 and when the stream reaches the pipeline."""
    per_segment = []
    matched_event_ids = set()

    for seg in annotations:
        det = seg["detector"]
        s, e = seg["start_sec"], seg["end_sec"]
        exp = seg.get("expected", {})
        # HAR: widen the window by the tolerance (coarse, scene-level localization)
        tol = har_tolerance_sec if det == "har" else 0.0
        ms, me = s - tol, e + tol
        in_seg = [(i, ev) for i, ev in enumerate(events)
                  if ev["detector"] == det and ms <= ev["t_rel"] - frame_origin <= me]
        # HAR: ignore the "normal" class when matching
        if det == "har":
            in_seg = [(i, ev) for i, ev in in_seg if ev["label"] != "normal"]

        detected = len(in_seg) > 0
        first_t = min((ev["t_rel"] - frame_origin for _, ev in in_seg), default=None)
        confs = [ev["confidence"] for _, ev in in_seg if ev["confidence"] is not None]
        for i, _ in in_seg:
            matched_event_ids.add(i)

        result = {
            "detector": det, "start_sec": s, "end_sec": e, "expected": exp,
            "detected": detected,
            # latency = how long after the segment start the first detection appears;
            # 0 if already present (relevant for HAR with tolerance).
            "detection_latency_sec": round(max(0.0, first_t - s), 2) if first_t is not None else None,
            "n_detections": len(in_seg),
            "conf_mean": round(statistics.fmean(confs), 4) if confs else None,
            "conf_min": round(min(confs), 4) if confs else None,
            "conf_max": round(max(confs), 4) if confs else None,
        }

        # detector-specific correctness
        labels = [ev["label"] for _, ev in in_seg]
        if det == "har" and "class" in exp:
            result["classification_correct"] = exp["class"] in labels
        elif det == "fire" and "class" in exp:
            result["classification_correct"] = exp["class"] in labels
        elif det == "face":
            n_unknown = sum(1 for lbl in labels if lbl == "Unknown")
            if exp.get("identity") == "known":
                want = exp.get("name", "")
                result["identity_correct"] = any(
                    lbl != "Unknown" and (not want or _name_match(lbl, want))
                    for lbl in labels)
            elif exp.get("identity") == "unknown":
                # correct = the person is detected and NOT falsely recognized as an
                # enrolled employee (most frames -> "Unknown").
                result["identity_correct"] = (
                    detected and labels and n_unknown >= len(labels) / 2)
            result["frac_unknown"] = round(n_unknown / len(labels), 3) if labels else None
        elif det == "plate" and "plate" in exp:
            want = _norm_plate(exp["plate"])
            reads = [_norm_plate(l) for l in labels if l]
            best_cer, exact = None, False
            for rd in reads:
                cer = _levenshtein(rd, want) / max(len(want), 1)
                best_cer = cer if best_cer is None else min(best_cer, cer)
                exact = exact or (rd == want)
            result["ocr_exact_match"] = exact
            result["ocr_cer"] = round(best_cer, 4) if best_cer is not None else None
        elif det == "weapon":
            result["classification_correct"] = detected  # presence

        per_segment.append(result)

    # false positives = detections outside their detector's segments
    # (with the same tolerance as for HAR matching).
    fp_by_det = {d: 0 for d in DETS}
    seg_by_det = {}
    for seg in annotations:
        tol = har_tolerance_sec if seg["detector"] == "har" else 0.0
        seg_by_det.setdefault(seg["detector"], []).append(
            (seg["start_sec"] - tol, seg["end_sec"] + tol))
    for i, ev in enumerate(events):
        if ev["detector"] == "har" and ev["label"] == "normal":
            continue
        det = ev["detector"]
        ranges = seg_by_det.get(det, [])
        inside = any(s <= ev["t_rel"] - frame_origin <= e for s, e in ranges)
        if not inside:
            fp_by_det[det] = fp_by_det.get(det, 0) + 1

    # aggregate per detector
    recall = {}
    for d in DETS:
        segs = [r for r in per_segment if r["detector"] == d]
        if segs:
            recall[d] = round(sum(1 for r in segs if r["detected"]) / len(segs), 3)
    return {"per_segment": per_segment, "recall_per_detector": recall,
            "false_positives_per_detector": fp_by_det,
            "har_tolerance_sec": har_tolerance_sec}


# --------------------------------------------------------------------------- #
#                        Summary + LaTeX tables                               #
# --------------------------------------------------------------------------- #
def write_csv(path, rows, fieldnames=None):
    if not rows:
        Path(path).write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_summary(mode, status_rows, resource_rows, logs, alarms, events, scoring, meta):
    summary = {"mode": mode, "meta": meta}

    # global throughput (from the last cumulative row)
    if status_rows:
        last = status_rows[-1]
        proc_fps = [r["process_fps"] for r in status_rows if r["process_fps"]]
        native_fps = (meta or {}).get("fps_native") or 30.0
        summary["throughput"] = {
            "frames_arrived": last["frames_captured"] + last.get("frames_skipped", 0),
            "frames_captured": last["frames_captured"],
            "frames_processed": last["frames_processed"],
            "frames_dropped_queue": last["frames_dropped"],
            "frames_skipped_realtime": last.get("frames_skipped", 0),
            "queue_skips": last["queue_skips"],
            # % of the 30 fps stream skipped to stay real-time
            "realtime_skip_pct": last.get("realtime_skip_pct"),
            # % of read frames lost to full queues
            "queue_drop_pct": last.get("queue_drop_pct"),
            "process_fps_mean": round(statistics.fmean(proc_fps), 2) if proc_fps else None,
            "native_fps": round(native_fps, 1),
        }
        # mean per-detector latency under load (from the last available samples)
        lat = {}
        for d in DETS:
            vals = [r[f"{d}_avg_ms"] for r in status_rows if r.get(f"{d}_avg_ms") not in ("", None)]
            if vals:
                lat[d] = round(vals[-1], 2)  # the mean is already cumulative in the backend
        summary["detector_latency_ms_under_load"] = lat

    # resources
    if resource_rows:
        def col(name):
            return [float(r[name]) for r in resource_rows if r.get(name) not in ("", None)]
        res = {}
        for name in ("cpu_pct", "ram_used_mb", "gpu_util_pct", "vram_used_mb", "gpu_temp_c"):
            vals = col(name)
            if vals:
                res[name] = {"mean": round(statistics.fmean(vals), 1),
                             "max": round(max(vals), 1)}
        summary["resources"] = res

    # detections
    summary["detection_counts_db"] = {}
    for lg in logs:
        t = lg.get("type", "?")
        summary["detection_counts_db"][t] = summary["detection_counts_db"].get(t, 0) + 1
    summary["alarms_total"] = len(alarms)
    summary["alarms_by_type"] = {}
    for al in alarms:
        t = al.get("type", "?")
        summary["alarms_by_type"][t] = summary["alarms_by_type"].get(t, 0) + 1

    # detection co-occurrence (how many detector types fire in 1s windows)
    if events:
        windows = {}
        for ev in events:
            if ev["detector"] == "har" and ev["label"] == "normal":
                continue
            w = int(ev["t_rel"])
            windows.setdefault(w, set()).add(ev["detector"])
        counts = [len(s) for s in windows.values()]
        if counts:
            summary["concurrency"] = {
                "max_detector_types_simultaneous": max(counts),
                "mean_detector_types_per_active_sec": round(statistics.fmean(counts), 2),
                "active_seconds": len(counts),
            }

    if scoring:
        summary["functional_scoring"] = scoring
    return summary


def write_latex(path, mode, summary):
    lines = []
    L = {"face": "Recunoaștere facială", "plate": "Plăcuțe înmatriculare",
         "fire": "Detecție foc/fum", "har": "Recunoaștere acțiuni (SlowFast)",
         "weapon": "Detecție armă"}

    if mode == "stress":
        tp = summary.get("throughput", {})
        lat = summary.get("detector_latency_ms_under_load", {})
        lines += [r"% Tabel throughput + latență sub încărcare (stress test)",
                  r"\begin{tabular}{lrr}", r"\toprule",
                  r"Detector & Latență medie sub load (ms) & Detecții \\",
                  r"\midrule"]
        counts = summary.get("detection_counts_db", {})
        for d in DETS:
            if d in lat:
                lines.append(f"{L[d]} & {lat[d]:.1f} & {counts.get(d, 0)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", "",
                  r"% Throughput global",
                  r"\begin{tabular}{lr}", r"\toprule", r"Metrică & Valoare \\",
                  r"\midrule",
                  f"FPS sursă (nativ) & {tp.get('native_fps', '-')} \\\\",
                  f"FPS procesare (medie) & {tp.get('process_fps_mean', '-')} \\\\",
                  f"Cadre sosite (30 fps) & {tp.get('frames_arrived', '-')} \\\\",
                  f"Cadre procesate & {tp.get('frames_processed', '-')} \\\\",
                  f"Sărite în timp real (\\%) & {tp.get('realtime_skip_pct', '-')} \\\\",
                  f"Pierdute la cozi (\\%) & {tp.get('queue_drop_pct', '-')} \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        res = summary.get("resources", {})
        if res:
            lines += ["", r"% Resurse", r"\begin{tabular}{lrr}", r"\toprule",
                      r"Resursă & Medie & Vârf \\", r"\midrule"]
            label = {"cpu_pct": "CPU (\\%)", "ram_used_mb": "RAM (MB)",
                     "gpu_util_pct": "GPU (\\%)", "vram_used_mb": "VRAM (MB)",
                     "gpu_temp_c": "Temp. GPU (°C)"}
            for k, lab in label.items():
                if k in res:
                    lines.append(f"{lab} & {res[k]['mean']} & {res[k]['max']} \\\\")
            lines += [r"\bottomrule", r"\end{tabular}"]

    else:  # sequential
        sc = summary.get("functional_scoring")
        if sc:
            lines += [r"% Rezultate funcționale per segment (test secvențial)",
                      r"\begin{tabular}{llcrr}", r"\toprule",
                      r"Detector & Așteptat & Detectat & Latență (s) & Încredere medie \\",
                      r"\midrule"]
            for r_ in sc["per_segment"]:
                exp = r_["expected"]
                exp_str = exp.get("name") or exp.get("plate") or exp.get("class") \
                    or exp.get("identity") or ("prezent" if exp.get("present") else "-")
                det = "da" if r_["detected"] else "nu"
                latv = r_["detection_latency_sec"]
                conf = r_["conf_mean"]
                lines.append(f"{L.get(r_['detector'], r_['detector'])} & {exp_str} & {det} & "
                             f"{latv if latv is not None else '-'} & "
                             f"{conf if conf is not None else '-'} \\\\")
            lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
#                                   main                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Harness de testare end-to-end pentru pipeline")
    ap.add_argument("--mode", choices=["sequential", "stress"], required=True)
    ap.add_argument("--video", required=True, help="calea clipului de test")
    ap.add_argument("--annotations", default=None, help="JSON cu timeline-ul așteptat (sequential)")
    ap.add_argument("--loop", action="store_true", help="reia clipul la EOF")
    ap.add_argument("--duration", type=float, default=None,
                    help="oprește forțat după N secunde (altfel: până la EOF)")
    ap.add_argument("--out", default=None, help="director de ieșire (implicit: testing/results/<mode>_<ts>)")
    ap.add_argument("--host", default=os.environ.get("HARNESS_HOST", "http://localhost:8000"))
    ap.add_argument("--user", default=os.environ.get("HARNESS_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("HARNESS_PASSWORD", "admin"))
    ap.add_argument("--camera-id", default="CAM-TEST")
    ap.add_argument("--no-ws", action="store_true", help="dezactivează colectorul WebSocket")
    args = ap.parse_args()

    video = str(Path(args.video).resolve())
    if not os.path.isfile(video):
        ap.error(f"Fișier video inexistent: {video}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else \
        Path(__file__).resolve().parent / "results" / f"{args.mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rezultate în: {out_dir}")

    print("Autentificare...")
    be = Backend(args.host, args.user, args.password)
    print("Activez toți detectorii...")
    be.enable_all_detectors()

    # Time reference to isolate only this run's detections. The DB stores
    # created_at in naive LOCAL time, so we use local time (not UTC); otherwise
    # the filter would let older runs through.
    run_start = datetime.now()
    date_from = run_start.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Pornesc clipul ca {args.camera_id}...")
    info = be.start_file_stream(video, args.camera_id, args.loop)
    fps = info.get("fps")
    print(f"   FPS nativ: {fps}")

    stop = threading.Event()
    t0 = time.monotonic()
    pollers = [
        StatusPoller(be, out_dir / "status_timeseries.csv", stop, t0),
        ResourceSampler(out_dir / "resources_timeseries.csv", stop, t0),
    ]
    ws = None
    if not args.no_ws and _HAS_WS:
        ws = WSCollector(args.host, be.token, args.camera_id,
                         out_dir / "detections.jsonl", stop, t0)
        pollers.append(ws)
    for p in pollers:
        p.start()

    print("Rulez... (Ctrl-C pentru oprire)")
    try:
        while True:
            time.sleep(2.0)
            elapsed = time.monotonic() - t0
            if args.duration and elapsed >= args.duration:
                print(f"Atins --duration={args.duration}s")
                break
            if not args.loop and be.stream_finished(args.camera_id):
                print("Clipul s-a terminat (EOF).")
                # give the queues 2 more seconds to drain
                time.sleep(2.0)
                break
    except KeyboardInterrupt:
        print("\nÎntrerupt manual.")

    print("Opresc colectoarele și stream-ul...")
    stop.set()
    be.stop_stream(args.camera_id)
    for p in pollers:
        p.join(timeout=5.0)

    total_dur = round(time.monotonic() - t0, 1)
    print("Interoghez log-urile și alarmele din DB...")

    def _since_run(rows):
        """Keep only rows from this run (created_at >= run_start). Alarms have no
        date filter in the API, and the DB filter may be imprecise — so we filter
        here too, so each CSV/summary reflects a single run."""
        out = []
        for r in rows:
            ca = r.get("created_at")
            try:
                # accept "...T..." or "... ..."; ignore the timezone if present
                dt = datetime.fromisoformat(str(ca).replace("Z", "").split("+")[0])
            except Exception:
                out.append(r)          # if we cannot parse it, keep the row
                continue
            if dt >= run_start:
                out.append(r)
        return out

    logs = _since_run(be.get_logs(args.camera_id, date_from))
    alarms = _since_run(be.get_alarms(args.camera_id))
    write_csv(out_dir / "db_detection_logs.csv", logs,
              fieldnames=["id", "camera_id", "type", "subject", "confidence",
                          "severity", "status", "created_at"])
    write_csv(out_dir / "db_alarms.csv", alarms,
              fieldnames=["id", "camera_id", "type", "severity", "status",
                          "description", "created_at"])

    # scoring (sequential + annotations only)
    scoring = None
    events = ws.events if ws else []
    if args.mode == "sequential" and args.annotations:
        ann_path = Path(args.annotations)
        if ann_path.is_file():
            annotations = json.loads(ann_path.read_text())
            frame_origin = (ws.first_frame_t or 0.0) if ws else 0.0
            scoring = score_sequential(annotations, events, frame_origin=frame_origin)
            write_csv(out_dir / "scoring.csv", scoring["per_segment"])
            (out_dir / "scoring.json").write_text(json.dumps(scoring, indent=2, ensure_ascii=False))
            print("Scoring funcțional scris (scoring.json / scoring.csv).")
        else:
            print(f"  ⚠  Adnotări inexistente: {ann_path}")

    meta = {
        "video": video, "fps_native": fps, "mode": args.mode,
        "duration_sec": total_dur, "loop": args.loop,
        "ws_events": len(events), "db_logs": len(logs), "db_alarms": len(alarms),
        "timestamp": ts,
    }
    summary = build_summary(args.mode, pollers[0].rows, pollers[1].rows,
                            logs, alarms, events, scoring, meta)
    # GPU from the time series (mean/peak), not a single instant reading.
    gpu = summary.get("resources", {}).get("gpu_util_pct")
    vram = summary.get("resources", {}).get("vram_used_mb")
    if gpu:
        meta["gpu_util_pct_mean"] = gpu["mean"]
        meta["gpu_util_pct_max"] = gpu["max"]
    if vram:
        meta["vram_used_mb_max"] = vram["max"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    write_latex(out_dir / "summary.tex", args.mode, summary)

    print("\nGata. Fișiere generate:")
    for f in sorted(out_dir.iterdir()):
        print(f"   {f.name}")
    print(f"\nDurată totală: {total_dur}s  |  detecții WS: {len(events)}  |  "
          f"log-uri DB: {len(logs)}  |  alarme: {len(alarms)}")


if __name__ == "__main__":
    sys.exit(main())
