#!/usr/bin/env python
"""Collect the full per-frame metric set for one or more runs into a single JSON + CSV.

Everything is read from the run's TensorBoard event file -- no re-running, no re-deriving. Emits BOTH
the per-frame series (so PSNR-over-frame / storage-over-frame can be plotted and drift fitted) and the
summary each run reduces to.

Storage note: `storage/total_kb` is a Shannon ENTROPY ESTIMATE of the quantized payload, and it is
the ONLY storage axis this pipeline reports. It excludes the CDF/symbol tables a real range coder must
also emit, so an actual coded file runs ~8-20% larger -- the gap widens as the per-frame payload
shrinks, since that side-info is roughly fixed. Always quote it as an estimate, never as a file size.
The keyframe term dominates the amortized cost -- at 300 frames it is ~76% of the bill -- so
`total_clip_mb` and `avg_kb_per_frame` below include it explicitly rather than letting the per-frame
number stand in for the clip cost.

Usage:
    python tools/collect_metrics.py --out results.json  <run_dir> [<run_dir> ...]
    python tools/collect_metrics.py --out results.json --glob '/path/to/output/*/*/'
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import os
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# per-frame series we want out of every run
SERIES = {
    "psnr": "eval-test/psnr",
    "ssim": "eval-test/ssim",
    "lpips": "eval-test/lpips",
    "fps": "eval-test/fps",
    "incr_gs": "storage/appearance_n_gaussians",
    "n_dynamic": "storage/motion_n_dynamic",
    "n_static": "storage/motion_n_static",
    # Entropy ESTIMATE (Shannon -sum n*log2 p over rounded residuals). This is the only storage axis
    # the pipeline logs and the number we report. total_kb = motion + appearance per frame.
    "motion_kb": "storage/motion_kb",
    "appearance_kb": "storage/appearance_kb",
    "total_kb": "storage/total_kb",
    "total_amortized_kb": "storage/total_amortized_kb",
    "base_keyframe_kb": "storage/base_keyframe_kb",
}


def frame_times(run: Path) -> list:
    """Per-frame TRAIN time, parsed from the pipeline's own timer in the INFO log.

    The timer prints `frame: <sec> +/- ...` once per frame. This is train time only -- it excludes the
    storage accounting, which is instrumentation, not method cost.
    """
    out = []
    for lg in list(run.glob("logging.INFO")) + list(run.glob("logging.*.log.INFO.*")):
        try:
            txt = Path(lg).resolve().read_text(errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("frame:"):
                try:
                    out.append(float(s.split(":")[1].split("±")[0].strip()))
                except Exception:
                    pass
        if out:
            break
    return out


def collect(run: Path) -> dict:
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    rec = {"run": str(run), "name": run.parent.name}

    got = {}
    for key, tag in SERIES.items():
        got[key] = [s.value for s in ea.Scalars(tag)] if tag in tags else []
    rec["series"] = got

    ft = frame_times(run)
    rec["series"]["frame_train_s"] = ft

    p = np.asarray(got["psnr"], dtype=float)
    rec["n_frames"] = int(len(p))
    if len(p) == 0:
        rec["status"] = "NO eval-test/psnr — run produced no test metrics"
        return rec
    rec["status"] = "ok"

    def mean(k):
        v = got.get(k) or []
        return float(np.mean(v)) if len(v) else None

    rec["summary"] = {
        "psnr": float(p.mean()),
        "ssim": mean("ssim"),
        "lpips": mean("lpips"),
        "fps": mean("fps"),
        "incr_gs": mean("incr_gs"),
        "n_dynamic": mean("n_dynamic"),
        "train_s_per_frame": float(np.mean(ft)) if ft else None,
        # Entropy-estimate storage is the reported per-frame number, and the only one.
        "kb_per_frame": mean("total_kb"),
        # raw PSNR slope. NOTE: this includes the SCENE's own drift -- the baseline has it too
        # (measured -0.166 dB/100f on cut_roasted_beef). Only a slope measured against a matched
        # baseline is a statement about the codec. Cross-scene, compare slopes to each other.
        "psnr_slope_db_per_100f": float(np.polyfit(np.arange(len(p)), p, 1)[0] * 100),
    }

    # Whole-clip cost, including the one-time keyframe. The per-frame number alone understates the
    # real cost by ~4x at 300 frames, because the keyframe is ~76% of it.
    n = len(p) + 1  # inter-frames + the keyframe itself
    base = mean("base_keyframe_kb")
    pf = mean("total_kb")
    if base and pf:
        total_kb = base + len(p) * pf
        rec["summary"]["base_keyframe_kb"] = base
        rec["summary"]["total_clip_mb"] = total_kb / 1024
        rec["summary"]["avg_kb_per_frame_incl_keyframe"] = total_kb / n
        rec["summary"]["keyframe_share_pct"] = 100 * base / total_kb
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    runs = list(a.runs)
    if a.glob:
        runs += [Path(p) for p in sorted(globmod.glob(a.glob))]
    runs = [r for r in runs if (r / "config.yaml").exists() or list(r.glob("events.out.tfevents*"))]
    if not runs:
        raise SystemExit("no runs found")

    recs = []
    for r in runs:
        print(f"==> {r}")
        try:
            recs.append(collect(r))
        except Exception as e:
            print(f"    FAILED: {e}")
            recs.append({"run": str(r), "status": f"FAILED: {e}"})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(a.out, "w"), indent=1)

    cols = ["name", "n_frames", "psnr", "ssim", "lpips", "fps", "train_s_per_frame",
            "kb_per_frame", "incr_gs", "psnr_slope_db_per_100f",
            "base_keyframe_kb", "total_clip_mb", "avg_kb_per_frame_incl_keyframe"]
    csv_path = a.out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            s = r.get("summary", {})
            w.writerow([r.get("name", "?"), r.get("n_frames", 0)] +
                       [("" if s.get(c) is None else round(s[c], 4)) for c in cols[2:]])

    print(f"\nwrote {a.out}  and  {csv_path}\n")
    # KB/f is the entropy ESTIMATE (Shannon) -- the only storage axis reported.
    hdr = (f"{'scene':22s} {'n':>4s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'FPS':>6s} "
           f"{'s/frame':>8s} {'KB/f est':>9s} {'clip MB':>8s} {'slope':>7s}")
    print(hdr); print("-" * len(hdr))
    for r in recs:
        s = r.get("summary")
        if not s:
            print(f"{r.get('name','?'):22s}  {r.get('status','?')}")
            continue
        f2 = lambda v, d=2: ("     -" if v is None else f"{v:.{d}f}")
        print(f"{r.get('name','?'):22s} {r['n_frames']:4d} {f2(s['psnr'],3):>7s} {f2(s['ssim'],4):>7s} "
              f"{f2(s['lpips'],4):>7s} {f2(s['fps'],1):>6s} {f2(s['train_s_per_frame']):>8s} "
              f"{f2(s['kb_per_frame']):>9s} {f2(s.get('total_clip_mb')):>8s} "
              f"{f2(s['psnr_slope_db_per_100f'],3):>7s}")


if __name__ == "__main__":
    main()
