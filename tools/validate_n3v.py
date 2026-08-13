#!/usr/bin/env python
"""Validate a preprocessed data root before training on it.

Checks the things that fail SILENTLY -- a scene can look preprocessed and still be unusable:
  * dangling symlinks (N3DV's frames/ tree points into images/, which may not exist)
  * a near-empty points3D (triangulation "succeeded" but found nothing -> garbage reconstruction)
  * frame/camera count drift

Works for both datasets. Meet Room roots carry the `_undist` suffix and hold real PNGs rather than
symlinks, so the dangling count is always 0 there.

Usage:
    python tools/validate_n3v.py --root /path/to/n3v_recongs [scene ...]
    python tools/validate_n3v.py --root /path/to/meetroom_recongs discussion_undist trimming_undist vrheadset_undist

Env fallback for --root: N3V_OUT_ROOT
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from libgs.data.utils.colmap import read_points3D_binary
except ImportError:
    raise SystemExit("FATAL: cannot import libgs -- run with "
                     "PYTHONPATH=$PWD:$PWD/third_party/libgs (see README section 1)")

N3DV_SCENES = ["coffee_martini", "cook_spinach", "cut_roasted_beef",
               "flame_salmon_1", "flame_steak", "sear_steak"]


def check(root: Path, name: str, expect_frames: int):
    d = root / name
    r = {"scene": name, "ok": True, "notes": []}
    if not d.is_dir():
        return {**r, "ok": False, "notes": ["missing"]}

    frames = sorted((d / "frames").glob("[0-9]*")) if (d / "frames").is_dir() else []
    r["frames"] = len(frames)
    r["cams"] = len(list(frames[0].glob("*.png"))) if frames else 0

    # dangling links, sampled across the sequence (checking all 6300 is slow and adds nothing)
    dangling = 0
    for fr in frames[:: max(1, len(frames) // 10)] if frames else []:
        for p in fr.glob("*.png"):
            if not p.resolve().exists():
                dangling += 1
    r["dangling"] = dangling

    sp = d / "sparse" / "0"
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sp / f).exists():
            r["ok"] = False
            r["notes"].append(f"no {f}")
    try:
        xyz, _, _ = read_points3D_binary(sp / "points3D.bin")
        r["points"] = len(xyz)
    except Exception as e:
        r["points"] = 0
        r["notes"].append(f"points3D unreadable: {e}")

    if r["frames"] != expect_frames:
        r["ok"] = False; r["notes"].append(f"{r['frames']} frame dirs, expected {expect_frames}")
    if dangling:
        r["ok"] = False; r["notes"].append(f"{dangling} dangling symlinks")
    if r.get("points", 0) < 2000:
        r["ok"] = False; r["notes"].append(f"only {r.get('points')} points -- triangulation suspect")
    return r


def main():
    ap = argparse.ArgumentParser(description="Validate a preprocessed QuARC-GS data root.")
    ap.add_argument("scenes", nargs="*", help=f"default: the 6 N3DV scenes {N3DV_SCENES}")
    ap.add_argument("--root", default=os.environ.get("N3V_OUT_ROOT"),
                    help="dir holding <scene>/{frames,sparse}  [$N3V_OUT_ROOT]")
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    if not a.root:
        raise SystemExit("FATAL: --root is required (or set $N3V_OUT_ROOT)")
    root = Path(a.root).expanduser()

    names = a.scenes or N3DV_SCENES
    print(f"{'scene':20s} {'frames':>7s} {'cams':>5s} {'points3D':>9s} {'dangling':>9s}  status")
    print("-" * 72)
    bad = 0
    for n in names:
        r = check(root, n, a.frames)
        bad += not r["ok"]
        print(f"{r['scene']:20s} {r.get('frames',0):7d} {r.get('cams',0):5d} "
              f"{r.get('points',0):9,d} {r.get('dangling',0):9d}  "
              f"{'OK' if r['ok'] else 'FAIL: ' + '; '.join(r['notes'])}")
    print()
    print("all good" if not bad else f"{bad} scene(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
