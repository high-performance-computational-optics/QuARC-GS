#!/usr/bin/env python
"""Meet Room stage 2: undistort all 300 frames x 13 cams and assemble the training root.

Reuses the OPENCV->PINHOLE rectification implied by the frame-0 model that tools/prepare_meetroom.py
built, applying it with cv2.remap. COLMAP's own image_undistorter would give identical pixels but
would have to re-run per frame; the maps are constant because the rig is static, so one
initUndistortRectifyMap serves all 3,900 images. The agreement is CHECKED, not assumed: frame 0 is
remapped and compared against COLMAP's own output, and the script aborts below 38 dB.

    <work-root>/<scene>_undist_colmap/sparse/0/cameras.bin    IN: OPENCV intrinsics (distorted)
    <work-root>/<scene>_undist_colmap/undist/sparse/          IN: PINHOLE model + poses
    <work-root>/<scene>/images/camNN/NNNN.png                 IN: all extracted frames
    <out-root>/<scene>_undist/frames/NNNN/camNN.png           OUT: undistorted frames
    <out-root>/<scene>_undist/sparse/0/{cameras,images,points3D}.bin  OUT: the PINHOLE model

`<out-root>/<scene>_undist` is what you pass as `data.root`. Note the `_undist` suffix is part of the
name the shipped config expects -- see config/quarc_gs_meetroom.yaml.

Unlike N3DV this writes REAL PNGs, not symlinks: the undistorted pixels do not exist anywhere else.
Budget ~3,900 PNGs per scene.

Usage:
    python tools/undistort_meetroom.py <scene|all> \\
        --work-root /path/to/meetroom_work \\
        --out-root  /path/to/meetroom_recongs \\
        [--frames 300] [--workers 12]

Env fallbacks: MEETROOM_WORK_ROOT  MEETROOM_OUT_ROOT
"""

import argparse
import glob
import os
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

SCENES = ["discussion", "trimming", "vrheadset"]


def read_cameras_bin(p):
    NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
    cams = {}
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cid, model = struct.unpack('<ii', f.read(8))
            w, h = struct.unpack('<QQ', f.read(16))
            params = struct.unpack('<%dd' % NP[model], f.read(8 * NP[model]))
            cams[cid] = (model, w, h, params)
    return cams


def build_maps(work: Path):
    d = list(read_cameras_bin(work / "sparse/0/cameras.bin").values())[0]       # OPENCV: fx fy cx cy k1 k2 p1 p2
    u = list(read_cameras_bin(work / "undist/sparse/cameras.bin").values())[0]  # PINHOLE: fx fy cx cy
    dp, up = d[3], u[3]
    uw, uh = int(u[1]), int(u[2])
    K = np.array([[dp[0], 0, dp[2]], [0, dp[1], dp[3]], [0, 0, 1]], float)
    D = np.array([dp[4], dp[5], dp[6], dp[7]], float)
    Knew = np.array([[up[0], 0, up[2]], [0, up[1], up[3]], [0, 0, 1]], float)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Knew, (uw, uh), cv2.CV_32FC1)
    return m1, m2, (uw, uh)


def do_frame(fr, cams, src, out, m1p, m2p):
    # The maps are ~3 MB each; loading them per task beats pickling them to every worker.
    m1, m2 = np.load(m1p), np.load(m2p)
    d = Path(out) / f"{fr:04d}"
    d.mkdir(parents=True, exist_ok=True)
    if len(glob.glob(f"{d}/*.png")) >= len(cams):
        return
    for cam in cams:
        img = cv2.imread(f"{src}/{cam}/{fr:04d}.png")
        if img is not None:
            cv2.imwrite(str(d / f"{cam}.png"), cv2.remap(img, m1, m2, cv2.INTER_LINEAR))


def process(scene: str, work_root: Path, out_root: Path, num_frames: int, workers: int):
    work = work_root / f"{scene}_undist_colmap"
    src = work_root / scene / "images"
    dst = out_root / f"{scene}_undist"
    out = dst / "frames"

    if not (work / "undist/sparse/cameras.bin").exists():
        raise SystemExit(f"FATAL {scene}: {work}/undist/ missing -- run tools/prepare_meetroom.py first")

    m1, m2, size = build_maps(work)

    # Verify cv2's rectification against COLMAP's own frame-0 output before spending an hour on
    # 3,900 images. A mismatch here means the two disagree on the undistortion model, and every
    # downstream pixel would be subtly wrong while still looking plausible.
    mine = cv2.remap(cv2.imread(str(work / "images/cam00.png")), m1, m2, cv2.INTER_LINEAR)
    ref = cv2.imread(str(work / "undist/images/cam00.png"))
    mse = np.mean((mine.astype(float) - ref.astype(float)) ** 2)
    psnr = 99.0 if mse < 1e-9 else 10 * np.log10(255 ** 2 / mse)
    print(f"{scene}: undistorted size {size}  cv2-vs-COLMAP {psnr:.1f} dB", flush=True)
    if psnr < 38:
        raise SystemExit(f"ABORT {scene}: cv2 undistort does not match COLMAP ({psnr:.1f} dB) -- "
                         f"fall back to running image_undistorter per frame")

    cams = sorted(os.path.basename(c.rstrip('/')) for c in glob.glob(f"{src}/cam*"))
    if not cams:
        raise SystemExit(f"FATAL {scene}: no extracted frames under {src}")
    m1p, m2p = work / "m1.npy", work / "m2.npy"
    np.save(m1p, m1)
    np.save(m2p, m2)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(partial(do_frame, cams=cams, src=str(src), out=str(out),
                            m1p=str(m1p), m2p=str(m2p)), range(num_frames)))

    n = len(glob.glob(f"{out}/*/"))
    if n < num_frames:
        raise SystemExit(f"FATAL {scene}: only {n}/{num_frames} frame dirs written")

    # The PINHOLE model from image_undistorter IS the training model -- same poses, distortion-free
    # intrinsics matching the pixels we just wrote. libgs derives points3D.ply from the .bin on first
    # load, so only the three .bin files are copied.
    model = dst / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.copy(work / "undist" / "sparse" / f, model / f)

    print(f"  {scene}: {n}/{num_frames} frame dirs x {len(cams)} cams, sparse/0 written", flush=True)
    print(f"  data.root={dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Undistort all Meet Room frames and assemble the QuARC-GS training root.")
    ap.add_argument("scene", help=f"one of {SCENES}, or 'all'")
    ap.add_argument("--work-root", default=os.environ.get("MEETROOM_WORK_ROOT"),
                    help="where tools/prepare_meetroom.py wrote its output  [$MEETROOM_WORK_ROOT]")
    ap.add_argument("--out-root", default=os.environ.get("MEETROOM_OUT_ROOT"),
                    help="dir to write <scene>_undist/  [$MEETROOM_OUT_ROOT]")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    if not a.work_root or not a.out_root:
        raise SystemExit("FATAL: --work-root and --out-root are required "
                         "(or set $MEETROOM_WORK_ROOT / $MEETROOM_OUT_ROOT)")
    work_root, out_root = Path(a.work_root).expanduser(), Path(a.out_root).expanduser()
    for scene in (SCENES if a.scene == "all" else [a.scene]):
        process(scene, work_root, out_root, a.frames, a.workers)
    print("UNDIST_DONE", flush=True)


if __name__ == "__main__":
    main()
