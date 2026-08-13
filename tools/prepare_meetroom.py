#!/usr/bin/env python
"""Meet Room stage 1: raw videos -> frames -> COLMAP SfM (OPENCV) -> image_undistorter.

Meet Room ships no poses, so unlike N3DV this runs a REAL reconstruction (`colmap mapper`) rather
than triangulating into known poses. It also has real lens distortion (k1~0.084 on discussion), which
must be removed before training -- 3DGStream and HiCoM do the same and 3DGStream calls it *critical*.
Training on the distorted build scores ~3.7 dB lower and reads as a method regression.

    <raw-root>/<scene>/cam_N.mp4                    INPUT: 13 videos, note the cam_N naming
    <work-root>/<scene>/camNN.mp4                   normalised names (symlinks)
    <work-root>/<scene>/images/camNN/NNNN.png       extracted frames
    <work-root>/<scene>_undist_colmap/
      images/camNN.png                              frame 0 only -- SfM input
      sparse/0/{cameras,images,points3D}.bin        OPENCV model (distorted)
      undist/{images,sparse}/                       image_undistorter output (PINHOLE)

Run tools/undistort_meetroom.py next: it reuses the OPENCV->PINHOLE maps from this model to undistort
all 300 frames and assembles the training root.

Only frame 0 goes through SfM: the rig is static, so one frame's reconstruction serves all 300.

!! THE MODEL AND THE FRAME-0 CHECKPOINT MUST TRAVEL TOGETHER. A COLMAP reconstruction is determined
only up to a similarity transform (arbitrary scale, rotation, origin), so `mapper` lands in a
different world frame every time it runs. A frame-0 checkpoint stores Gaussian POSITIONS, so it is
only valid against the model it was trained on. If you rebuild the model, retrain frame 0 too --
mixing a checkpoint with a freshly built model scores ~12.8 dB instead of ~30.8.
N3DV does not have this problem: its world frame comes from poses_bounds.npy, not from a solver.

Usage:
    python tools/prepare_meetroom.py <scene> \\
        --raw-root  /path/to/meetroom_raw \\
        --work-root /path/to/meetroom_work \\
        [--colmap colmap] [--frames 300]

Env fallbacks: MEETROOM_RAW_ROOT  MEETROOM_WORK_ROOT  COLMAP_BIN
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCENES = ["discussion", "trimming", "vrheadset"]


def _run(cmd, what, env=None):
    print(f"  $ {Path(str(cmd[0])).name} {cmd[1] if len(cmd) > 1 else ''}", flush=True)
    # errors="replace": COLMAP emits Windows-1252 bytes in some messages, and text=True would raise
    # UnicodeDecodeError *while reporting a failure*, masking the real error with a decode traceback.
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"FATAL: {what} failed")
    return r


def normalise_cams(raw_scene: Path, work_scene: Path):
    """`cam_0.mp4 .. cam_12.mp4` -> `cam00.mp4 .. cam12.mp4` as symlinks.

    The zero padding is not cosmetic: every downstream consumer (this script, undistort_meetroom.py,
    libgs's COLMAP loader, and `eval_pose_indices`) orders cameras by SORTED NAME. Unpadded names
    sort as cam_0, cam_1, cam_10, cam_11, cam_12, cam_2, ... which silently permutes the held-out
    camera and every pose<->image pairing.
    """
    work_scene.mkdir(parents=True, exist_ok=True)
    vids = sorted(raw_scene.glob("cam*.mp4"))
    if not vids:
        raise SystemExit(f"FATAL: no cam*.mp4 under {raw_scene}")
    cams = []
    for v in vids:
        m = re.fullmatch(r"cam_?(\d+)", v.stem)
        if not m:
            raise SystemExit(f"FATAL: cannot parse camera index from {v.name}")
        name = f"cam{int(m.group(1)):02d}"
        link = work_scene / f"{name}.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(v.resolve())
        cams.append(name)
    cams = sorted(cams)
    if len(set(cams)) != len(cams):
        raise SystemExit(f"FATAL: duplicate camera indices in {raw_scene}")
    print(f"  {len(cams)} cameras: {cams[0]}..{cams[-1]}", flush=True)
    return cams


def _extract_one(args):
    import cv2
    from PIL import Image

    video, out_dir, num_frames = args
    out_dir.mkdir(parents=True, exist_ok=True)
    done = len(list(out_dir.glob("*.png")))
    if done >= num_frames:
        return video.stem, done, "cached"
    cap = cv2.VideoCapture(str(video))
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, image = cap.read()
        if not ret:
            break
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(out_dir / f"{count:04d}.png")
        count += 1
    cap.release()
    return video.stem, count, "extracted"


def extract_frames(work_scene: Path, cams, num_frames: int, workers: int = 12):
    jobs = [(work_scene / f"{c}.mp4", work_scene / "images" / c, num_frames) for c in cams]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for cam, n, how in ex.map(_extract_one, jobs):
            print(f"  {cam}: {n} frames ({how})", flush=True)
            if n < num_frames:
                raise SystemExit(f"FATAL {work_scene.name}/{cam}: only {n}/{num_frames} frames")


def run_sfm(work_scene: Path, cdir: Path, cams, colmap: str):
    """Frame-0 SfM with ONE shared OPENCV camera, then image_undistorter.

    `--ImageReader.single_camera 1` is load-bearing: the 13 Meet Room cameras are the same physical
    model, and sharing one intrinsic block gives the distortion coefficients 13x the observations to
    fit. Per-camera intrinsics on 13 images each are badly conditioned and the undistortion comes out
    visibly wrong. `OPENCV` (fx fy cx cy k1 k2 p1 p2) is the model image_undistorter expects to
    invert into PINHOLE.
    """
    if cdir.exists():
        shutil.rmtree(cdir)
    (cdir / "images").mkdir(parents=True)
    for c in cams:
        shutil.copy(work_scene / "images" / c / "0000.png", cdir / "images" / f"{c}.png")

    db = cdir / "database.db"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("COLMAP_GPU", "0"))

    _run([colmap, "feature_extractor",
          "--database_path", str(db),
          "--image_path", str(cdir / "images"),
          "--ImageReader.camera_model", "OPENCV",
          "--ImageReader.single_camera", "1",
          "--SiftExtraction.max_image_size", "4096",
          "--SiftExtraction.max_num_features", "16384",
          "--SiftExtraction.estimate_affine_shape", "1",
          "--SiftExtraction.domain_size_pooling", "1"], "feature_extractor", env)

    _run([colmap, "exhaustive_matcher", "--database_path", str(db)], "exhaustive_matcher", env)

    (cdir / "sparse").mkdir(exist_ok=True)
    _run([colmap, "mapper",
          "--database_path", str(db),
          "--image_path", str(cdir / "images"),
          "--output_path", str(cdir / "sparse")], "mapper", env)

    model = cdir / "sparse" / "0"
    if not (model / "points3D.bin").exists():
        raise SystemExit(f"FATAL {cdir.name}: mapper produced no sparse/0 model")
    # mapper can return a PARTIAL reconstruction (a subset of images) without failing. That yields a
    # model which loads fine and reconstructs a scene missing cameras, so check the count explicitly.
    n_reg = _n_images(model / "images.bin")
    if n_reg != len(cams):
        raise SystemExit(
            f"FATAL {cdir.name}: mapper registered {n_reg}/{len(cams)} images. "
            f"A partial reconstruction is unusable here -- every camera must be posed.")
    print(f"  sparse/0: {n_reg}/{len(cams)} images registered", flush=True)

    _run([colmap, "image_undistorter",
          "--image_path", str(cdir / "images"),
          "--input_path", str(model),
          "--output_path", str(cdir / "undist"),
          "--output_type", "COLMAP"], "image_undistorter", env)

    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        if not (cdir / "undist" / "sparse" / f).exists():
            raise SystemExit(f"FATAL {cdir.name}: image_undistorter produced no undist/sparse/{f}")
    print(f"  undist/ ok", flush=True)


def _n_images(p: Path) -> int:
    import struct
    with open(p, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def main():
    ap = argparse.ArgumentParser(
        description="Meet Room raw videos -> frames + COLMAP OPENCV model + image_undistorter output.")
    ap.add_argument("scene", help=f"one of {SCENES}, or 'all'")
    ap.add_argument("--raw-root", default=os.environ.get("MEETROOM_RAW_ROOT"),
                    help="dir holding <scene>/cam_N.mp4  [$MEETROOM_RAW_ROOT]")
    ap.add_argument("--work-root", default=os.environ.get("MEETROOM_WORK_ROOT"),
                    help="dir to write frames + COLMAP models  [$MEETROOM_WORK_ROOT]")
    ap.add_argument("--colmap", default=os.environ.get("COLMAP_BIN", "colmap"),
                    help="colmap executable  [$COLMAP_BIN]")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()

    if not a.raw_root or not a.work_root:
        raise SystemExit("FATAL: --raw-root and --work-root are required "
                         "(or set $MEETROOM_RAW_ROOT / $MEETROOM_WORK_ROOT)")
    raw_root, work_root = Path(a.raw_root).expanduser(), Path(a.work_root).expanduser()
    scenes = SCENES if a.scene == "all" else [a.scene]

    for scene in scenes:
        raw_scene = raw_root / scene
        if not raw_scene.is_dir():
            raise SystemExit(f"no such scene: {raw_scene}")
        work_scene = work_root / scene
        cdir = work_root / f"{scene}_undist_colmap"

        t0 = time.time()
        print(f"=== {scene}: {a.frames} frames ===", flush=True)
        print("[1/3] normalise camera names", flush=True)
        cams = normalise_cams(raw_scene, work_scene)
        if not a.skip_extract:
            print("[2/3] extract", flush=True)
            extract_frames(work_scene, cams, a.frames)
        print("[3/3] colmap SfM + undistort", flush=True)
        run_sfm(work_scene, cdir, cams, a.colmap)
        print(f"=== {scene} DONE in {(time.time()-t0)/60:.1f} min ===\n", flush=True)

    print("Next: python tools/undistort_meetroom.py <scene> "
          "--work-root <work> --out-root <out>", flush=True)


if __name__ == "__main__":
    main()
