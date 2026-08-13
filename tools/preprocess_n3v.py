#!/usr/bin/env python
"""Preprocess an N3V (Neural 3D Video / DyNeRF) scene into the layout QuARC-GS loads.

    <raw-root>/<scene>/camXX.mp4                       INPUT: per-camera videos
    <raw-root>/<scene>/poses_bounds.npy                INPUT: LLFF poses
    <raw-root>/<scene>/images/camXX/NNNN.png           real PNGs, 300/cam (written here)
    <out-root>/<scene>/frames/NNNN/camXX.png           SYMLINKS (transposed layout)
    <out-root>/<scene>/sparse/0/*.bin                  COLMAP model

The two layouts are transposes of each other; the symlink tree does the transposition so the 26 GB of
PNGs are stored once. `libgs/data/dataset/colmap.py` reads `<root>/sparse/0` and auto-derives
points3D.ply from points3D.bin on first load. `<out-root>/<scene>` is what you pass as `data.root`.

POSES, NOT SfM. N3V ships known camera poses in poses_bounds.npy (LLFF format), so we do NOT run
`colmap mapper`. We convert the poses to a COLMAP text model and run `point_triangulator` against it,
which only needs to triangulate points into a KNOWN pose graph. Camera ids are non-contiguous in most
scenes (coffee_martini has no cam03/15/17), and poses_bounds rows correspond to the SORTED camera
list -- verified per scene by _check_counts() rather than assumed.
The world frame is fixed by poses_bounds, not chosen by an SfM solver, so a fresh run lands in the
same coordinates every time -- a frame-0 checkpoint stays valid if you regenerate this model.
(Meet Room is the opposite case -- see tools/prepare_meetroom.py.)

Only frame 0 goes through COLMAP: the rig is static, so one frame's triangulation serves all 300.

Usage:
    python tools/preprocess_n3v.py <scene> \\
        --raw-root  /path/to/n3v_raw \\
        --out-root  /path/to/n3v_recongs \\
        --libgs-tool third_party/libgs/tool \\
        [--colmap colmap] [--python python] [--frames 300] [--skip-extract] [--skip-colmap]

Every path flag also reads an environment variable, so you can export once and omit them:
    N3V_RAW_ROOT  N3V_OUT_ROOT  LIBGS_TOOL  COLMAP_BIN  PYTHON_BIN
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Resolved in main() from CLI flags / env vars. Module-level so the helpers below can read them
# without threading five arguments through every call.
N3V = RECONGS = LIBGS_TOOL = None
COLMAP = PYTHON = None
load_poses = rotmat2qvec = None


def _bind_libgs(libgs_tool: Path):
    """`llff2colmap` lives in libgs/tool/, which is not a package -- import it by path.

    It provides the LLFF->COLMAP pose algebra. We import it rather than vendoring it so the two
    stay in lockstep; libgs is a required dependency of this repo anyway (see README §1).
    """
    global load_poses, rotmat2qvec
    if not (libgs_tool / "llff2colmap.py").exists():
        raise SystemExit(
            f"FATAL: {libgs_tool}/llff2colmap.py not found.\n"
            f"       Point --libgs-tool (or $LIBGS_TOOL) at libgs/tool/ -- see README section 1a.")
    sys.path.insert(0, str(libgs_tool))
    from llff2colmap import load_poses as _lp, rotmat2qvec as _rq  # noqa: E402
    load_poses, rotmat2qvec = _lp, _rq


def _cams(scene: Path):
    return sorted(p.stem for p in scene.glob("cam*.mp4"))


def _check_counts(scene: Path):
    """A pose row count that disagrees with the video count means the sorted-zip pairing below is
    wrong, which would silently produce a plausible-looking but misaligned model. Fail loudly."""
    cams = _cams(scene)
    n_pose = np.load(scene / "poses_bounds.npy").shape[0]
    if len(cams) != n_pose:
        raise SystemExit(
            f"FATAL {scene.name}: {len(cams)} videos but {n_pose} poses_bounds rows. "
            f"The pose<->camera pairing is order-based; refusing to guess.")
    return cams


def _extract_one(args):
    """One video -> images/camXX/NNNN.png at full resolution. Mirrors libgs/tool/video.py's
    cv2+PIL logic (BGR->RGB->PIL), but writes the images/camXX/NNNN.png layout, not frames/."""
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


def extract_frames(scene: Path, cams, num_frames: int, workers: int = 12):
    jobs = [(scene / f"{c}.mp4", scene / "images" / c, num_frames) for c in cams]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for cam, n, how in ex.map(_extract_one, jobs):
            print(f"  {cam}: {n} frames ({how})", flush=True)
            if n < num_frames:
                raise SystemExit(f"FATAL {scene.name}/{cam}: only {n}/{num_frames} frames")


def link_frames(scene: Path, dst: Path, cams, num_frames: int):
    for f in range(num_frames):
        d = dst / "frames" / f"{f:04d}"
        d.mkdir(parents=True, exist_ok=True)
        for c in cams:
            link, target = d / f"{c}.png", scene / "images" / c / f"{f:04d}.png"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
    print(f"  linked {num_frames} x {len(cams)} = {num_frames * len(cams)} symlinks", flush=True)


def _write_model(scene: Path, cdir: Path, out: Path, cams):
    """Write the COLMAP TEXT model that point_triangulator consumes, aligned to the database.

    Three things here are load-bearing and were each found the hard way against COLMAP 3.13:

    1. IDS COME FROM THE DATABASE, not from sorted order. feature_extractor assigns image_ids in
       extraction order (cam02 -> 1, cam01 -> 2, cam00 -> 5 ...), so llff2colmap's "pose row i <->
       sorted image i" numbering silently disagrees with the db. We read the db's own ids and look
       poses up BY NAME.
    2. rigs.txt + frames.txt ARE REQUIRED. COLMAP >= 3.12 has a rig/frame model; the db already
       holds one rig + one frame per image. A text model lacking them gets a conflicting implicit
       assignment and the loader aborts with
           Check failed: existing_frame.RigId() == frame.RigId()
       We mirror the db: rig_id == camera_id, frame_id == image_id. The reference scene's model
       (cut_roasted_beef) has exactly this shape.
    3. THE CAMERA MODEL MUST MATCH THE DB. feature_extractor creates SIMPLE_RADIAL cameras; writing
       PINHOLE here aborts with
           Check failed: existing_camera.model_id == camera.model_id (kPinhole vs. kSimpleRadial)
       until database.py rewrites the db cameras from this file -- so database.py must run AFTER
       this, and before the matcher.
    """
    import sqlite3

    db = sqlite3.connect(cdir / "database.db")
    imgs = list(db.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id"))
    rigs = list(db.execute("SELECT rig_id, ref_sensor_id FROM rigs ORDER BY rig_id"))
    frames = {r[0]: r[1] for r in db.execute("SELECT frame_id, rig_id FROM frames")}
    fdata = {r[0]: (r[1], r[2]) for r in db.execute("SELECT frame_id, data_id, sensor_id FROM frame_data")}
    w, h = db.execute("SELECT width, height FROM cameras LIMIT 1").fetchone()
    db.close()

    poses, H, W, focal, _ = load_poses(scene)
    pose_of = {c: poses[i] for i, c in enumerate(cams)}      # poses_bounds row i <-> sorted cam i
    f = focal / (W / w)

    def qt(name):
        """LLFF pose -> COLMAP world-from-camera qvec/tvec (same algebra as llff2colmap)."""
        pose = pose_of[Path(name).stem]
        R = pose[:3, :3].copy(); R = -R; R[:, 0] = -R[:, 0]
        T = pose[:3, 3]
        R = np.linalg.inv(R); T = -np.matmul(R, T)
        return rotmat2qvec(R), T

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cameras.txt", "w") as fh:
        for cid, _ in rigs:
            print(cid, "PINHOLE", w, h, f, f, w / 2, h / 2, file=fh)
    with open(out / "rigs.txt", "w") as fh:
        for rid, ref in rigs:                                # one rig per camera, 1 sensor each
            print(rid, 1, "CAMERA", ref, file=fh)
    with open(out / "images.txt", "w") as fh:
        for iid, name, cid in imgs:
            q, T = qt(name)
            print(iid, *[repr(float(x)) for x in q], *[repr(float(x)) for x in T], cid, name, file=fh)
            print("", file=fh)                               # empty POINTS2D line
    with open(out / "frames.txt", "w") as fh:
        for fid, rid in sorted(frames.items()):
            data_id, sensor_id = fdata[fid]
            name = next(n for i, n, _ in imgs if i == data_id)
            q, T = qt(name)
            print(fid, rid, *[repr(float(x)) for x in q], *[repr(float(x)) for x in T],
                  1, "CAMERA", sensor_id, data_id, file=fh)
    (out / "points3D.txt").write_text("")                    # triangulator fills this in
    print(f"  model: {len(rigs)} rigs/cameras, {len(imgs)} images, {len(frames)} frames", flush=True)


def run_colmap(scene: Path, dst: Path, cams):
    """Poses -> text model -> triangulate frame 0. Trimmed from libgs/tool/rebuild_colmap.sh:
    that script also runs image_undistorter + patch_match_stereo + stereo_fusion (dense MVS), which
    takes hours and produces nothing we use -- QuARC-GS only needs sparse/0.

    Only frame 0 is triangulated: the rig is static, so one frame's points serve all 300."""
    cdir = scene / "colmap"
    if cdir.exists():
        shutil.rmtree(cdir)
    (cdir / "images").mkdir(parents=True)

    # frame-0 images, materialised (COLMAP wants real files; the frames/ entries are symlinks)
    for c in cams:
        shutil.copy(scene / "images" / c / "0000.png", cdir / "images" / f"{c}.png")

    custom = cdir / "sparse_custom"
    db = cdir / "database.db"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("COLMAP_GPU", "0"))

    # feature_extractor must run FIRST: it creates the db (ids, rigs, frames) that the text model
    # below has to agree with.
    # Feature density is env-tunable (denser-init experiments): COLMAP_MAX_FEATURES raises the SIFT
    # cap, COLMAP_PEAK_THRESHOLD lowers the detection threshold to pull more points off low-texture
    # surfaces. Defaults reproduce the original behaviour exactly.
    maxf = os.environ.get("COLMAP_MAX_FEATURES", "16384")
    peak = os.environ.get("COLMAP_PEAK_THRESHOLD", "")
    extract = [COLMAP, "feature_extractor", "--database_path", str(db),
               "--image_path", str(cdir / "images"),
               "--SiftExtraction.max_image_size", "4096",
               "--SiftExtraction.max_num_features", maxf,
               "--SiftExtraction.estimate_affine_shape", "1",
               "--SiftExtraction.domain_size_pooling", "1"]
    if peak:
        extract += ["--SiftExtraction.peak_threshold", peak]
    print(f"  $ colmap feature_extractor", flush=True)
    r = subprocess.run(extract, env=env, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"FATAL {scene.name}: feature_extractor failed")

    _write_model(scene, cdir, custom, cams)

    steps = [
        [PYTHON, str(LIBGS_TOOL / "database.py"),           # rewrite db cameras -> PINHOLE + known f
         "--database_path", str(db), "--txt_path", str(custom / "cameras.txt")],
        [COLMAP, "exhaustive_matcher", "--database_path", str(db)],
        [COLMAP, "point_triangulator", "--database_path", str(db),
         "--image_path", str(cdir / "images"), "--input_path", str(custom),
         "--output_path", str(cdir / "sparse" / "0"), "--clear_points", "1"],
    ]
    (cdir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for cmd in steps:
        print(f"  $ {Path(cmd[0]).name} {cmd[1] if len(cmd) > 1 else ''}", flush=True)
        # errors="replace": COLMAP emits Windows-1252 bytes (0x92, a curly quote) in some messages,
        # and text=True would raise UnicodeDecodeError *while reporting a failure* -- masking the
        # actual error with a decode traceback.
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(r.stdout[-4000:]); print(r.stderr[-4000:])
            raise SystemExit(f"FATAL {scene.name}: {Path(cmd[0]).name} {cmd[1]} failed")

    out = dst / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = cdir / "sparse" / "0" / f
        if not src.exists():
            raise SystemExit(f"FATAL {scene.name}: triangulation produced no {f}")
        shutil.copy(src, out / f)
    sz = (out / "points3D.bin").stat().st_size
    # An empty/near-empty points3D means triangulation silently found nothing -- the main failure
    # mode here, and it yields a scene that loads but reconstructs garbage.
    if sz < 50_000:
        raise SystemExit(f"FATAL {scene.name}: points3D.bin is only {sz} B -- triangulation failed")
    print(f"  sparse/0 ok (points3D.bin {sz/1024:.0f} KB)", flush=True)


def main():
    global N3V, RECONGS, LIBGS_TOOL, COLMAP, PYTHON

    ap = argparse.ArgumentParser(
        description="N3DV raw videos + LLFF poses -> the frames/ + sparse/0 layout QuARC-GS loads.")
    ap.add_argument("scene", help="e.g. cut_roasted_beef")
    ap.add_argument("--raw-root", default=os.environ.get("N3V_RAW_ROOT"),
                    help="dir holding <scene>/camXX.mp4 + poses_bounds.npy  [$N3V_RAW_ROOT]")
    ap.add_argument("--out-root", default=os.environ.get("N3V_OUT_ROOT"),
                    help="dir to write <scene>/{frames,sparse}  [$N3V_OUT_ROOT]")
    ap.add_argument("--libgs-tool", default=os.environ.get("LIBGS_TOOL", "third_party/libgs/tool"),
                    help="path to libgs/tool/ (for llff2colmap.py, database.py)  [$LIBGS_TOOL]")
    ap.add_argument("--colmap", default=os.environ.get("COLMAP_BIN", "colmap"),
                    help="colmap executable  [$COLMAP_BIN]")
    ap.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable),
                    help="python used for libgs/tool/database.py  [$PYTHON_BIN]")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-colmap", action="store_true")
    a = ap.parse_args()

    if not a.raw_root or not a.out_root:
        raise SystemExit("FATAL: --raw-root and --out-root are required (or set $N3V_RAW_ROOT / $N3V_OUT_ROOT)")
    N3V, RECONGS = Path(a.raw_root).expanduser(), Path(a.out_root).expanduser()
    LIBGS_TOOL = Path(a.libgs_tool).expanduser().resolve()
    COLMAP, PYTHON = a.colmap, a.python
    _bind_libgs(LIBGS_TOOL)

    scene, dst = N3V / a.scene, RECONGS / a.scene
    if not scene.is_dir():
        raise SystemExit(f"no such scene: {scene}")
    if not (scene / "poses_bounds.npy").exists():
        raise SystemExit(f"FATAL: {scene}/poses_bounds.npy missing -- N3DV ships it next to the videos")

    t0 = time.time()
    cams = _check_counts(scene)
    print(f"=== {a.scene}: {len(cams)} cams, {a.frames} frames ===", flush=True)
    dst.mkdir(parents=True, exist_ok=True)

    if not a.skip_extract:
        print("[1/3] extract", flush=True); extract_frames(scene, cams, a.frames)
    print("[2/3] symlink", flush=True); link_frames(scene, dst, cams, a.frames)
    if not a.skip_colmap:
        print("[3/3] colmap", flush=True); run_colmap(scene, dst, cams)
    print(f"=== {a.scene} DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
