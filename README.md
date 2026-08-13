<div align="center">

# QuARC-GS — Quantized-Motion, Change-Gated Streaming of Dynamic Gaussians

Storage-aware online free-viewpoint video with **STE-quantized anchor motion**, **change-gated
appearance densification**, and **entropy-based storage accounting**.

</div>

## 0. What you need

Raw videos in, reported metrics out. Nothing is downloaded but the public datasets — this repository
builds the COLMAP models and trains the canonical first frame itself.

| | |
|---|---|
| datasets | N3DV (6 scenes) · Meet Room (3 scenes) |
| disk | ~16 GB of extracted PNGs per N3DV scene |
| GPU | one CUDA GPU; reference numbers on an RTX PRO 6000 |
| time per scene | ~6 min frames + ~15 s COLMAP + ~25 min frame 0 + ~23 min for 300 frames |

---

## 1. Setup

### 1a. Clone

The three third-party dependencies are pinned as **git submodules**, so cloning recursively gets
byte-for-byte what the reported results were produced with:

```bash
git clone --recurse-submodules <this-repo-url> QuARC-GS
cd QuARC-GS

# already cloned without --recurse-submodules?
git submodule update --init --recursive
```

```
QuARC-GS/
└── third_party/
    ├── libgs/                            Awesome3DGS/libgs            @ 7702313
    ├── diff-gaussian-rasterization/      graphdeco-inria              @ 59f5f77
    │   └── third_party/glm/              g-truc/glm                   @ 5c46b9c   (nested)
    └── simple-knn/                       gitlab.inria.fr/bkerbl       @ 86710c2
```

### 1b. Conda environment

```bash
# 1. env + build toolchain. gcc and nvcc come from conda, NOT the system, so the two CUDA
#    extensions compile against the same toolchain torch's cu128 wheels were built with.
conda create -n quarcgs -c conda-forge -c nvidia -y \
    python=3.10 gcc_linux-64=13.4 gxx_linux-64=13.4 cuda-toolkit=12.8
conda activate quarcgs
nvcc --version | tail -2          # -> release 12.8, V12.8.93 -- must be the conda one, not /usr/local/cuda

# 2. torch FIRST -- everything below compiles against it and must be able to import it
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. pure-python deps
pip install -r requirements.txt --no-build-isolation
pip install ninja                 # --no-build-isolation stops pip fetching it; without ninja the
                                  # two extension builds fall back to a slow serial compile

# 4. the two CUDA extensions, from the pinned submodules. Both exports below are load-bearing --
#    see "Three things that will bite you" underneath.
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
echo "$TORCH_CUDA_ARCH_LIST"        # -> 12.0 on the RTX PRO 6000. Must NOT be conda's 13-arch default.
export NVCC_PREPEND_FLAGS='-include cstdint'
pip install -e third_party/diff-gaussian-rasterization --no-build-isolation
pip install    third_party/simple-knn --no-build-isolation      # NOT -e

# 5. libgs, with --no-deps: step 3 already installed everything it needs, and without the flag pip
#    re-fetches the rasterizer from its git URL and silently replaces your pinned build
pip install -e third_party/libgs --no-deps
```

#### Three things that will bite you

**`TORCH_CUDA_ARCH_LIST` — conda picks a value torch rejects.** `conda activate quarcgs` runs
`$CONDA_PREFIX/etc/conda/activate.d/~cuda-nvcc_activate.sh`, shipped with `cuda-toolkit`, which
exports `TORCH_CUDA_ARCH_LIST="5.0;...;10.0;10.1;12.0+PTX"` whenever the variable is unset. `10.1`
is not in torch 2.11's supported-arch table, so *both* extension builds abort before compiling a
single file:

```
ValueError: Unknown CUDA arch (10.1) or GPU not supported
```

Setting the variable yourself is what step 4 does — it wins because the conda script only assigns
when unset, and it also cuts build time from thirteen architectures to one. If you build in a fresh
shell, export it again; it is not persisted.

**`NVCC_PREPEND_FLAGS='-include cstdint'`.** The pinned rasterizer's
`cuda_rasterizer/rasterizer_impl.h` uses `uint32_t`, `uint64_t` and `std::uintptr_t` without
including `<cstdint>`. Older libstdc++ pulled it in transitively; gcc 13 does not, so the build dies
with `error: identifier "uint32_t" is undefined`. Prepending the include fixes it without patching
the submodule, which keeps the pin byte-for-byte.

**simple-knn must be installed *non*-editable.** `third_party/simple-knn/simple_knn/` holds no
`__init__.py` — just a `.gitkeep`, with the compiled `_C` dropped in beside it. A regular install
copies `_C` into `site-packages/simple_knn/`, which Python resolves as a namespace package. `pip
install -e` instead writes an editable *finder* whose lookup requires an `__init__`, so it never
matches and the import fails with `ModuleNotFoundError: No module named 'simple_knn'` — with the
mapping and the `.so` both present and correct. `diff-gaussian-rasterization` does ship an
`__init__.py`, so `-e` is right there.

Verify before touching data:

```bash
python -c "import torch, diff_gaussian_rasterization, simple_knn._C, torch_scatter, pykeops, jax; \
           print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
PYTHONPATH=$PWD:$PWD/third_party/libgs python -c "import libgs.pipeline as p; print(p.__file__)"
```

Import `simple_knn._C`, not `simple_knn` — the latter is a namespace package and imports fine even
when the CUDA extension is missing entirely.

The second line must print a path under `third_party/libgs`. Note it imports `libgs.pipeline`, not
`libgs` — **`libgs` is a namespace package** (no `__init__.py`), so `libgs.__file__` is always `None`
and `libgs.__path__` legitimately lists several directories. The practical consequence: if an older
non-editable `libgs` is sitting in `site-packages`, individual submodules can silently resolve there
instead of here. Step 4 above replaces it with an editable install pointing at the submodule, so both
entries refer to the same tree. If you inherited a stale one, `pip uninstall libgs` first.

(`jax` is in the first line deliberately — it is *not* optional. libgs's camera types do
`from jaxtyping import Array`, and jaxtyping resolves `Array` by importing jax on first access, so a
missing jax surfaces as a confusing failure deep inside dataset loading.)

### 1c. COLMAP

```bash
# a separate env keeps COLMAP's CUDA runtime away from torch's.
# Pin the version -- unpinned resolves to a build that cannot start, see below.
conda create -n colmapenv -c conda-forge colmap=3.13.0 -y
```

**Why the pin.** Unpinned, conda-forge now resolves to 4.1.1, whose `colmap` binary links
`libfaiss.so` and `libOpenImageIO.so.3.1` but declares neither as a run dependency. conda is never
told to install them, so every invocation dies in the dynamic loader:

```
error while loading shared libraries: libfaiss.so: cannot open shared object file
```

`conda activate colmapenv` does not help — colmap's RPATH is `$ORIGIN/../lib`, so it looks in the
env's own `lib/`, where the files genuinely are not. 3.13.0 needs neither library and is the version
the reported numbers were produced with. To repair an env you already created unpinned:
`conda install -n colmapenv -c conda-forge colmap=3.13.0 -y`.

Then pass that binary explicitly to the preparation scripts:

```bash
export COLMAP_BIN=$(conda run -n colmapenv printenv CONDA_PREFIX)/bin/colmap
$COLMAP_BIN --help | head -1                        # -> COLMAP 3.13.0 ...
```

Resolve it by path, **not** with `conda run -n colmapenv which colmap`. `conda run` inherits your
`PATH`, so any other colmap earlier on it — `~/.local/bin/colmap` is the usual culprit — wins, and
you silently prepare your data with a different COLMAP than the one you just installed. If the
version line above is not 3.13.0, that is what happened.

---


---

## 2. Reproduce

Raw videos in, metrics out, with nothing but this repository and the public datasets. Build the
COLMAP models (§2a N3DV, §2b Meet Room), train frame 0 (§2c), then stream all 300 frames (§2d).

### 2a. N3DV: raw videos → trainable root

Download the N3DV release. Each scene
is per-camera `.mp4` files plus an LLFF `poses_bounds.npy`:

#### Environment variables

Every command below assumes these. Put them in your shell profile or re-export per session:

```bash
cd /path/to/QuARC-GS                               # repo root
export PYTHONPATH=$PWD:$PWD/third_party/libgs      # libgs comes from the pinned submodule
export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 # REQUIRED: the frame-0 checkpoint stores numpy scalars,
                                          # which torch 2.11's default weights_only=True refuses to load

# where your data lives -- used throughout sections 2-5
export RAW=/path/to/raw               # downloaded n3dv datasets
export DATA=/path/to/preprocessed     # trainable roots (this is what data.root points at)
export OUT=/path/to/output            # runs, checkpoints, TensorBoard logs
```

```
<RAW>/<scene>/camXX.mp4
<RAW>/<scene>/poses_bounds.npy
```

```bash
conda activate quarcgs
for s in coffee_martini cook_spinach flame_salmon_1 flame_steak sear_steak; do
  python tools/preprocess_n3v.py $s \
      --raw-root   $RAW/ \
      --out-root   $DATA/n3dv \
      --libgs-tool third_party/libgs/tool \
      --colmap     $COLMAP_BIN \
      --frames 300
done
```

Frame extraction dominates: measured **0.78 s/frame/camera** (2704x2028 PNG encode), so a
300-frame 20-camera scene is ~6 min at the default 12 workers and lands **~16 GB** of PNGs. The
COLMAP stage is ~15 s. Add `--skip-extract` to re-run only COLMAP.

This extracts every frame, builds the transposed symlink tree the loader wants, and triangulates a
COLMAP model:

```
<DATA>/<scene>/
├── frames/<NNNN>/camXX.png                  # per-frame images (symlinks into <RAW>/<scene>/images/)
└── sparse/0/{cameras,images,points3D}.bin   # COLMAP model
```

Check the result before training on it:

```bash
conda activate quarcgs
python tools/validate_n3v.py --root $DATA/n3dv
```

It catches the two failures that are silent at train time: dangling symlinks, and a near-empty
`points3D` from a triangulation that "succeeded" but found nothing.

---


### 2b. Meet Room: raw videos → trainable root

#### Environment variables

Every command below assumes these. Put them in your shell profile or re-export per session:

```bash
cd /path/to/QuARC-GS                               # repo root
export PYTHONPATH=$PWD:$PWD/third_party/libgs      # libgs comes from the pinned submodule
export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 # REQUIRED: the frame-0 checkpoint stores numpy scalars,
                                          # which torch 2.11's default weights_only=True refuses to load

# where your data lives -- used throughout sections 2-5
export RAW=/path/to/raw               # downloaded Meetroom datasets
export DATA=/path/to/preprocessed     # trainable roots (this is what data.root points at)
export OUT=/path/to/output            # runs, checkpoints, TensorBoard logs
```

Raw layout — note the `cam_N` naming, which the first stage normalises to `camNN`:

```
<RAW>/<scene>/cam_0.mp4 … cam_12.mp4
```


```bash
# stage 1: normalise names, extract frames, SfM on frame 0, image_undistorter
python tools/prepare_meetroom.py all \
    --raw-root  $RAW \
    --work-root $DATA/meetroom_work \
    --colmap    $COLMAP_BIN \
    --frames 300

# stage 2: undistort all 300x13 frames, assemble the training root
python tools/undistort_meetroom.py all \
    --work-root $DATA/meetroom_work \
    --out-root  $DATA/meetroom \
    --frames 300
```

`all` does the three scenes in sequence; pass a single scene name to do one.

Output Structures
```
<DATA>/<scene>_undist/
├── frames/<NNNN>/camNN.png                  # real PNGs (~3,900/scene), not symlinks
└── sparse/0/{cameras,images,points3D}.bin   # PINHOLE model from image_undistorter
```

Pass `<DATA>/<scene>_undist` as `data.root` — the `_undist` suffix is part of the name the shipped
config documents.


### 2c. Train the canonical frame 0 

```bash
# N3DV: 15000 init steps, ~4 min/scene   |   Meet Room: 10000 steps, ~3 min/scene
# Change <scene> to one of the following {coffee_martini cook_spinach flame_salmon_1 flame_steak sear_steak}
s=<scene>
python main.py --pipeline=QuARC-GS --config=config/quarc_gs_n3dv.yaml \
    data.root=$DATA/n3dv/$s output_dir=$OUT/frame0 experiment_name=$s \
    data.extra_dataset_kwargs.num_frames=1
```

```bash
# N3DV: 15000 init steps, ~4 min/scene   |   Meet Room: 10000 steps, ~3 min/scene
# Change <scene> to one of the following {coffee_martini cook_spinach flame_salmon_1 flame_steak sear_steak}
s=<scene>
python main.py --pipeline=QuARC-GS --config=config/quarc_gs_meetroom.yaml \
    data.root=$DATA/meetroom/$s output_dir=$OUT/frame0 experiment_name=$s \
    data.extra_dataset_kwargs.num_frames=1
```


#### Redraw until the frame-0 PSNR reaches its target

Training frame 0 is **nondeterministic**, and the draw sets the ceiling for the entire clip. Do not
accept the first one. Re-run the scene until its held-out PSNR is within **±0.3 dB** of the target:

| N3DV scene | target | | Meet Room scene | target |
|---|---:|---|---|---:|
| coffee_martini | 28.40 | | discussion | 32.17 |
| cook_spinach | 33.75 | | trimming | 31.65 |
| cut_roasted_beef | 33.70 | | vrheadset | 30.91 |
| flame_salmon_1 | 28.80 | | | |
| flame_steak | 33.66 | | | |
| sear_steak | 33.48 | | | |

**Select on the held-out PSNR — the number the run prints — never on train PSNR.** 

```bash
grep -A1 "Evaluate test dataset" $OUT/frame0/<scene>/<ts>/logging.INFO | tail -1
```

Automate it — redraw until the target is met, keep the winner:

```bash
TARGET=33.70; TOL=0.3; s=cut_roasted_beef        # from the table above
for draw in $(seq 1 10); do
  python main.py --pipeline=QuARC-GS --config=config/quarc_gs_n3dv.yaml \
      data.root=$DATA/n3dv/$s output_dir=$OUT/frame0 experiment_name=${s}_d${draw} \
      data.extra_dataset_kwargs.num_frames=1
  P=$(grep -ho "PSNR: [0-9.]*" $OUT/frame0/${s}_d${draw}/*/logging.INFO | tail -1 | cut -d' ' -f2)
  echo "draw $draw: $P  (target $TARGET)"
  awk -v p="$P" -v t="$TARGET" -v tol="$TOL" 'BEGIN{exit !(p >= t - tol)}' && \
    { echo "  accepted"; break; }
done
```

### 2d. Run the full 300-frame reconstruction

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
for s in coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak; do
  KF=$(ls $OUT/frame0/$s/*/ckpt-0-15000.pth | tail -1)
  python main.py --pipeline=QuARC-GS --config=config/quarc_gs_n3dv.yaml \
      data.root=$DATA/n3dv/$s output_dir=$OUT experiment_name=$s ckpt_path=$KF
done

KF=<path_to_initial_frame_checkpoint>
for s in discussion trimming vrheadset; do
  KF=$(ls $OUT/frame0/$s/*/ckpt-0-10000.pth | tail -1)
  python main.py --pipeline=QuARC-GS --config=config/quarc_gs_meetroom.yaml \
      data.root=$DATA/meetroom/${s}_undist output_dir=$OUT experiment_name=$s ckpt_path=$KF
done
```

Confirm `Training frame 0: 0it` in the log — that is the keyframe being **restored**. If you see
15000 (or 10000) steps there instead, `num_init_steps` and the checkpoint disagree.

Then go to [§3](#3-configuration) for the knobs and
[§4](#4-report-metrics-psnr--ssim--lpips--train-timeframe--storageframe--fps) to read the numbers
out. The per-scene frame-0 targets to compare against are in the table in §2c above; read its note
on frame-0 variance first.

---

## 3. Configuration

`config/quarc_gs_n3dv.yaml` **is** the full method on N3DV; `config/quarc_gs_meetroom.yaml` is the
Meet Room counterpart. Any config field can be overridden on the CLI as `<dotted.key>=value` —
**no leading dashes** (`--` is reserved for the two absl flags `--pipeline` / `--config`, and
`--data.root=...` is rejected as an unknown flag). The four you will always set:

| override | meaning |
|---|---|
| `data.root=` | the preprocessed `<scene>` (N3DV) or `<scene>_undist` (Meet Room) directory |
| `output_dir=` | where checkpoints, images and TensorBoard logs are written |
| `experiment_name=` | subdirectory name for this run |
| `ckpt_path=` | the **frame-0 keyframe** checkpoint (see below) |

**The frame-0 checkpoint.** Every later frame is a residual on a canonical reconstruction of
frame 0, so `ckpt_path` and `num_init_steps` have to agree — if they don't, the trainer silently
retrains frame 0 instead of restoring it:

| what you want | `ckpt_path=` | `num_init_steps` |
|---|---|---|
| train frame 0, stop (§2c) | `null` | the training budget: 15000 N3DV / 10000 Meet Room |
| stream from it (§2d) | `<OUT>/frame0/<scene>/<ts>/ckpt-0-<N>.pth` | must equal the `<N>` in that filename |
| do both in one run | `null` | trains for this many steps, then streams straight on |

`Training frame 0: 0it` in the log confirms a restore. Anything else means the two disagree.

Splitting §2c from §2d is worth it: frame-0 training is the expensive, high-variance part, and
keeping it separate lets you look at the draw before committing to 300 frames.

### Running several scenes at once

Give each concurrent process its own `CUDA_VISIBLE_DEVICES` **and its own `gui_port=`** — the viewer
server binds unconditionally, so two runs on the default port die with
`Errno 98: Address already in use`. Concurrent runs also contend for the GPU, so **do not quote
train-time or FPS from them**; use a dedicated single run for timing.

Key defaults (in `config/quarc_gs_n3dv.yaml`): `data.resolution: 2` (the N3DV protocol resolution —
**keep this for comparability**), `num_frames: 300`, stage-1 = 100 steps + stage-2 = 100 steps
(`num_incr_steps: 200`), `max_gs_per_grid: 4` (the shipping value; 24 was the old default and scores
~0.22 dB lower), motion `quant_step_xyz: 0.0017`, `change_densify: true`.

---


---

## 4. Report metrics (PSNR · SSIM · LPIPS · train-time/frame · storage/frame · FPS)

All six come out of a normal run. Nothing is re-run and nothing is re-derived — `collect_metrics.py`
reads the TensorBoard event file and the INFO log.

```bash
# one run
python tools/collect_metrics.py $OUT/cut_roasted_beef/<timestamp> --out metrics.json

# every scene at once -- the glob must reach the <experiment>/<timestamp>/ level
python tools/collect_metrics.py --glob "$OUT/*/*/" --out metrics.json
```

It prints a table, and writes `metrics.json` (per-frame series + per-run summary) plus
`metrics.csv` (one row per run) next to it:

```
scene                     n    PSNR    SSIM   LPIPS    FPS  s/frame  KB/f est  clip MB   slope
----------------------------------------------------------------------------------------------
cut_roasted_beef        299  33.699  0.9607  0.1277  250.8     4.60     28.80    14.03  -0.166
```

- `n` — inter-frames scored (299 for a 300-frame clip; frame 0 is the keyframe and is not scored)
- `KB/f est` — `storage/total_kb`, the entropy estimate, averaged over inter-frames
- `clip MB` — `keyframe + 299 × per-frame`, the honest whole-clip cost
- `slope` — dB/100 frames, a drift check. It includes the scene's own drift, so compare slopes
  across scenes rather than reading one in isolation.

The per-frame series in `metrics.json` (`series.psnr`, `series.total_kb`, …) are ordered lists, one
entry per streamed frame — use them for PSNR-over-frame or storage-over-frame plots.

Where each metric comes from:

| metric | source | log key |
|---|---|---|
| **PSNR / SSIM / LPIPS** | `validation_step`, on the held-out camera each frame | TensorBoard `eval-test/{psnr,ssim,lpips}` |
| **FPS** | render timed with `cuda.synchronize()` in `validation_step` | TensorBoard `eval-test/fps` |
| **Train time / frame** | the `frame:` line from the per-frame `Timer` (train only, excludes eval) | stdout / INFO log → parsed by `collect_metrics` |
| **Storage / frame** | `Module.log_storage`, Shannon entropy estimate of the quantized payload | TensorBoard `storage/total_kb` |

`collect_metrics.py` reports `psnr, ssim, lpips, fps, train_s_per_frame, total_kb` (means over the
streamed frames; frame 0 excluded from storage).

The run-to-run noise floor, with the frame-0 checkpoint held fixed, is **±0.027 dB** (N3DV) and
**±0.024 dB** (Meet Room). A gap wider than that is a real difference, not run variance — and the
first thing to check is the frame-0 draw (§2c), which sets the ceiling for the whole clip.

---

## Repository layout

```
main.py                       # entry point:  --pipeline=QuARC-GS --config=<cfg>
config/                       # quarc_gs_n3dv.yaml, quarc_gs_meetroom.yaml (the reported configs)
pipeline/QuARC-GS/
├── module.py                 # training/eval, storage accounting, change-gated densification
├── model/deformation.py      # anchor hierarchy + STE-quantized delta
├── model/gaussian.py         # Gaussian model
└── trainer.py                # frame loop
tools/
├── preprocess_n3v.py         # N3DV: videos + LLFF poses -> frames/ + COLMAP model    (§2a)
├── validate_n3v.py           # catch silent preprocessing failures                    (§2a)
├── prepare_meetroom.py       # Meet Room: videos -> frames + OPENCV SfM               (§2b)
├── undistort_meetroom.py     # Meet Room: undistort, assemble training root           (§2b)
└── collect_metrics.py        # aggregate the metric set from a run into JSON           (§4)
third_party/                  # pinned git submodules -- clone with --recurse-submodules  (§1a)
├── libgs/                    # the framework this pipeline plugs into (PYTHONPATH)
├── diff-gaussian-rasterization/   # CUDA rasterizer (+ nested glm submodule)
└── simple-knn/               # CUDA kNN used at initialisation
```

---

## Acknowledgement & Citation

Built on [ReCon-GS](https://github.com/jyfu-vcl/ReCon-GS) (NeurIPS 2025), and on
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and
[libgs](https://github.com/Awesome3DGS/libgs).

```bibtex
@inproceedings{fu2025recongs,
  title     = {ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction of Dynamic Scenes},
  author    = {Fu, Jiaye and Gao, Qiankun and Wen, Chengxiang and Wu, Yanmin and Ma, Siwei and Zhang, Jiaqi and Zhang, Jian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```
