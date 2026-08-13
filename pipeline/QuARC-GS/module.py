from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from absl import logging
from jaxtyping import Bool, Float
from PIL import Image
from torch import Tensor
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.data.types import TensorSpace
from libgs.metric import psnr, ssim, lpips
from libgs.model.merged_gaussian import MergedGaussianModel
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig
from libgs.renderer.network_gui import interact_with_gui
from libgs.utils.time import Timer

from .data import DataModule
from .model.deformation import Deformation, DeformConfig
from .model.gaussian import GaussianModel

import os
from libgs.data.types import storePly
import time

@dataclass
class GaussianConfig:
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 15000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    percent_dense: float = 0.01


@dataclass
class DensifyConfig:
    from_step: int = 500
    until_step: int = 5000
    interval: int = 100
    grad_threshold: float = 2e-4
    opacity_reset_interval: int = 30000
    max_gaussians: int = 10000000


@dataclass
class MotionConfig:
    """Appearance-side knobs of the shipped method: what gets transmitted, and how it is priced."""

    # Accounting quantum used when the run is NOT quantizing in-loop (deformation.enable_quant=false,
    # i.e. the ablation arm): the continuous delta is rounded to this step for CODING ONLY, so both
    # arms land on the same bytes/frame axis. In a quant run the run's own quant_step is used instead.
    q_mag_xyz: float = 1.0e-4
    q_mag_quat: float = 1.0e-3
    # Apply the q_app_* steps IN THE TRAINING LOOP (STE) rather than only at accounting time. Off =>
    # the model renders unquantized attrs while the payload is priced on quantized ones: an OPEN loop
    # that reports bytes for one system and PSNR for another. On => it renders exactly what is
    # transmitted, matching how motion already works.
    quant_appearance: bool = False
    q_app_xyz: float = 1.0e-3       # world-unit position
    q_app_fdc: float = 1.0e-2       # SH DC color (3)
    q_app_frest: float = 1.0e-2     # SH higher-order (9)
    q_app_opacity: float = 1.0e-2   # opacity logit (1)
    q_app_scaling: float = 1.0e-2   # log-scale (3)
    q_app_rotation: float = 1.0e-2  # rotation quaternion (4)
    # Change-gated densification. Gate the bake-clone by temporal change: scale each base Gaussian's
    # accumulated densify-gradient by grad_diff = |d/dmeans2D [L1(render,gt_t) - L1(render,gt_{t-1})]|,
    # so static-but-under-reconstructed regions stop minting a fresh incremental Gaussian every frame
    # (which merge_to_base would otherwise accumulate into the base, bloating the payload).
    change_densify: bool = False
    change_views: int = 4          # train views summed for the grad_diff weight
    change_floor: float = 0.0      # keep at least this weight everywhere (0 = pure change gate)


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 3
    random_background: bool = False  # for training
    lambda_dssim: float = 0.2
    lambda_deform: float = 1.0
    noise_scale: Optional[float] = 0.01
    saving_gs_steps: List[int] = field(default_factory=lambda: [5000])
    saving_gs_every_n_frames: int = 1000
    num_saving_images: int = 5
    full_eval: bool = True
    lpips_required: bool = False
    merge_to_base: bool = True
    re_hierarchization: bool = True
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    gaussian_stage2: GaussianConfig = field(default_factory=GaussianConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    deformation: DeformConfig = field(default_factory=DeformConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)


@torch.no_grad()
def entropy_bytes(x: Tensor, steps: List[float], ident: Optional[Tensor] = None):
    """Entropy-coded size of an [M, C] payload at per-channel quant `steps`.

    Rounds each channel (its residual from `ident`, if given) to that channel's step, then charges
    every symbol -log2 p(symbol) from its per-channel histogram -- the Shannon size an ideal range
    coder would reach. A channel with step<=0 is not counted. Returns
        (total_bytes: float, per_row_bits: Tensor[M], symbols: Tensor[M, C]).
    per_row_bits lets the caller split the cost across rows (e.g. dynamic vs static anchors)."""
    M, C = x.shape
    if M == 0:
        return 0.0, x.new_zeros(0), x.new_zeros(0, C, dtype=torch.long)
    step_t = x.new_tensor(list(steps))
    resi = x if ident is None else x - ident
    sym = torch.round(resi / step_t.clamp_min(1e-12)).long()          # [M, C] integer symbols
    per_row = x.new_zeros(M)
    for j in range(C):
        if step_t[j] <= 0:
            continue
        _, inv, counts = torch.unique(sym[:, j], return_inverse=True, return_counts=True)
        bits_j = -(counts.float() / counts.sum()).clamp_min(1e-12).log2()  # per-symbol bits
        per_row += bits_j[inv]                                             # scatter to each row
    return float(per_row.sum()) / 8.0, per_row, sym


class Module(BaseModule):
    datamodule: DataModule
    trainer: "Trainer"
    gaussians: GaussianModel
    background: torch.Tensor  # TODO remove
    timer: Timer

    @property
    def current_frame(self) -> int:
        return self.datamodule.current_frame

    @property
    def num_steps(self) -> int:
        return self.trainer.num_steps

    def setup(self):
        white_background = self.datamodule.config.white_background
        bg_color_fn = torch.ones if white_background else torch.zeros
        self.register_buffer("background", bg_color_fn(3, device=self.device))

        self.gaussians = GaussianModel(self.config.sh_degree)
        self.gaussians.create_from_pcd(
            self.datamodule.scene.point_cloud, self.datamodule.cameras_extent
        )
        self.gaussians.training_setup(self.config.gaussian)
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self._arm_app_quant(self.gaussians_incr)
        self.renderer = Renderer(self.config.renderer, self.gaussians)
        self.deform = Deformation(self.config.deformation)
        self.timer = Timer()


    def prev_frame_image(self, view: TensorSpace) -> Tensor:
        """The same camera's GT at frame t-1, at the current resolution (path-edit trick)."""
        from libgs.data.dataset.base import to_tensor

        path = view.path
        width = len(path.parts[-2])
        prev = path.parents[1] / f"{self.current_frame - 1:0{width}d}" / path.name
        size = (view.image.shape[2], view.image.shape[1])  # (W, H)
        return to_tensor(Image.open(prev), size).to(self.device)

    @torch.enable_grad()
    def grad_diff_weight(self, num_views: int, floor: float = 0.0) -> Tensor:
        """Per-base-Gaussian temporal-change weight in [floor, 1), QUEEN's dynamic-densification prior.

        Screen-space gradient magnitude of L1(render, gt_t) - L1(render, gt_{t-1}). A Gaussian that
        renders equally close to both frames (static) cancels to ~0; one on content that changed
        keeps a large gradient. Measured AFTER apply_deformation, so anchors have already carried the
        motion they can -- what survives is exactly the change the deformation could NOT capture,
        i.e. where new incremental Gaussians are actually needed. Normalised gd/(gd+median) so the
        median Gaussian maps to 0.5 and truly static ones toward 0.
        """
        from concurrent.futures import ThreadPoolExecutor

        n = self.gaussians.get_xyz.shape[0]
        accum = self.gaussians.get_xyz.new_zeros(n)
        # Index the frame dataset directly in the main process. train_dataloader() would spawn 8
        # workers every call; the persistent training Fetcher pays that once, a per-frame call must
        # not. resolution in {1,2,4,8} so resolve_size never hits the worker-only warning().
        dataset = self.datamodule.get_train_dataset()
        # The frame dataset reorders cameras each frame, so pin a FIXED camera set (chosen once) and
        # look it up by name -- keeps the gate consistent AND lets the prev-frame cache hit every frame.
        name2idx = {it.path.name: j for j, it in enumerate(dataset.items)}
        if getattr(self, "_gd_cams", None) is None:
            self._gd_cams = list(name2idx.keys())[:num_views]
        idxs = [name2idx[c] for c in self._gd_cams if c in name2idx]
        # Decode the current-frame views in parallel (PIL releases the GIL) -- what the stock 8-worker
        # dataloader did for free, without spawning workers each frame (~400 ms -> ~120 ms).
        with ThreadPoolExecutor(max_workers=max(len(idxs), 1)) as ex:
            vps = [vp.to(self.device) for vp in ex.map(lambda j: dataset[j], idxs)]

        prev_cache = getattr(self, "_gd_prev", {})   # {cam: (frame, current_image)} from t-1
        new_prev = {}
        for vp in vps:
            cam = vp.path.name
            # frame t-1's current image IS frame t's prev image -- reuse it, no disk decode.
            pe = prev_cache.get(cam)
            prev = pe[1] if (pe is not None and pe[0] == self.current_frame - 1) else self.prev_frame_image(vp)
            pkg = self.renderer(vp, self.background)
            img = pkg["render"]
            loss = l1_loss(img, vp.image) - l1_loss(img, prev)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            g = pkg["viewspace_points"].grad
            if g is not None:
                accum += torch.norm(g[:, :2], dim=-1)
            new_prev[cam] = (self.current_frame, vp.image.detach())
        self._gd_prev = new_prev
        # scrub so nothing leaks into the optimizer steps that follow the bake
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        med = accum.median().clamp_min(1e-12)
        w = accum / (accum + med)                       # [0,1), static -> ~0, median -> 0.5
        if floor > 0.0:
            w = floor + (1.0 - floor) * w
        return w

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["gaussians"] = self.gaussians.capture()
        if not self.trainer.is_first_frame:
            ckpt["gaussians_incr"] = self.gaussians_incr.capture()
        ckpt["deformation"] = self.deform.capture()

    def on_load_checkpoint(self, ckpt: dict):
        self.gaussians.restore(ckpt["gaussians"], self.config.gaussian)
        if not self.trainer.is_first_frame:
            self.gaussians_incr.restore(
                ckpt["gaussians_incr"], self.config.gaussian_stage2
            )
        self.deform.restore(ckpt["deformation"])

    def forward(
        self,
        viewpoint: TensorSpace,
        training: bool = False,
        **kwargs,
    ) -> dict:
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)

        results = {}
        if not self.trainer.is_first_frame:
            num_stage1_steps = self.deform.config.num_stage1_steps
            enable_deform = False
            if self.global_step < num_stage1_steps:
                enable_deform = True
            elif self.global_step == num_stage1_steps and training:
                enable_deform = True

            if enable_deform:
                delta_xyz, delta_rot = self.deform(self.gaussians.get_xyz_ori)
                results["delta_xyz"] = delta_xyz
                results["delta_rotation"] = delta_rot
                self.gaussians.delta_xyz = delta_xyz
                self.gaussians.delta_rot = delta_rot
        elif training:
            self.gaussians.noise_scale = self.config.noise_scale

        results.update(self.renderer(viewpoint, bg_color, **kwargs))
        for attr in ["delta_xyz", "delta_rot", "noise_scale"]:
            setattr(self.gaussians, attr, None)
        return results

    def pre_training_step(self):
        interact_with_gui(
            self.global_step,
            self.renderer.config,  # can be modified
            self,
            self.datamodule.config.root,
            self.num_steps,
        )

        if self.trainer.is_first_step:
            self.timer.clock("frame").start()

        if not self.trainer.is_first_frame and self.trainer.is_first_step:
            self.timer.clock("frame setup").start()
            num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
            if self.config.merge_to_base and num_incr_gs > 0:
                logging.info(f"Merge and prune {num_incr_gs} gaussians")
                self.gaussians.extend(self.gaussians_incr)
                indices = self.gaussians.get_opacity.flatten().sort().indices
                self.gaussians.prune_points(indices[:num_incr_gs])

            # Seed the anchor sampling with the FRAME NUMBER so the [3,N] Gaussian->anchor binding is
            # a pure function of (canonical geometry, frame). A receiver holds that geometry
            # bit-exactly and knows the frame, so it DERIVES the binding instead of being sent it --
            # turning a ~1.16 MB/frame side channel into 0 bytes. `reset` needs no seed: it only
            # re-runs the knn against anchors that already exist.
            reset_grid = self.current_frame == 1
            if self.current_frame == 1:
                self.deform.setup(self.gaussians.get_xyz.detach(), reset_grid, seed=self.current_frame)
            elif self.current_frame % self.deform.config.grid_reset_interval == 0 and self.config.re_hierarchization:
                self.deform.reset_grid(self.gaussians.get_xyz.detach(), seed=self.current_frame)
            else:
                self.deform.reset(self.gaussians.get_xyz.detach())
            self.deform.to(self.device)
            self.gaussians_incr = GaussianModel(self.config.sh_degree)
            self._arm_app_quant(self.gaussians_incr)
            logging.info("Reset gaussians of renderer")
            self.renderer.gaussians = self.gaussians

            self._tx_delta = None   # invalidate last frame's bake snapshot (set again at this bake)

            self.timer.clock("frame setup").stop()

        num_stage2_steps = self.global_step - self.deform.config.num_stage1_steps
        if not self.trainer.is_first_frame and num_stage2_steps == 1:
            self.gaussians_incr.training_setup(self.config.gaussian_stage2)
            self.renderer.gaussians = MergedGaussianModel(
                [self.gaussians, self.gaussians_incr]
            )

    def training_step(
        self, viewpoint: TensorSpace, current_step: int
    ) -> Tuple[Tensor, dict, dict]:
        self.timer.clock("training step").start()

        num_stage1_steps = self.deform.config.num_stage1_steps
        if self.trainer.is_first_frame:
            self.gaussians.update_learning_rate(current_step)
        elif (current_stage2_step := current_step - num_stage1_steps) > 0:
            self.gaussians_incr.update_learning_rate(current_stage2_step)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if current_step % 1000 == 0 and self.trainer.is_first_frame:
            self.gaussians.oneupSHdegree()

        # Render
        if (current_step - 1) == self.renderer.config.debug_from:
            self.renderer.config.debug = True

        render_pkg = self.forward(viewpoint, training=True)
        image = render_pkg["render"]
        gt_image = viewpoint.image
        loss_rgb = l1_loss(image, gt_image)
        loss_dssim = 1.0 - ssim(image, gt_image)

        lambda_dssim = self.config.lambda_dssim
        loss = (1.0 - lambda_dssim) * loss_rgb + lambda_dssim * loss_dssim

        metrics = {
            "loss/rgb": loss_rgb,
            "loss/dssim": loss_dssim,
            "psnr": psnr(image, gt_image).mean().double(),
            "gs": self.renderer.gaussians.get_xyz.shape[0],
        }

        if "delta_xyz" in render_pkg and self.config.lambda_deform > 0:
            loss_reg = self.deform.reg_loss()
            loss = loss + self.config.lambda_deform * loss_reg
            metrics["loss/reg"] = loss_reg

        self.timer.clock("training step").stop()

        self.timer.clock("tensorboard logging").start()
        self.log_dict(metrics)
        self.timer.clock("tensorboard logging").stop()

        return loss, metrics, render_pkg

    def transmit_delta(self) -> Tuple[Tensor, float, float]:
        """The [A,7] delta actually transmitted this frame, plus the (step_xyz, step_quat) it is
        entropy-coded at."""
        mc, dcfg = self.config.motion, self.deform.config
        d = self.deform.delta
        if dcfg.enable_quant:
            d = self.deform.quantize_delta(d)
            sx, sq = dcfg.quant_step_xyz, dcfg.quant_step_quat
        else:
            sx, sq = mc.q_mag_xyz, mc.q_mag_quat
        return d.detach(), sx, sq

    def app_quant_steps(self) -> dict:
        """The per-attribute transmission grid for the incremental Gaussians, keyed to match
        GaussianModel._qr. Order/steps must stay in lockstep with log_storage's app_steps and with
        log_storage's app_steps, or the model renders one thing and we account for another."""
        mc = self.config.motion
        return dict(xyz=mc.q_app_xyz, fdc=mc.q_app_fdc, frest=mc.q_app_frest,
                    opacity=mc.q_app_opacity, scaling=mc.q_app_scaling, rotation=mc.q_app_rotation)

    def _arm_app_quant(self, g) -> None:
        """Turn on in-loop STE quantization for the INCREMENTAL Gaussians only.

        Never the base: the base is the one-time keyframe, transmitted once at its own (coded)
        precision, not re-sent per frame. Quantizing it here would degrade carried content that
        nobody is paying the per-frame appearance bitrate for. The incrementals become base at the
        next merge_to_base, and by then they are already rounded -- so both sides agree.
        """
        if self.config.motion.quant_appearance:
            g.quant_steps = self.app_quant_steps()

    def transmitted_delta(self) -> Tuple[Tensor, float, float]:
        """The delta this frame ACTUALLY baked, snapshotted at the bake.

        Everything that measures or ships the payload must go through here rather than calling
        transmit_delta() directly, because transmit_delta() reads the LIVE delta -- and after the
        bake the deform optimizer steps once more, moving it. The snapshot is what the canonical
        geometry was built from, so it is what a receiver needs and what we must be charged for."""
        d = getattr(self, "_tx_delta", None)
        if d is None:   # no bake ran this frame (first frame) -- fall back to the live delta
            return self.transmit_delta()
        return d, self._tx_sx, self._tx_sq

    @torch.no_grad()
    def log_storage(self):
        """Clean per-frame total: motion + appearance on ONE entropy byte axis.

        This is the ONLY storage axis the pipeline reports, and it is an ESTIMATE: the Shannon size
        (-log2 p per symbol) of the quantized payload. It excludes the CDF/symbol tables a real range
        coder must also emit, so an actual coded file runs ~8-20% larger (the gap widens as the
        per-frame payload shrinks, since that side-info is roughly fixed). Quote it as an estimate.
        The anchor->Gaussian binding is NOT charged: it is re-derived by the receiver from the base
        geometry it already holds (see DeformModel index), so it is not transmitted side-info.

        Motion = the transmitted [A,7] anchor delta, split into DYNAMIC anchors (>=1 non-identity
        quantized channel) and STATIC anchors (all channels round to identity -> the near-free
        symbol-0 tail). Appearance = the incremental Gaussians' 23 attributes entropy-coded at the
        q_app_* accounting steps. Called once at frame end, after stage-2 densification settles."""
        if self.trainer.is_first_frame:
            return
        mc = self.config.motion

        # --- motion: dynamic vs static split ---
        # The delta SNAPSHOTTED at the bake, not the live one (see the bake site): the deform
        # optimizer steps once more after the bake, so the live tensor is not what was transmitted.
        d, sx, sq = self.transmitted_delta()
        ident = d.new_zeros(1, 7); ident[:, 3] = 1.0
        motion_steps = [sx, sx, sx, sq, sq, sq, sq]
        _, bits, sym = entropy_bytes(d, motion_steps, ident=ident)
        # Dynamic = the anchor actually moves: any non-zero translation (cols 0-2) or rotation
        # (imaginary quat cols 4-6). Col 3 is the quaternion w, an UNNORMALISED scale (~const across
        # anchors) that carries no rotation, so it must NOT count toward "dynamic" -- otherwise every
        # anchor reads dynamic. Its (cheap, near-constant) bits still land in the static tail below.
        dynamic = (sym[:, [0, 1, 2, 4, 5, 6]] != 0).any(dim=1)
        n_dyn, n_sta = int(dynamic.sum()), int((~dynamic).sum())
        dyn_kb = float(bits[dynamic].sum()) / 8 / 1024
        sta_kb = float(bits[~dynamic].sum()) / 8 / 1024
        motion_kb = dyn_kb + sta_kb

        # --- appearance: incremental Gaussians, entropy-coded at the same footing ---
        g = self.gaussians_incr
        n_gs = g.get_xyz.shape[0]
        app_steps = ([mc.q_app_xyz] * 3 + [mc.q_app_fdc] * 3 + [mc.q_app_frest] * 9
                     + [mc.q_app_opacity] + [mc.q_app_scaling] * 3 + [mc.q_app_rotation] * 4)
        if n_gs > 0:
            attrs = torch.cat([
                g._xyz, g._features_dc.reshape(n_gs, -1), g._features_rest.reshape(n_gs, -1),
                g._opacity, g._scaling, g._rotation,
            ], dim=1)                                          # [N, 23]
            app_kb = entropy_bytes(attrs.detach(), app_steps)[0] / 1024
        else:
            app_kb = 0.0

        total_kb = motion_kb + app_kb

        # --- base keyframe: the ONE-TIME carried model (frame-0 Gaussians), NOT a per-frame cost.
        # Computed once (it is ~constant across frames) and cached, so it does not slow every frame.
        # Surfaced here so the per-frame total is never read as the whole story: the honest clip cost
        # is base_keyframe + num_frames * total_kb, i.e. total_amortized_kb per frame.
        if getattr(self, "_base_kb", None) is None:
            gb = self.gaussians
            nb = gb.get_xyz.shape[0]
            battrs = torch.cat([
                gb._xyz, gb._features_dc.reshape(nb, -1), gb._features_rest.reshape(nb, -1),
                gb._opacity, gb._scaling, gb._rotation,
            ], dim=1).detach()
            self._base_n = nb
            # Shannon estimate of the carried keyframe. NOTE: this UNDERSTATES a real coded file
            # by ~15% -- it charges -log2(p) per symbol and ignores the CDF + symbol tables a decoder
            # needs. This term dominates the amortized per-frame cost, so quote it as an estimate.
            self._base_kb = entropy_bytes(battrs, app_steps)[0] / 1024
        try:
            num_frames = int(self.datamodule.config.extra_dataset_kwargs["num_frames"])
        except Exception:
            num_frames = 0

        self.log("storage/motion_kb", motion_kb)
        self.log("storage/motion_dynamic_kb", dyn_kb)
        self.log("storage/motion_static_kb", sta_kb)
        self.log("storage/motion_n_dynamic", n_dyn)
        self.log("storage/motion_n_static", n_sta)
        self.log("storage/appearance_kb", app_kb)
        self.log("storage/appearance_n_gaussians", n_gs)
        self.log("storage/total_kb", total_kb)                      # per-frame incremental only
        self.log("storage/base_keyframe_n", self._base_n)
        self.log("storage/base_keyframe_kb", self._base_kb)         # entropy-coded, one-time
        if num_frames > 0:
            # honest per-frame cost once the one-time keyframe is amortized over the clip
            self.log("storage/total_amortized_kb", total_kb + self._base_kb / num_frames)



    @torch.no_grad()
    def post_training_step_init(self, render_pkg):
        if self.global_step < self.config.densify.until_step:
            # Densification, depend on grad
            self.densify_gaussians(
                self.global_step,
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def post_training_step_incr(self, render_pkg):
        if self.global_step < self.config.densify.until_step:
            self.adaptive_densify_gaussians(
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        if self.global_step > self.deform.config.num_stage1_steps:
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)
        elif render_pkg.get("delta_xyz", None) is not None:
            self.deform.optimizer.step()
        self.deform.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def post_training_step(self, render_pkg: dict):
        self.timer.clock("post training step").start()
        if self.trainer.is_first_frame:
            self.post_training_step_init(render_pkg)
        else:
            self.post_training_step_incr(render_pkg)
        self.timer.clock("post training step").stop()

        self.save_point_cloud()

        if self.trainer.is_last_step:
            self.timer.clock("frame").stop()
            if not self.trainer.is_first_frame:
                num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
                logging.info(f"Numer of increment gaussians: {num_incr_gs}")
                self.log("gs/incr_count", num_incr_gs)
                self.log("gs/base_count", self.gaussians.get_xyz.shape[0])
                self.log_storage()   # clean motion+appearance total on one entropy-coded axis
            time_stats = self.timer.display()
            logging.info(f"Time stats at {self.global_step} step:\n{time_stats}\n")
            self.timer.reset()

    def validation_step(
        self, viewpoint: TensorSpace, idx: int, loader_idx: int
    ) -> dict:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        torch.cuda.synchronize()
        render_dt = time.perf_counter() - t0        # accurate render time (GPU synced)
        gt_image = viewpoint.image
        if idx <= self.config.num_saving_images:
            root = self.datamodule.config.root
            path = self.output_dir / f"image" / viewpoint.path.relative_to(root)
            save_dir, name, ext = path.parent, path.stem, path.suffix
            save_dir.mkdir(parents=True, exist_ok=True)
            save_image(image, save_dir / f"{name}_step{self.global_step}{ext}")

            testing_steps = self.trainer.config.testing_steps + [self.num_steps]
            if self.global_step == testing_steps[0]:
                save_image(gt_image, save_dir / f"{name}_gt{ext}")

        # Always stream the full quality/speed set: PSNR, SSIM, LPIPS, and render FPS.
        metrics = dict(
            l1=l1_loss(image, gt_image),
            psnr=psnr(image, gt_image),
            ssim=ssim(image, gt_image),
            lpips=lpips(image, gt_image, net_type="vgg").mean(),
            fps=1.0 / max(render_dt, 1e-8),
        )
        return metrics

    def validation_end(self, results: Union[dict, List[dict]], num_loaders: int = 1):
        def fn(metrics_list, idx):
            mean = lambda xs: sum(xs) / len(xs)
            metrics = {k: mean([ms[k] for ms in metrics_list]) for k in metrics_list[0]}
            name = self.datamodule.eval_names[idx]
            self.log_dict({f"eval-{name}/{k}": v for k, v in metrics.items()})
            logging.info(
                f"Evaluate {name} dataset at step {self.global_step}:\n\t"
                + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
            )

        results = [results] if num_loaders == 1 else results
        for idx, result in enumerate(results):
            fn(result, idx)

        self.log_histogram("scene/opacity_histogram", self.gaussians.get_opacity)
        self.log("total_points", self.gaussians.get_xyz.shape[0])

    def save_point_cloud(self):
        skiping = self.current_frame % self.config.saving_gs_every_n_frames > 0
        if not self.trainer.is_last_frame and skiping:
            return

        skiping = self.global_step not in self.config.saving_gs_steps
        if not self.trainer.is_last_step and skiping:
            return

        filename = f"gaussians-{self.current_frame}-{self.global_step}.ply"
        save_path = self.output_dir / "point_cloud" / filename
        logging.info(f"\nSave gaussians to {save_path}")
        if self.trainer.is_first_frame:
            self.gaussians.save_ply(save_path)
        else:
            self.gaussians_incr.save_ply(save_path)

        if not self.trainer.is_first_frame and self.trainer.is_last_step:
            filename = f"initial-gaussians-{self.current_frame}.ply"
            save_path = self.output_dir / "point_cloud" / filename
            logging.info(f"Save deformed initial gaussians to {save_path}")

    def densify_gaussians(
        self,
        current_step: int,
        visibility_filter: Float[Tensor, "n"],
        viewspace_point_tensor: Float[Tensor, "n 3"],
        radii: Float[Tensor, "n"],
    ):
        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(
            self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.gaussians.add_densification_stats(
            viewspace_point_tensor, visibility_filter
        )

        opacity_reset_interval = self.config.densify.opacity_reset_interval
        max_gaussians = self.config.densify.max_gaussians

        densify_started = current_step > self.config.densify.from_step
        at_interval = current_step % self.config.densify.interval == 0
        within_limit = self.gaussians.get_xyz.shape[0] < max_gaussians
        if densify_started and at_interval and within_limit:
            exceed_reset_inverval = current_step > opacity_reset_interval
            size_threshold = 20 if exceed_reset_inverval else None
            self.gaussians.densify_and_prune(
                self.config.densify.grad_threshold,
                0.005,
                self.datamodule.cameras_extent,
                size_threshold,
            )

        white_background = self.datamodule.config.white_background
        at_reset_interval = current_step % opacity_reset_interval == 0
        at_densify_start = current_step == self.config.densify.from_step
        if at_reset_interval or (white_background and at_densify_start):
            self.gaussians.reset_opacity()

    def adaptive_densify_gaussians(
        self,
        visibility_filter: Bool[Tensor, "n"],
        viewspace_point_tensor: Float[Tensor, "n 3"],
        radii: Float[Tensor, "n"],
    ):
        num_init_gs = self.gaussians.get_xyz.shape[0]
        num_stage1_steps = self.deform.config.num_stage1_steps
        current_stage2_step = self.global_step - num_stage1_steps

        # Keep track of max radii in image-space for pruning
        visibility_filter_init = visibility_filter[:num_init_gs]
        radii_init = radii[:num_init_gs]
        self.gaussians.max_radii2D[visibility_filter_init] = torch.max(
            self.gaussians.max_radii2D[visibility_filter_init],
            radii_init[visibility_filter_init],
        )
        self.gaussians.add_densification_stats(
            viewspace_point_tensor, visibility_filter_init, align="left"
        )

        if current_stage2_step > 0:
            visibility_filter_incr = visibility_filter[num_init_gs:]
            radii_incr = radii[num_init_gs:]
            self.gaussians_incr.max_radii2D[visibility_filter_incr] = torch.max(
                self.gaussians_incr.max_radii2D[visibility_filter_incr],
                radii_incr[visibility_filter_incr],
            )
            self.gaussians_incr.add_densification_stats(
                viewspace_point_tensor, visibility_filter_incr, align="right"
            )

        if self.global_step == num_stage1_steps:
            logging.info("Apply deformation to previous gaussians")
            mc = self.config.motion
            # TRANSMIT WHAT WE BAKE. Snapshot the delta this bake is about to consume. log_storage
            # runs at frame end, by which point post_training_step has called deform.optimizer.step()
            # once more (it fires at global_step == num_stage1_steps, since the stage-2 branch needs
            # global_step > num_stage1_steps). Calling transmit_delta() there returns a delta that has
            # moved on from the baked one -- measured: ~6% of anchors off by one quant step, which
            # cost -0.031 dB on decode and meant "reconstructs bit-exactly" and "costs N bytes" were
            # claims about two different tensors. The post-bake update never touches geometry (it only
            # warm-starts the next frame), so the baked delta is the one a receiver needs.
            if not self.trainer.is_first_frame:
                d_tx, sx_tx, sq_tx = self.transmit_delta()
                self._tx_delta, self._tx_sx, self._tx_sq = d_tx.clone(), sx_tx, sq_tx
            delta_xyz, delta_rotation = self.deform(self.gaussians.get_xyz.detach())
            self.gaussians.apply_deformation(delta_xyz, delta_rotation)
            if mc.change_densify and not self.trainer.is_first_frame:
                # Down-weight the accumulated densify-gradient by temporal change, so the clone only
                # mints incremental Gaussians where the frame actually changed (not on static-but-
                # hard-to-fit texture, which the carried base already covers).
                w = self.grad_diff_weight(mc.change_views, mc.change_floor)
                self.gaussians.xyz_gradient_accum.mul_(w.unsqueeze(-1))
                self.log("gs/change_weight_mean", w.mean().item())
            self.gaussians.adaptive_densify_and_clone(
                self.deform.config.densify_grad_threshold,
                self.datamodule.cameras_extent,
            )
            self.gaussians_incr.restore(
                self.gaussians.capture(), self.config.gaussian_stage2
            )
            mask = torch.zeros_like(self.gaussians.get_xyz[:, 0], dtype=torch.bool)
            mask[num_init_gs:] = True
            self.gaussians.prune_points(mask)
            self.gaussians_incr.prune_points(~mask)

        densify_interval = self.deform.config.densify_interval
        if current_stage2_step > 0 and current_stage2_step % densify_interval == 0:
            self.gaussians_incr.adaptive_densify_and_prune(
                self.deform.config.densify_grad_threshold,
                self.deform.config.opacity_threshold,
                self.datamodule.cameras_extent,
                20,
            )
