import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from absl import logging
from jaxtyping import Float
from torch import Tensor, nn
import time

from ..utils import average_quaternions, batched_knn, batched_keops_knn, uniform_grid_sampling_optimized


class _STERound(torch.autograd.Function):
    """Round in the forward pass, pass the gradient through unchanged (QUEEN's StraightThrough,
    scene/decoders.py:37). Lets the delta train quantization-aware: the render sees the rounded
    value, but delta still gets a gradient so surviving anchors adapt to their own quantization."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def ste_round(x: Tensor) -> Tensor:
    return _STERound.apply(x)


class Grid(nn.Module):
    def __init__(
        self,
        size: Tuple[int, int, int],
        xyz_min: Float[Tensor, "3"],
        xyz_max: Float[Tensor, "3"],
    ):
        super().__init__()

        size = size if torch.is_tensor(size) else torch.tensor(size)
        base = torch.tensor([size[1] * size[2], size[2], 1])

        self.register_buffer("size", size)
        self.register_buffer("base", base)
        self.register_buffer("xyz_min", xyz_min)
        self.register_buffer("xyz_max", xyz_max)

        logging.info(f"Set grid min: {xyz_min.tolist()}, max: {xyz_max.tolist()}")

    def normalize(self, xyz: Float[Tensor, "n 3"], clamp: bool = True):
        xyz = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)
        if clamp:
            return xyz.clamp(0, 1)
        return xyz

    def hash(self, xyz: Float[Tensor, "n 3"]) -> torch.Tensor:
        xyz_normed = self.normalize(xyz)
        index = ((xyz_normed * self.size - 0.5).clamp(0).int() * self.base).sum(dim=-1)
        return index
        
class Adaptive_Grid(nn.Module):
    def __init__(
        self,
        vertex_num: int,
        xyz: Float[Tensor, "n 3"],
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.vertex_num = vertex_num
        t0 = time.time()
        self.vertex_idx = uniform_grid_sampling_optimized(xyz, self.vertex_num, seed=seed)
        t1 = time.time()
        self.vertex = xyz[self.vertex_idx]
        t2 = time.time()
        self.p2v = batched_keops_knn(xyz, self.vertex, 1)
        t3 = time.time()
        logging.info(f"Setup Adaptive_grids: {self.vertex_num}, total_time for fps: {t1-t0}, total time for knn: {t3-t2}")

    def reset_p2v(self, xyz: Float[Tensor, "n 3"]):
        t2 = time.time()
        self.p2v = batched_keops_knn(xyz, self.vertex, 1)
        t3 = time.time()
        logging.info(f"Reset p2v, Adaptive_grids: {self.vertex_num}, total time for knn: {t3-t2}")


@dataclass
class DeformConfig:
    quantile: float = 0.05
    lr: float = 0.0005
    num_stage1_steps: int = 100
    max_gs_per_grid: int = 5
    num_grid_levels: int = 3
    grid_level_ratio: int = 2
    momentum: Optional[float] = 0.6
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    grid_reset_interval: int = 4

    # --- STE quantization of the anchor delta (QUEEN port; round-don't-drop) ---
    # When enabled, the per-anchor delta residual is rounded to a fixed step BEFORE it is summed
    # and baked. This is applied in every forward, so the bake sees exactly the transmitted
    # (quantized) delta -- a closed reconstruction loop, transmit-faithful for free. Contrast with
    # magnitude drop, which zeroes small motion; quantization preserves it (rounds to +/-1 step)
    # and only sends genuinely sub-step motion to exact zero.
    enable_quant: bool = False
    quant_step_xyz: float = 0.0   # world units; 0 disables the xyz channel
    quant_step_quat: float = 0.0  # quaternion residual units; 0 disables the rotation channel



class Deformation(nn.Module):
    config: DeformConfig
    grids: List[Grid]
    optimizer: Optional[torch.optim.Optimizer] = None

    def __init__(self, config: DeformConfig):
        super().__init__()
        self.config = config
        self.rotation_activation = torch.nn.functional.normalize
        self.grids = []
        self.grids_num = 0
        
    @property
    def get_grid_num(self) -> int:
        return self.grids_num

    @torch.no_grad()
    def create_grids(self, xyz: Float[Tensor, "n 3"], seed: Optional[int] = None):
        """Build the anchor hierarchy. `seed` (the frame number) makes the anchor set a pure
        function of the geometry, so a receiver holding the same canonical geometry derives the
        identical [3,N] binding instead of being sent it. Each level gets its own derived seed --
        the levels sample different vertex counts from the same points, and reusing one seed would
        correlate their randperm draws for no reason."""
        q, max_gs_per_grid = self.config.quantile, self.config.max_gs_per_grid
        n = math.ceil((xyz.shape[0] / max_gs_per_grid))
        grids, level = [], self.config.num_grid_levels
        num_grids = 0
        while n > 0 and level > 0:
            num_grids += (n)
            lvl_seed = None if seed is None else int(seed) * 1000 + (self.config.num_grid_levels - level)
            grids.append(Adaptive_Grid(n, xyz, seed=lvl_seed).to(xyz.device))
            n, level = n // (self.config.grid_level_ratio), level - 1
        return grids, num_grids
    
    def reset_grid(self, xyz: Float[Tensor, "n 3"], seed: Optional[int] = None):
        old_grids = self.grids
        old_delta = self.delta
        self.grids, num_grids = self.create_grids(xyz, seed=seed)
        new_delta = torch.zeros(num_grids, 7, device=xyz.device)
        new_delta[:, 3] = 1
        offset, old_offset = 0, 0

        for level_idx in range(self.config.num_grid_levels):
            og = old_grids[level_idx]
            ng = self.grids[level_idx]
            # The old code indexed old_delta with the NEW grid's running offset. That is only
            # correct while old and new level sizes coincide (N_base is held constant by the
            # merge+prune in Module.pre_training_step). Keep an explicit old_offset so a future
            # change to merge_to_base / max_gs_per_grid fails loudly instead of silently
            # inheriting from the wrong level.
            assert og.vertex_num == ng.vertex_num, (
                f"level {level_idx}: anchor count changed {og.vertex_num} -> {ng.vertex_num}; "
                "deformation inheritance assumes a constant base Gaussian count"
            )
            dist, ng2og = batched_knn(ng.vertex, og.vertex,k=3)
            ng_corr_old_delta = old_delta[ng2og + old_offset]
            ng_delta3 = ng_corr_old_delta[:,:,:3].mean(dim=1)
            ng_delta4 = average_quaternions(ng_corr_old_delta[:, :, 3:7])
            new_delta[offset:offset + ng.vertex_num] = torch.cat([ng_delta3, ng_delta4], dim=1)
            offset += ng.vertex_num
            old_offset += og.vertex_num

        self.delta.data.copy_(new_delta)

        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
            
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum
            
        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")

        self.reset_optimizer()

    def reset(self, xyz: Float[Tensor, "n 3"]):
        for level_idx in range(self.config.num_grid_levels):
            self.grids[level_idx].reset_p2v(xyz)
            
        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
            
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum
            
        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")
    

    def setup(self, xyz: Float[Tensor, "n 3"], reset_grid: bool = False, seed: Optional[int] = None):
        if not self.grids or reset_grid:
            self.grids, num_grids = self.create_grids(xyz, seed=seed)
            delta = torch.zeros(num_grids, 7, device=xyz.device)
            delta[:, 3] = 1  # 0 0 0 1 0 0 0 -> x y z q1 q2 q3 q4
            self.register_parameter("delta", nn.Parameter(delta))

        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum

        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")
        
        self.reset_optimizer()

    def reg_loss(self):
        """L1 pull of every anchor delta toward identity (weighted by lambda_deform)."""
        identity = torch.zeros_like(self.delta[:1, :])
        identity[:, 3] = 1
        return (self.delta - identity).abs().mean(dim=-1).mean()

    def reset_optimizer(self):
        groups = [{"params": [self.delta], "lr": self.config.lr, "name": "delta"}]
        self.optimizer = torch.optim.Adam(groups, lr=self.config.lr, eps=1e-15)

    def capture(self) -> Dict[str, Any]:
        if not self.grids:
            return {}
        return dict(delta=self.delta, index=self.index)  # TODO grids state

    def restore(self, ckpt: Dict[str, Any]):
        if "delta" in ckpt:
            logging.info("Restore deformation from checkpoint")
            self.register_buffer("index", ckpt["index"])
            self.register_parameter("delta", ckpt["delta"])
            self.reset_optimizer()

    def forward(
        self,
        xyz: Float[Tensor, "n 3"],
        normalized: bool = False,
    ):
        d = self.delta
        if self.config.enable_quant:
            d = self.quantize_delta(d)
        delta = d[self.index].sum(dim=0)
        delta_xyz = delta[:, :3].contiguous()
        delta_rot = delta[:, 3:].contiguous()
        return delta_xyz, self.rotation_activation(delta_rot)

    def quantize_delta(self, d: Float[Tensor, "a 7"]) -> Float[Tensor, "a 7"]:
        """STE-round the per-anchor delta residual to a fixed step (QUEEN-style, round-not-drop).

        xyz is a residual from 0; the quaternion is a residual from identity, so we quantize the
        deviation and add identity back. A residual below step/2 rounds to EXACTLY zero (free
        sparsity + zero drift on static), everything else is preserved to the nearest step.
        """
        sx, sq = self.config.quant_step_xyz, self.config.quant_step_quat
        ident = d.new_zeros(1, 4)
        ident[:, 0] = 1.0  # quaternion (w, x, y, z)
        xyz = ste_round(d[:, :3] / sx) * sx if sx > 0 else d[:, :3]
        qres = d[:, 3:] - ident
        quat = ident + (ste_round(qres / sq) * sq if sq > 0 else qres)
        return torch.cat([xyz, quat], dim=1)

    @staticmethod
    @torch.no_grad()
    def delta_entropy_bytes(d: Float[Tensor, "a 7"], step_xyz: float, step_quat: float) -> float:
        """Entropy-coded size of the transmitted delta, in bytes (QUEEN's accounting,
        gaussian_model.py:855-860): round each channel's residual to its step, then sum the
        per-symbol information -sum(count * log2 p) over all 7 channels. Dropped/static anchors
        become the symbol 0, whose probability mass makes it nearly free -- so this captures the
        sparsity directly, no separate mask term needed."""
        ident = d.new_zeros(1, 7)
        ident[:, 3] = 1.0
        resi = d - ident
        steps = d.new_tensor([step_xyz] * 3 + [step_quat] * 4)
        sym = torch.round(resi / steps.clamp_min(1e-12)).long()  # [A,7] integer symbols
        bits = 0.0
        for j in range(7):
            if steps[j] <= 0:
                continue
            _, counts = torch.unique(sym[:, j], return_counts=True)
            p = counts.float() / counts.sum()
            bits += float((-(p.clamp_min(1e-12).log2()) * counts).sum())
        return bits / 8.0
