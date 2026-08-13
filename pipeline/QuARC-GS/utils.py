from typing import Optional
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_min
from pykeops.torch import LazyTensor

def uniform_grid_sampling_optimized(points: torch.Tensor, vertex_num: int, batch_size: int = 100000,
                                    seed: Optional[int] = None) -> torch.Tensor:
    """Pick `vertex_num` anchor points from `points` by uniform grid sampling.

    `seed` makes the result a pure function of (points, vertex_num, seed) -- see the randperm note
    at the bottom of this function. Pass it wherever the RECEIVER must reproduce the anchor set."""
    N, D = points.shape
    device = points.device

    min_xyz = points.min(dim=0).values
    max_xyz = points.max(dim=0).values
    diff = max_xyz - min_xyz

    M = max(int(round(vertex_num ** (1/3))), 1)

    normalized = (points - min_xyz) / (diff + 1e-6)
    grid_indices = (normalized * M).long().clamp(0, M - 1)  # [N, 3]

    z_coords = grid_indices[:, 2]
    unique_z = torch.unique(z_coords.cpu())
    selected_indices_list = []
    
    for z_val in unique_z:
        layer_mask = (z_coords == z_val)
        if layer_mask.sum() == 0:
            continue
        layer_indices = torch.nonzero(layer_mask, as_tuple=True)[0] 
        layer_grid = grid_indices[layer_indices]  # shape: [n_layer, 3]
        cell_idx_layer = layer_grid[:, 0] + layer_grid[:, 1] * M
        cell_idx_layer_cpu = cell_idx_layer.cpu()
        unique_cells_layer_cpu, inverse_layer_cpu = torch.unique(cell_idx_layer_cpu, return_inverse=True)
        unique_cells_layer = unique_cells_layer_cpu.to(device)
        inverse_layer = inverse_layer_cpu.to(device) 
        
        unique_x = (unique_cells_layer % M).float()
        unique_y = (unique_cells_layer // M).float()
        unique_z_val = torch.full_like(unique_x, float(z_val))
        cell_center_layer = torch.stack([unique_x, unique_y, unique_z_val], dim=1) + 0.5
        cell_center_layer = cell_center_layer / M * diff + min_xyz

        layer_points = points[layer_indices]  # [n_layer, 3]
        layer_cell_centers = cell_center_layer[inverse_layer]  # [n_layer, 3]
        layer_dists = ((layer_points - layer_cell_centers) ** 2).sum(dim=1)  # [n_layer]

        num_cells_layer = unique_cells_layer.numel()
        min_vals, argmins = scatter_min(layer_dists, inverse_layer, dim=0, dim_size=num_cells_layer)
        selected_layer_indices = layer_indices[argmins]
        selected_indices_list.append(selected_layer_indices)
    
    if selected_indices_list:
        selected_indices = torch.cat(selected_indices_list, dim=0)
    else:
        selected_indices = torch.empty(0, dtype=torch.long, device=device)
    
    # The grid rarely yields EXACTLY vertex_num cells, so the count is patched here by randperm.
    #
    # Unseeded, these draw from the GLOBAL RNG stream: the result depends on every RNG draw that
    # happened earlier in the process. A receiver is not training, so it has a different RNG history
    # and cannot reproduce the anchor set -- which forces us to TRANSMIT the [3,N] Gaussian->anchor
    # index (~1.16 MB/frame, ~200x the actual payload).
    #
    # With `seed`, the anchor set becomes a pure function of (points, vertex_num, seed). The receiver
    # holds the same canonical geometry bit-exactly and knows the frame number, so it derives the
    # identical index itself and the binding costs ZERO bytes.
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    if selected_indices.numel() > vertex_num:
        perm = torch.randperm(selected_indices.numel(), device=device, generator=gen)
        selected_indices = selected_indices[perm[:vertex_num]]
    elif selected_indices.numel() < vertex_num:
        remaining = vertex_num - selected_indices.numel()
        extra = torch.randperm(N, device=device, generator=gen)[:remaining]
        selected_indices = torch.cat([selected_indices, extra])

    return selected_indices

def batched_knn(A, B, k=3, batch_size_A=10000):
    N = A.shape[0]
    best_dists_list = []
    best_idx_list = []
    
    for i in range(0, N, batch_size_A):
        A_batch = A[i:i+batch_size_A] 
        dists = torch.cdist(A_batch, B)
        batch_dists, batch_idx = torch.topk(dists, k, dim=1, largest=False, sorted=True)
        best_dists_list.append(batch_dists)
        best_idx_list.append(batch_idx)
    
    best_dists = torch.cat(best_dists_list, dim=0)
    best_idx = torch.cat(best_idx_list, dim=0)
    
    return best_dists, best_idx

def batched_keops_knn(A, B, k=1, batch_size=10000):
    N = A.shape[0]
    best_dists_list = []
    best_idx_list = []
    
    for i in range(0, N, batch_size):
        A_batch = A[i:i+batch_size]
        x_i = LazyTensor(A_batch[:, None, :])   # (batch_size, 1, D)
        y_j = LazyTensor(B[None, :, :])           # (1, M, D)
        D_ij = ((x_i - y_j) ** 2).sum(-1)         # (batch_size, M)
        idx= D_ij.argKmin(dim=1, K=k)     
        best_idx_list.append(idx)
        
    return torch.cat(best_idx_list, dim=0)

def average_quaternions(quats: torch.Tensor) -> torch.Tensor:
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ref = quats[:, 0:1, :]  # shape: (N, 1, 4)
    dots = (quats * ref).sum(dim=-1, keepdim=True)  # shape: (N, 3, 1)
    quats = torch.where(dots < 0, -quats, quats)
    outer_products = quats.unsqueeze(-1) * quats.unsqueeze(-2)
    M = outer_products.sum(dim=1)  # shape: (N, 4, 4)
    # cusolver's batched eigh (cusolverDnXsyevBatched) returns CUSOLVER_STATUS_INVALID_VALUE
    # for batch sizes > ~31k on Blackwell/cu128 (verified: 31291 is the largest that works).
    # The grid vertex count exceeds this, so route this tiny 4x4 batched eigh to CPU
    # (~0.07s for 36k mats; reset_grid runs only every grid_reset_interval frames).
    eigenvalues, eigenvectors = torch.linalg.eigh(M.cpu())  # eigenvectors: (N, 4, 4)
    eigenvectors = eigenvectors.to(quats.device)
    avg_quats = eigenvectors[:, :, -1]
    avg_quats = avg_quats / avg_quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return avg_quats