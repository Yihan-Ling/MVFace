from __future__ import annotations

import torch


def project(points_3d: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project 3D world points into 2D pixels

    Args:
        points_3d: (N, 3) world-frame points.
        P: (3, 4) view specific projection matrix P = K @ Rt.

    Returns:
        (N, 2) pixel coordinates (u, v)
    """
    ones = torch.ones((points_3d.shape[0], 1), dtype=points_3d.dtype, device=points_3d.device)
    points_h = torch.cat((points_3d, ones), dim=-1)
    
    proj = points_h @ P.T

    uv = proj[:, :2] / proj[:, 2:3]
    return uv


def triangulate_dlt(
    points_2d: torch.Tensor,
    Ps: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct Linear Transform (DLT) Triangulation. Recover 3D position of a poitn given 2D coordinates across views

    Args:
        points_2d: (V, 2) pixel coordinate of the same world point.
        Ps: (V, 3, 4) projection matrices P = K @ Rt, one per view.
        weights: (V,) optional per-view confidence.

    Returns:
        (3,) world-frame 3D point.
    """

    p1 = Ps[:, 0, :]
    p2 = Ps[:, 1, :]
    p3 = Ps[:, 2, :]

    u = points_2d[:, 0:1]
    v = points_2d[:, 1:2]

    # two DLT rows per view:  (u*p3 - p1) . Xh = 0 ,  (v*p3 - p2) . Xh = 0
    row_u = u*p3 - p1
    row_v = v*p3 - p2
    
    # apply confidence weighting if any
    if weights is not None:
        w = weights[:, None]
        row_u = row_u * w
        row_v = row_v * w

    # stack into A
    A = torch.cat([row_u, row_v], dim=0)

    # run SVD and get the least-squares null-space solution
    U, S, Vh = torch.linalg.svd(A)
    Xh = Vh[-1]

    # dehomogenize
    X = Xh[:3]/Xh[3]
    return X

def _normalize_rows(a: torch.Tensor) -> torch.Tensor:
    """Scale each row to unit L2 norm, with a floor so an all-zero row stays finite."""
    return a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)

def triangulate_dlt_batch(
    points_2d: torch.Tensor,
    Ps: torch.Tensor,
    weights: torch.Tensor | None = None,
    depths: torch.Tensor | None = None,             # (M, V) eye-space z, WORLD units
    depth_weights: torch.Tensor | None = None,      # (M, V) per-view depth reliability
    lambda_depth: float = 1.0,                      # global depth-vs-2D trade-off
) -> torch.Tensor:
    """Batched DLT triangulation, optionally augmented with per-view depth rows

    Args:
        points_2d: (M, V, 2) pixel coordinate of the same world point.
        Ps: (M, V, 3, 4) projection matrices P = K @ Rt, one per view.
        weights: (M, V) optional per-view confidence.
        depths: (M, V)  metric z-depth in mm (eye-space z along optical axis)
        depth_weights: (M, V)  depth reliability
        lambda_depth: global depth-vs-2D trade-off

    Returns:
        (M, 3) triangulated 3D world points.
    """
    p1 = Ps[..., 0, :]
    p2 = Ps[..., 1, :]
    p3 = Ps[..., 2, :]

    u = points_2d[..., 0:1]
    v = points_2d[..., 1:2]

    row_u = _normalize_rows(u * p3 - p1)
    row_v = _normalize_rows(v * p3 - p2)

    if weights is not None:
        w = weights[..., None]
        row_u = row_u * w
        row_v = row_v * w

    rows = [row_u, row_v]

    # optional depth rows
    if depths is not None:
        # depth constraint row:  [p3_xyz, p3_w - d] . Xh = 0
        row_d = torch.cat(
            [p3[..., :3], p3[..., 3:4] - depths[..., None]],
            dim=-1,
        )   # (M, V, 4)
        row_d = _normalize_rows(row_d)
        if depth_weights is not None:
            row_d = row_d * (lambda_depth * depth_weights[..., None])
        elif lambda_depth != 1.0:
            row_d = row_d * lambda_depth
        rows.append(row_d)

    A = torch.cat(rows, dim=1)
    _, _, Vh = torch.linalg.svd(A)
    Xh = Vh[:, -1, :]

    # Dehomogenize with a signed floor on the homogeneous coord: a near-zero w (point near infinity / degenerate rays) would otherwise blow X up to Inf.
    w = Xh[:, 3:4]
    w_safe = w.abs().clamp(min=1e-8) * torch.where(w < 0, -1.0, 1.0)
    X = Xh[:, :3] / w_safe
    return X


def camera_depth(points_3d: torch.Tensor, Ps: torch.Tensor) -> torch.Tensor:
    """Camera-space z (eye-space depth) of each world point in each view.

    p3 . [X;1] equals eye-space z when P = K @ [R|t] with K's last row [0,0,1].
    Use it to form depth residuals  r = camera_depth(X) - measured_depth  for the
    robust reweighting / correction loop (weights_depth = vis * valid * rho(r)).

    Args:
        points_3d: (M, 3) world points.
        Ps: (M, V, 3, 4) per-point projection matrices, or (V, 3, 4) for a shared
            rig (broadcast over M).
    Returns:
        (M, V) predicted eye-space depth per view (same units as world coords).
    """
    ones = points_3d.new_ones(points_3d.shape[0], 1)
    Xh = torch.cat([points_3d, ones], dim=-1)            # (M, 4)
    p3 = Ps[..., 2, :]                                   # (M, V, 4) or (V, 4)
    return (p3 * Xh[:, None, :]).sum(dim=-1)             # (M, V)
