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


def triangulate_dlt_batch(
    points_2d: torch.Tensor,
    Ps: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched DLT triangulation

    Args:
        points_2d: (M, V, 2) pixel coordinate of the same world point.
        Ps: (M, V, 3, 4) projection matrices P = K @ Rt, one per view.
        weights: (M, V) optional per-view confidence.

    Returns:
        (M, 3) triangulated 3D world points.
    """
    p1 = Ps[..., 0, :]
    p2 = Ps[..., 1, :]
    p3 = Ps[..., 2, :]

    u = points_2d[..., 0:1]
    v = points_2d[..., 1:2]

    row_u = u * p3 - p1
    row_v = v * p3 - p2

    if weights is not None:
        w = weights[..., None]
        row_u = row_u * w
        row_v = row_v * w

    A = torch.cat([row_u, row_v], dim=1)
    U, S, Vh = torch.linalg.svd(A)
    Xh = Vh[:, -1, :]
    
    # Dehomogenize with a signed floor on the homogeneous coord: a near-zero w (point near infinity / degenerate rays) would otherwise blow X up to Inf.
    w = Xh[:, 3:4]
    w_safe = w.abs().clamp(min=1e-8) * torch.where(w < 0, -1.0, 1.0)
    X = Xh[:, :3] / w_safe
    return X
