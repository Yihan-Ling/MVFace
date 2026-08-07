"""GT unit / frame consistency check for the FaceScape multi-view pipeline.

Run this on ONE real capture from your dataloader BEFORE trusting late-fusion
(depth-row) results or MPJPE numbers. It verifies, on ground-truth landmarks,
that all four coordinate conventions agree:

  1. project(landmarks_3d, proj)   ~= landmarks_2d        (3D / proj / 2D frame)
  2. triangulate(landmarks_2d)     ~= landmarks_3d        (2D -> 3D round trip)
  3. camera_depth(landmarks_3d)    is sane positive mm    (metric scale)
  4. depth_raw sampled at landmarks ~= camera_depth       (**depth units** — the
     one that silently biases every depth row if it's wrong, e.g. mm vs m)
  5. triangulate(+depth rows)      ~= landmarks_3d        (depth-augmented solve)
  6. mpjpe(gt, gt) == 0 and IOD is face-scale (~90-100 mm)

Expected shapes for ONE capture (V = number of views, J = 68 landmarks):
    landmarks_3d : (J, 3)      world-frame, same frame proj projects from
    landmarks_2d : (V, J, 2)   pixels (u, v)
    proj         : (V, 3, 4)   P = K' @ [R|t]
    depth_raw    : (V, H, W)   metric depth map (raw mm)
    vis          : (V, J)      1 = landmark visible in that view
"""
from __future__ import annotations

import torch

import geometry as G


def _sample_depth(depth_raw: torch.Tensor, uv: torch.Tensor):
    """Nearest-neighbour sample a depth map at pixel coords.

    Args:
        depth_raw: (V, H, W) depth maps.
        uv:        (V, J, 2) pixel coords (u=col, v=row).
    Returns:
        depth: (V, J) sampled depth; valid: (V, J) bool in-bounds & finite & >0.
    """
    V, H, W = depth_raw.shape
    u = uv[..., 0].round().long()
    v = uv[..., 1].round().long()
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uc = u.clamp(0, W - 1)
    vc = v.clamp(0, H - 1)
    vidx = torch.arange(V, device=depth_raw.device)[:, None].expand_as(uc)
    depth = depth_raw[vidx, vc, uc]
    valid = in_bounds & torch.isfinite(depth) & (depth > 0)
    return depth, valid


def check_geometry_units(
    landmarks_3d: torch.Tensor,   # (J, 3)
    landmarks_2d: torch.Tensor,   # (V, J, 2)
    proj: torch.Tensor,           # (V, 3, 4)
    depth_raw: torch.Tensor,      # (V, H, W)
    vis: torch.Tensor,            # (V, J)
    tol_px: float = 1.5,
    tol_mm: float = 5.0,
    verbose: bool = True,
) -> dict:
    dt = torch.float64
    lm3 = landmarks_3d.to(dt)
    lm2 = landmarks_2d.to(dt)
    P = proj.to(dt)
    depth_raw = depth_raw.to(dt)
    vis_b = vis.bool()
    V, J = lm2.shape[0], lm2.shape[1]
    out, ok = {}, True

    def report(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        out[name] = passed
        if verbose:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # 1) projection matches stored 2D
    proj_err = []
    for i in range(V):
        uv_i = G.project(lm3, P[i])                       # (J, 2)
        e = torch.linalg.norm(uv_i - lm2[i], dim=-1)      # (J,)
        proj_err.append(e[vis_b[i]] if vis_b[i].any() else e)
    proj_err = torch.cat(proj_err)
    report("1_project_vs_2d", proj_err.median() < tol_px,
           f"median {proj_err.median():.3f}px (tol {tol_px})")

    # reshape to (J, V, ...) for the batch triangulator
    P_JV = P[None].expand(J, V, 3, 4)                     # (J, V, 3, 4)
    uv_JV = lm2.permute(1, 0, 2).contiguous()            # (J, V, 2)
    vis_JV = vis_b.permute(1, 0).contiguous().to(dt)     # (J, V)

    # 2) 2D-only round trip
    X_2d = G.triangulate_dlt_batch(uv_JV, P_JV, weights=vis_JV)
    e2 = torch.linalg.norm(X_2d - lm3, dim=-1)
    report("2_triangulate_2d_roundtrip", e2.median() < tol_mm,
           f"median {e2.median():.4f}mm (tol {tol_mm})")

    # 3) camera-space depth is sane positive metric
    cam_d = G.camera_depth(lm3, P_JV)                     # (J, V)
    cam_d_vis = cam_d[vis_JV.bool()]
    report("3_camera_depth_sane",
           bool((cam_d_vis > 0).all()) and (50 < cam_d_vis.median() < 5000),
           f"median {cam_d_vis.median():.1f}mm, range "
           f"[{cam_d_vis.min():.1f}, {cam_d_vis.max():.1f}]")

    # 4) THE units check: measured depth at landmark pixels vs camera_depth
    meas_d, meas_valid = _sample_depth(depth_raw, lm2)   # (V, J)
    meas_d = meas_d.permute(1, 0)                        # (J, V)
    meas_valid = meas_valid.permute(1, 0) & vis_JV.bool()
    if meas_valid.any():
        diff = (meas_d - cam_d).abs()[meas_valid]
        report("4_depth_units_match", diff.median() < tol_mm,
               f"median |measured - camera_z| {diff.median():.3f}mm over "
               f"{int(meas_valid.sum())} valid pts (tol {tol_mm}) — if this is "
               f"~1000x off, depth is in the wrong unit (mm vs m)")
    else:
        report("4_depth_units_match", False, "no valid sampled depths to compare")

    # 5) depth-augmented solve still recovers GT (units consistent inside solve)
    dw = (meas_valid.to(dt) * vis_JV)
    X_d = G.triangulate_dlt_batch(uv_JV, P_JV, weights=vis_JV,
                                  depths=meas_d, depth_weights=dw)
    e5 = torch.linalg.norm(X_d - lm3, dim=-1)
    report("5_triangulate_with_depth", e5.median() < tol_mm,
           f"median {e5.median():.4f}mm (tol {tol_mm})")

    # 6) MPJPE self-consistency + face-scale IOD (iBUG outer eye corners 36 / 45)
    zero = float(G.mpjpe(lm3[None], lm3[None]))
    iod = float(torch.linalg.norm(lm3[36] - lm3[45]))
    report("6_mpjpe_and_iod", zero < 1e-6 and (60 < iod < 140),
           f"mpjpe(gt,gt)={zero:.2e}, IOD={iod:.1f}mm (expect face-scale ~96)")

    if verbose:
        print(f"\n{'ALL CHECKS PASSED' if ok else '>>> SOME CHECKS FAILED <<<'}")
    out["all_passed"] = ok
    return out


if __name__ == "__main__":
    # ---- pull ONE capture from your dataloader and pass its fields ----
    # Adjust the import/instantiation to your actual dataset (memory: dataset at
    # src/data/facescape_multiview.py, imports use the src.* prefix).
    #
    #   from src.data.facescape_multiview import FaceScapeMultiView
    #   ds = FaceScapeMultiView(split="val", ...)          # your usual args
    #   b  = ds[0]                                          # one capture (dict)
    #   check_geometry_units(
    #       landmarks_3d=b["landmarks_3d"],   # (68, 3)
    #       landmarks_2d=b["landmarks_2d"],   # (V, 68, 2)
    #       proj        =b["proj"],           # (V, 3, 4)
    #       depth_raw   =b["depth_raw"],      # (V, H, W)
    #       vis         =b["vis"],            # (V, 68)
    #   )
    raise SystemExit("Fill in the dataloader block above, then re-run.")
