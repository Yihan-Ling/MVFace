"""
Turn multi-camera calibration into projection matrices for the MVFace pipeline.

Convention:
    P = K @ [R | t]             world point X (world/board frame) -> pixels
    X_cam = R @ X_world + t     so [R|t] maps WORLD -> CAMERA (solvePnP output).
    p3 . [X;1] = camera-space z = the metric depth used by the depth constraint.

Units: keep the WORLD frame in metres
so results land in the same frame the model was trained on (landmarks ~0.1 m, MM_PER_UNIT=1000).
RealSense depth is native mm -> multiply by 1e-3 before using as `depths`.
"""

"""
rig_calib.json format:
{
  "world_frame": "custom",
  "units": "m",
  "view_order": ["serial0", "serial1", "serial2"],
  "cameras": {
    "serial0": {"K": [[...],[...],[...]], "dist": [0,0,0,0,0],
                "R": [[...],[...],[...]], "t": [tx, ty, tz]},
    "serial1": { ... },
    "serial2": { ... }
  }
}
"""
from __future__ import annotations

import json
import numpy as np
import torch


def build_proj(K, R, t) -> torch.Tensor:
    """P = K @ [R|t].  K (3,3), R (3,3), t (3,) -> P (3,4).  world->pixels."""
    K = torch.as_tensor(K, dtype=torch.float32)
    R = torch.as_tensor(R, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(3, 1)
    Rt = torch.cat([R, t], dim=1)            # (3,4)  world->camera
    return K @ Rt                            # (3,4)  world->pixels


def load_rig(path: str):
    """Load a rig_calib.json written by calibrate_rig.py.

    Returns list (ordered by 'view_order') of dicts: {serial, K, R, t, dist}.
    """
    d = json.load(open(path))
    cams = d["cameras"]
    order = d.get("view_order", sorted(cams))
    out = []
    for s in order:
        c = cams[s]
        out.append({
            "serial": s,
            "K": np.array(c["K"], dtype=np.float32),
            "R": np.array(c["R"], dtype=np.float32),
            "t": np.array(c["t"], dtype=np.float32),
            "dist": np.array(c.get("dist", np.zeros(5)), dtype=np.float32),
        })
    return out


def proj_stack(rig) -> torch.Tensor:
    """(V,3,4) stack of P matrices, ordered as in the rig list."""
    return torch.stack([build_proj(c["K"], c["R"], c["t"]) for c in rig], dim=0)


def camera_depth_of(points_3d: torch.Tensor, Ps: torch.Tensor) -> torch.Tensor:
    """Camera-space z of world points per view: p3 . [X;1]. (M,3),(V,3,4)->(M,V)."""
    ones = points_3d.new_ones(points_3d.shape[0], 1)
    Xh = torch.cat([points_3d, ones], dim=-1)           # (M,4)
    p3 = Ps[:, 2, :]                                     # (V,4)
    return Xh @ p3.T                                     # (M,V)
