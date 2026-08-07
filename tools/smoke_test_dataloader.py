"""
Smoke test for facescape_multiview.py

    python smoke_test_dataloader.py

Checks (no training involved):
  1. Dataset builds and returns the expected keys / shapes / dtypes.
  2. depth_raw is in a sane mm range (face ~ hundreds of mm, not normalized).
  3. Projected landmarks_2d land inside the image and mostly on the face
     (i.e. near non-zero depth), which confirms proj + centering are consistent.
  4. Saves an RGB overlay per view (landmarks drawn) to ./smoke_out/ so you can
     eyeball that points sit on eyes/nose/mouth.

Nothing here writes to the dataset or trains anything.
"""

import os
import sys
import types
import numpy as np
import cv2
import torch

# ── minimal cfg shim so we don't need the full MVGFormer config ───────────────
def make_cfg():
    cfg = types.SimpleNamespace()
    cfg.DATASET = types.SimpleNamespace(
        ROOT='/nfs/turbo/coe-igmr-pub/seoin/FaceScape',
        NUM_VIEWS=5,
        USE_RETINAFACE=False,           # set True to test the crop path
        DEPTH_SCALE=200.0,
        RETINAFACE_ROOT='/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface',
        RETINAFACE_CHECKPOINT='/nfs/turbo/coe-igmr-pub/seoin/trained_weights/Resnet50_Final.pth',
    )
    cfg.NETWORK = types.SimpleNamespace(IMAGE_SIZE=(256, 256))
    return cfg


def main():
    from data.facescape_multiview import FaceScapeMultiView, NUM_LANDMARKS

    cfg = make_cfg()
    ds = FaceScapeMultiView(cfg, image_set='train', is_train=True)
    print(f'dataset size: {len(ds)}')
    assert len(ds) > 0, 'empty dataset — check ROOT / splits'

    sample = ds[0]

    # ── 1. keys / shapes / dtypes ─────────────────────────────────────────────
    N, Ht, Wt = cfg.DATASET.NUM_VIEWS, *cfg.NETWORK.IMAGE_SIZE
    expect = {
        'rgbd':         (N, 4, Ht, Wt),
        'depth_raw':    (N, Ht, Wt),
        'proj':         (N, 3, 4),
        'landmarks_3d': (NUM_LANDMARKS, 3),
        'landmarks_2d': (N, NUM_LANDMARKS, 2),
        'vis':          (N, NUM_LANDMARKS),
    }
    for k, shp in expect.items():
        assert k in sample, f'missing key: {k}'
        got = tuple(sample[k].shape)
        assert got == shp, f'{k}: shape {got} != {shp}'
        assert sample[k].dtype == torch.float32, f'{k}: dtype {sample[k].dtype}'
    print('shapes/dtypes OK:', {k: tuple(v.shape) for k, v in sample.items()})

    # ── 2. depth_raw sane mm range ────────────────────────────────────────────
    d = sample['depth_raw'].numpy()
    face = d > 0
    face_vals = d[face]
    print(f'depth_raw mm: min>0={face_vals.min():.1f} '
          f'median={np.median(face_vals):.1f} max={face_vals.max():.1f} '
          f'face_coverage={face.mean():.3f}')
    assert face_vals.min() > 1.0, 'depth looks normalized, not raw mm'
    assert np.median(face_vals) < 1e5, 'depth median implausibly large'

    # ── 3. landmarks_3d scale (IOD ~96mm TU-scale) ────────────────────────────
    lm3d = sample['landmarks_3d'].numpy()
    iod = np.linalg.norm(lm3d[45] - lm3d[36])
    print(f'IOD (3D, mm): {iod:.2f}   3D centroid: {lm3d.mean(0)}')
    assert 50 < iod < 160, f'IOD {iod:.1f}mm out of expected face range'
    assert np.allclose(lm3d.mean(0), 0, atol=1.0), 'landmarks not centered'

    # ── 4. projected 2D consistency + on-face check ───────────────────────────
    lm2d = sample['landmarks_2d'].numpy()
    vis  = sample['vis'].numpy()
    for vidx in range(N):
        u = lm2d[vidx, :, 0]; v = lm2d[vidx, :, 1]
        in_bounds = (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)
        frac_in = in_bounds.mean()

        # how many visible landmarks sit on non-zero depth (on the face surface)
        ui = np.clip(np.round(u).astype(int), 0, Wt - 1)
        vi = np.clip(np.round(v).astype(int), 0, Ht - 1)
        on_face = (d[vidx][vi, ui] > 0)
        vis_on_face = on_face[vis[vidx] > 0].mean() if (vis[vidx] > 0).any() else 0.0
        print(f'  view {vidx}: in_bounds={frac_in:.2f} '
              f'vis_count={int(vis[vidx].sum())} vis_on_face={vis_on_face:.2f}')
        assert frac_in > 0.7, f'view {vidx}: too many landmarks out of frame'

    # ── 5. save overlays for visual inspection ────────────────────────────────
    os.makedirs('smoke_out', exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    for vidx in range(N):
        rgb = sample['rgbd'][vidx, :3].numpy().transpose(1, 2, 0)
        rgb = (rgb * std + mean)                      # de-normalize
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for j in range(NUM_LANDMARKS):
            uu, vv = int(round(lm2d[vidx, j, 0])), int(round(lm2d[vidx, j, 1]))
            color = (0, 255, 0) if vis[vidx, j] > 0 else (0, 0, 255)  # green=vis, red=occl
            if 0 <= uu < Wt and 0 <= vv < Ht:
                cv2.circle(rgb, (uu, vv), 2, color, -1)
        out = f'smoke_out/view{vidx}_overlay.png'
        cv2.imwrite(out, rgb)
        print(f'  wrote {out}')

    # ── 6. default collate → batch dim ────────────────────────────────────────
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print('batched rgbd:', tuple(batch['rgbd'].shape),
          'proj:', tuple(batch['proj'].shape),
          'depth_raw:', tuple(batch['depth_raw'].shape))
    assert batch['rgbd'].shape[0] == 2

    print('\nALL SMOKE CHECKS PASSED — inspect smoke_out/*.png to confirm points sit on faces.')


if __name__ == '__main__':
    sys.exit(main())
