# MVFace

Multi-view 3D facial landmark estimation from calibrated RGB-D cameras.

MVFace predicts 68 3D facial landmarks for a face observed from $N$ synchronized,calibrated RGB-D views. As of Iteration 1, depth enters as a fourth input channel trained from scratch.

The architecture is a simplified adaptation of [MVGFormer](https://github.com/XunshanMan/MVGFormer) (which is a multi-view multi-person joint pose estimation model):
The pipeline is consisted of a ResNet-50 backbone; a 4-layer DETR-style decoder that refines a set of 3D landmark queries by projecting them into every view, sampling and self-attention, 
then re-triangulating via a differentiable Direct Linear Transform (DLT). The full pipeline is implemented in pure PyTorch. 

Iteration 1 training and testing uses custom rendered face models from [FaceScape](https://nju-3dv.github.io/projects/FaceScape/) dataset. 

## Repository layout

```
MVFace/
├── pyproject.toml              # packaging + dependencies (installable, src-layout)
├── src/
│   └── mvface/
│       ├── model.py            # MultiViewLandmark3D (top-level model)
│       ├── backbone.py         # RGBDPoseResNet50 + MultiViewBackbone
│       ├── decoder.py          # projective attention, query update, decoder
│       ├── geometry.py         # project() and (batched) DLT triangulation helper functions
│       ├── losses.py           # losses + MPJPE metric
│       ├── data/
│       │   └── facescape_dataset.py   # MultiViewFaceScape dataset + split helpers
│       └── assets/             # mean-face template + query-space config
└── tools/
    ├── train.py                # training entry point
    └── _init_paths.py          # adds src/ to sys.path for uninstalled runs
```

## Installation

Requires Python 3.10+ and, for GPU training, an NVIDIA GPU with a recent driver.

```bash
# from the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

This installs `mvface` as an editable package along with its dependencies
(PyTorch, torchvision, NumPy, SciPy, Pillow). The default PyTorch wheel is
CUDA-enabled; verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `False`, install a PyTorch build matching your CUDA version from
<https://pytorch.org/get-started/locally/>, then re-run `pip install -e .`.

## Usage

### Training

Training data is not included (FaceScape is license-gated). Point the trainer at a
local dataset root via `--root`:

```bash
python tools/train.py --root <path/to/data> --epochs 60 --bs 2 --lr 5e-5
```

A quick smoke test on a couple of subjects:

```bash
python tools/train.py --root <path/to/data> --limit 2 --epochs 1
```

Run `python tools/train.py --help` for the full list of options.

### Evaluation
TO BE ADDED

## Dataset
TO BE ADDED