"""Lazily expose the torch-heavy model classes so importing `mvface` (or the
`mvface.data` subpackage) does not eagerly import torch/torchvision.
"""

HEAVY = {
    "RGBDPoseResNet50": "mvface.backbone",
    "MultiViewBackbone": "mvface.backbone",
    "MultiViewLandmark3D": "mvface.model",
}

__all__ = list(HEAVY)


def __getattr__(name):  # PEP 562
    if name in HEAVY:
        import importlib

        return getattr(importlib.import_module(HEAVY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
