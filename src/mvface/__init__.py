"""Separate import heavy modules (the torch model classes) and light modules so the torch-free subpackage `multi_view.data` still imports in an environment without torch
"""

HEAVY = {
    "RGBDPoseResNet50": "multi_view.backbone",
    "MultiViewBackbone": "multi_view.backbone",
    "MultiViewLandmark3D": "multi_view.mv_model",
}

__all__ = list(HEAVY)


def __getattr__(name):  # PEP 562
    if name in HEAVY:
        import importlib

        return getattr(importlib.import_module(HEAVY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
