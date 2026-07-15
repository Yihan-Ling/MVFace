"""Data tooling for the multi-view head-pose project.

Currently holds the FaceScape multi-view reader/geometry helpers used to generate
virtual RGB-D data. Kept dependency-light: the pure-parameter helpers import only
numpy, while mesh/render helpers import trimesh lazily so this package can also be
imported from the torch training env (which does not have the render stack).
"""

from multi_view.data import facescape_reader

__all__ = ["facescape_reader"]
