"""Voxel-domain teacher/student surface prediction (crossres v2).

Fine-resolution teachers (the released m7 nnU-Net fine-tuned at 2.4 and
1.129 um) supervise an enhanced coarse-pitch drop-in replacement for m7,
entirely through R^3 -> R^3 volume operations.
"""

from .losses import SurfaceObjective
from .model import SurfaceModelConfig, SurfaceNet
from .policy import DataPolicy, PolicyError
from .schema import PairRecord, ScanSpec, SurfacePair

__all__ = [
    "DataPolicy",
    "PairRecord",
    "PolicyError",
    "ScanSpec",
    "SurfaceModelConfig",
    "SurfaceNet",
    "SurfaceObjective",
    "SurfacePair",
]

__version__ = "2.0.0"
