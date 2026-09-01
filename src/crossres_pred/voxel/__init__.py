"""Dense voxel-to-voxel cross-resolution distillation.

This package is intentionally independent of the retired point/geometry editor
pipeline. Its only supervision contract is a dense fine-resolution voxel field
registered to a coarse CT voxel grid.
"""

from .model import NNUNetConfig, VoxelNNUNet, initialize_from_m7
from .schema import VoxelPairRecord, load_pair_manifest
from .teacher import TeacherOptions, materialize_teacher

__all__ = [
    "NNUNetConfig",
    "TeacherOptions",
    "VoxelNNUNet",
    "VoxelPairRecord",
    "initialize_from_m7",
    "load_pair_manifest",
    "materialize_teacher",
]
