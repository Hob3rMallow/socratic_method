from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import sha256_file


class BaselinePredictionError(RuntimeError):
    pass


class M7BaselinePredictor:
    """Run the released Scroll Prize m7 nnU-Net on coarse-native patches."""

    def __init__(
        self,
        model_directory: str | Path,
        *,
        device: str = "cuda",
        fold: int = 0,
        checkpoint_name: str = "checkpoint_best.pth",
    ) -> None:
        self.model_directory = Path(model_directory).expanduser().resolve()
        self.fold = int(fold)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = (
            self.model_directory / f"fold_{self.fold}" / checkpoint_name
        )
        for path in (
            self.model_directory / "dataset.json",
            self.model_directory / "plans.json",
            self.checkpoint_path,
        ):
            if not path.is_file():
                raise BaselinePredictionError(f"m7 model file is missing: {path}")

        try:
            import torch
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise BaselinePredictionError(
                "m7 baseline inference requires the 'baseline' optional dependencies"
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BaselinePredictionError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=device.startswith("cuda"),
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(self.model_directory),
            use_folds=(self.fold,),
            checkpoint_name=self.checkpoint_name,
        )
        patch_size = tuple(
            int(item) for item in self._predictor.configuration_manager.patch_size
        )
        if patch_size != (192, 192, 192):
            raise BaselinePredictionError(
                f"expected released m7 patch size (192, 192, 192), got {patch_size}"
            )

    def predict(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3:
            raise BaselinePredictionError(
                f"m7 input must be a 3-D patch, got shape {image.shape}"
            )
        if tuple(image.shape) != (192, 192, 192):
            raise BaselinePredictionError(
                "m7 fallback currently requires a 192x192x192 training patch; "
                f"got {tuple(image.shape)}"
            )
        segmentation = self._predictor.predict_single_npy_array(
            image[np.newaxis],
            {"spacing": [1.0, 1.0, 1.0]},
        )
        segmentation = np.asarray(segmentation)
        if segmentation.shape != image.shape:
            raise BaselinePredictionError(
                f"m7 returned shape {segmentation.shape}, expected {image.shape}"
            )
        # The released dataset uses label 1 for surface and label 2 as ignore.
        return (segmentation == 1).astype(np.uint8) * np.uint8(255)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "scrollprize-surface-m7-nnunet",
            "model_directory": str(self.model_directory),
            "fold": self.fold,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "dataset_sha256": sha256_file(self.model_directory / "dataset.json"),
            "plans_sha256": sha256_file(self.model_directory / "plans.json"),
            "nnunetv2": importlib.metadata.version("nnunetv2"),
            "device": str(self.device),
            "use_mirroring": False,
            "tile_step_size": 0.5,
        }
