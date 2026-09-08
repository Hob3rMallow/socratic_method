from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import NNUNetConfig, VoxelNNUNet, initialize_from_m7
from .patches import normalize_m7_ct
from .resources import assert_cuda_power_limit, configure_cpu_budget
from .scrollfiesta_metrics import (
    ScrollFiestaPredMetrics,
    scrollfiesta_patch_pred_metrics,
)

M7_PATHOLOGY_SCORE_SCHEMA = "crossres-m7-pathology-score-v2"
M7_PATHOLOGY_STATE_SCHEMA = "crossres-m7-pathology-mining-state-v1"
M7_PATHOLOGY_IDENTITY_SCHEMA = "crossres-m7-pathology-mining-identity-v1"
M7_LOCAL_INFERENCE_CONTRACT = (
    "released-m7-local-192-fp16-no-tta-th0.2-center128-v2"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class M7PathologyMiningOptions:
    batch_size: int = 3
    device: str = "cuda"
    amp_dtype: str = "float16"
    threshold: float = 0.2
    splits: tuple[str, ...] = ("train", "val")
    max_cpu_threads: int = 4
    max_patches: int | None = None
    expected_count: int | None = None

    def validate(self) -> None:
        if not 1 <= self.batch_size <= 8:
            raise ValueError("batch_size must be in [1, 8]")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must be non-empty and unique")
        if not set(self.splits) <= {"train", "val"}:
            raise ValueError("pathology mining is restricted to train/val hold-in data")
        if not 1 <= self.max_cpu_threads <= 16:
            raise ValueError("max_cpu_threads must be in [1, 16]")
        if self.max_patches is not None and self.max_patches <= 0:
            raise ValueError("max_patches must be positive")
        if self.expected_count is not None and self.expected_count <= 0:
            raise ValueError("expected_count must be positive")

    def identity_options(self) -> dict[str, Any]:
        value = asdict(self)
        # A bounded invocation must be resumable later without changing the
        # numerical inference identity.
        value.pop("max_patches")
        value.pop("expected_count")
        value.pop("max_cpu_threads")
        return value


class M7LocalPatchPredictor:
    """Released m7 inference on immutable 192-cube corpus patches.

    The centered 128 cube is the ScrollFiesta deployment domain.  This local
    contract is deliberately distinct from a published whole-volume m7 field,
    whose globally anchored Gaussian blend sees additional neighboring tiles.
    It is a conservative hard-example ranker, never a replacement label.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        amp_dtype: str,
        threshold: float,
    ) -> None:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        assert_cuda_power_limit(device)
        self.device = device
        self.amp_dtype = (
            torch.float16 if amp_dtype == "float16" else torch.bfloat16
        )
        self.threshold = float(threshold)
        self.model = VoxelNNUNet(NNUNetConfig(preset="m7-resenc-l"))
        self.initialization = initialize_from_m7(self.model, checkpoint)
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("m7 prediction batch must not be empty")
        normalized: list[np.ndarray] = []
        for image in images:
            raw = np.asarray(image)
            if raw.shape != (192, 192, 192):
                raise ValueError(
                    f"m7 pathology input must be 192^3, got {raw.shape}"
                )
            normalized.append(normalize_m7_ct(raw))
        tensor = torch.from_numpy(np.stack(normalized)[:, None]).to(
            self.device,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model.full_resolution_logits(tensor)
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
        return (
            (probability >= self.threshold)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )


def score_m7_prediction(
    target_u8: np.ndarray,
    prediction_u8: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target_u8)
    prediction = np.asarray(prediction_u8)
    if target.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("target and prediction must be matching 3-D arrays")
    labels = {int(value) for value in np.unique(target)}
    if not labels <= {0, 1, 2}:
        raise ValueError(f"target contains invalid labels: {sorted(labels)}")
    if not {int(value) for value in np.unique(prediction)} <= {0, 1}:
        raise ValueError("m7 prediction must contain only 0/1")
    prediction_metrics = scrollfiesta_patch_pred_metrics(prediction)
    teacher_metrics = scrollfiesta_patch_pred_metrics(target == 1)
    origin = prediction_metrics.window_origin_zyx
    shape = prediction_metrics.window_shape_zyx
    window = tuple(
        slice(lower, lower + extent)
        for lower, extent in zip(origin, shape, strict=True)
    )
    target = target[window]
    prediction = prediction[window]
    known = target != 2
    known_voxels = int(np.count_nonzero(known))
    truth = target == 1
    predicted = prediction.astype(bool, copy=False)
    true_positive = int(np.count_nonzero(predicted & truth & known))
    false_positive = int(np.count_nonzero(predicted & ~truth & known))
    false_negative = int(np.count_nonzero(~predicted & truth & known))
    true_negative = known_voxels - true_positive - false_positive - false_negative
    denominator = 2 * true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "evaluation_window_origin_zyx": list(origin),
        "evaluation_window_shape_zyx": list(shape),
        "known_voxels": known_voxels,
        "target_positive_voxels": recall_denominator,
        "prediction_positive_voxels_known": precision_denominator,
        "true_positive_voxels": true_positive,
        "false_positive_voxels": false_positive,
        "false_negative_voxels": false_negative,
        "true_negative_voxels": true_negative,
        # A patch may have valid supervision in the 192-cube halo but no known
        # teacher voxel in the centered 128 deployment window. It is still a
        # corpus row and must receive a durable rank record. The empty known
        # domain carries no evidence of pathology, so rank it conservatively at
        # zero rather than aborting the entire append-only mining run.
        "pathology_score": (
            (false_positive + false_negative) / known_voxels
            if known_voxels
            else 0.0
        ),
        "dice": 2 * true_positive / denominator if denominator else 1.0,
        "precision": (
            true_positive / precision_denominator
            if precision_denominator
            else 1.0
        ),
        "recall": (
            true_positive / recall_denominator if recall_denominator else 1.0
        ),
        "scrollfiesta_pred_metrics": prediction_metrics.to_dict(),
        "scrollfiesta_teacher_metrics": teacher_metrics.to_dict(),
        "baseline_rejected_teacher_kept": (
            prediction_metrics.rejected and not teacher_metrics.rejected
        ),
    }


def _mining_identity(
    *,
    patch_manifests: tuple[Path, ...],
    checkpoint: Path,
    corpus_plan: Path | None,
    options: M7PathologyMiningOptions,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": M7_PATHOLOGY_IDENTITY_SCHEMA,
        "inference_contract": M7_LOCAL_INFERENCE_CONTRACT,
        "patch_manifests": [str(path) for path in patch_manifests],
        "m7_checkpoint": str(checkpoint),
        "m7_checkpoint_sha256": _sha256(checkpoint),
        "options": options.identity_options(),
        "patch_shape_zyx": [192, 192, 192],
        "evaluation_window_origin_zyx": [32, 32, 32],
        "evaluation_window_shape_zyx": [128, 128, 128],
        "normalization": "released-m7-fixed-ct",
        "mirror_tta": False,
        "warning": (
            "local 192-cube ranker; not numerically identical to a globally "
            "blended published whole-volume m7 field"
        ),
    }
    if corpus_plan is not None:
        identity["corpus_plan"] = str(corpus_plan)
        identity["corpus_plan_sha256"] = _sha256(corpus_plan)
    return json.loads(json.dumps(identity, sort_keys=True))


def _load_durable_scores(
    journal: Path,
    state: dict[str, Any],
    *,
    recover_suffix: bool = True,
) -> dict[str, dict[str, Any]]:
    durable_bytes = int(state.get("journal_bytes", -1))
    durable_scores = int(state.get("durable_scores", -1))
    if durable_bytes < 0 or durable_scores < 0:
        raise ValueError("pathology mining state has invalid durable counters")
    size = journal.stat().st_size
    if size < durable_bytes:
        raise ValueError("pathology score journal is shorter than durable state")
    if size > durable_bytes:
        if not recover_suffix:
            raise ValueError("completed pathology journal has a nondurable suffix")
        with journal.open("r+b") as stream:
            stream.truncate(durable_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    raw = journal.read_bytes()
    if len(raw) != durable_bytes:
        raise RuntimeError("could not read the durable pathology journal prefix")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"pathology score journal line {line_number} is invalid"
            ) from error
        _validate_score_record(value)
        patch_id = str(value.get("patch_id", ""))
        archive_sha256 = str(value.get("archive_sha256", ""))
        if not patch_id or len(archive_sha256) != 64:
            raise ValueError("pathology score has invalid patch identity")
        if patch_id in records:
            raise ValueError(f"duplicate pathology score for {patch_id}")
        records[patch_id] = value
    if len(records) != durable_scores:
        raise ValueError("pathology journal count differs from durable state")
    return records


def _validate_score_record(value: dict[str, Any]) -> None:
    if value.get("schema") != M7_PATHOLOGY_SCORE_SCHEMA:
        raise ValueError("unsupported pathology score schema")
    prediction = ScrollFiestaPredMetrics.from_dict(
        value["scrollfiesta_pred_metrics"]
    )
    teacher = ScrollFiestaPredMetrics.from_dict(
        value["scrollfiesta_teacher_metrics"]
    )
    if (
        value.get("evaluation_window_origin_zyx")
        != list(prediction.window_origin_zyx)
        or value.get("evaluation_window_shape_zyx")
        != list(prediction.window_shape_zyx)
        or prediction.window_origin_zyx != (32, 32, 32)
        or prediction.window_shape_zyx != (128, 128, 128)
        or teacher.window_origin_zyx != prediction.window_origin_zyx
        or teacher.window_shape_zyx != prediction.window_shape_zyx
    ):
        raise ValueError("pathology score deployment window is invalid")
    counts = {
        name: int(value[name])
        for name in (
            "known_voxels",
            "target_positive_voxels",
            "prediction_positive_voxels_known",
            "true_positive_voxels",
            "false_positive_voxels",
            "false_negative_voxels",
            "true_negative_voxels",
        )
    }
    if any(count < 0 for count in counts.values()):
        raise ValueError("pathology score has a negative voxel count")
    known = counts["known_voxels"]
    tp = counts["true_positive_voxels"]
    fp = counts["false_positive_voxels"]
    fn = counts["false_negative_voxels"]
    tn = counts["true_negative_voxels"]
    if (
        known < 0
        or known > 128**3
        or tp + fp + fn + tn != known
        or tp + fn != counts["target_positive_voxels"]
        or tp + fp != counts["prediction_positive_voxels_known"]
    ):
        raise ValueError("pathology score confusion counts are inconsistent")
    expected = {
        "pathology_score": (fp + fn) / known if known else 0.0,
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }
    if 2 * tp + fp + fn == 0:
        expected["dice"] = 1.0
    if any(
        not math.isclose(
            float(value[name]),
            expected_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        for name, expected_value in expected.items()
    ):
        raise ValueError("pathology score metrics disagree with voxel counts")
    actionable = prediction.rejected and not teacher.rejected
    if value.get("baseline_rejected_teacher_kept") is not actionable:
        raise ValueError("pathology score actionable flag is inconsistent")
    if (
        not str(value.get("patch_id", ""))
        or len(str(value.get("archive_sha256", ""))) != 64
        or not str(value.get("record_id", ""))
        or not str(value.get("scroll_id", ""))
        or str(value.get("split", "")) not in {"train", "val"}
        or not Path(str(value.get("source_manifest", ""))).is_absolute()
        or int(value.get("source_row_index", -1)) < 0
        or len(str(value.get("mining_identity_sha256", ""))) != 64
    ):
        raise ValueError("pathology score source identity is invalid")


@dataclass(frozen=True)
class CompletedM7PathologyScores:
    output: Path
    state: dict[str, Any]
    records: dict[str, dict[str, Any]]
    journal_sha256: str


def load_completed_m7_pathology_scores(
    output_path: str | Path,
    *,
    expected_count: int | None = None,
    source_manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> CompletedM7PathologyScores:
    """Load a complete, hash-bound score journal without repairing it."""

    output = Path(output_path).expanduser().resolve()
    state_path = output / "state.json"
    journal = output / "scores.jsonl"
    if not state_path.is_file() or not journal.is_file():
        raise FileNotFoundError(f"incomplete pathology output: {output}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema") != M7_PATHOLOGY_STATE_SCHEMA
        or state.get("state") != "complete"
    ):
        raise ValueError("pathology mining is not complete")
    identity = state.get("identity")
    if (
        not isinstance(identity, dict)
        or state.get("identity_sha256") != _canonical_sha256(identity)
    ):
        raise ValueError("pathology mining identity hash is invalid")
    records = _load_durable_scores(journal, state, recover_suffix=False)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"pathology journal has {len(records):,} scores, "
            f"expected {expected_count:,}"
        )
    if int(state.get("durable_scores", -1)) != len(records):
        raise ValueError("pathology completed count differs from journal")
    identity_sha256 = str(state["identity_sha256"])
    if any(
        value.get("mining_identity_sha256") != identity_sha256
        for value in records.values()
    ):
        raise ValueError("pathology score identity differs from completed state")
    if source_manifests is not None:
        manifests = tuple(
            Path(value).expanduser().resolve() for value in source_manifests
        )
        if identity.get("patch_manifests") != [str(path) for path in manifests]:
            raise ValueError("pathology source manifest order changed")
        recorded_hashes = state.get("source_manifest_sha256")
        expected_hashes = {str(path): _sha256(path) for path in manifests}
        if recorded_hashes != expected_hashes:
            raise ValueError("pathology source manifest hashes changed")
    return CompletedM7PathologyScores(
        output=output,
        state=state,
        records=records,
        journal_sha256=_sha256(journal),
    )


def _safe_archive(manifest: Path, row: dict[str, Any]) -> Path:
    relative = Path(str(row.get("path", "")))
    if not relative.as_posix() or relative.is_absolute():
        raise ValueError("patch archive path must be non-empty and relative")
    root = manifest.parent.resolve()
    archive = (root / relative).resolve()
    try:
        archive.relative_to(root)
    except ValueError as error:
        raise ValueError("patch archive escapes its manifest directory") from error
    if not archive.is_file():
        raise FileNotFoundError(archive)
    return archive


def _load_patch_arrays(
    manifest: Path,
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    archive = _safe_archive(manifest, row)
    with np.load(archive, allow_pickle=False) as payload:
        if set(payload.files) != {"image", "target_u8"}:
            raise ValueError(
                f"{archive}: baseline-missing patch archive has unexpected arrays"
            )
        image = np.asarray(payload["image"])
        target = np.asarray(payload["target_u8"])
    if image.shape != (192, 192, 192) or target.shape != image.shape:
        raise ValueError(f"{archive}: pathology mining requires 192-cube arrays")
    return image, target


def _append_scores(
    *,
    journal: Path,
    state_path: Path,
    state: dict[str, Any],
    scores: list[dict[str, Any]],
    observed: dict[str, int],
) -> None:
    encoded = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        for value in scores
    ).encode("utf-8")
    with journal.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        durable_bytes = stream.tell()
    state.update(
        {
            "state": "mining",
            "durable_scores": int(state["durable_scores"]) + len(scores),
            "journal_bytes": durable_bytes,
            "source_rows_observed": observed["rows"],
            "eligible_rows_observed": observed["eligible"],
            "updated_at": _utc_now(),
        }
    )
    _atomic_json(state_path, state)


def mine_m7_pathology(
    *,
    patch_manifests: list[str | Path] | tuple[str | Path, ...],
    output_path: str | Path,
    m7_checkpoint: str | Path,
    options: M7PathologyMiningOptions,
    corpus_plan: str | Path | None = None,
) -> dict[str, Any]:
    """Score every baseline-missing hold-in patch with released local m7.

    The score journal is append-only and crash-safe.  A batch is durable only
    after both its bytes and the state counters are fsynced; an interrupted
    suffix is truncated to the last state-bound byte offset on resume.
    """

    options.validate()
    configure_cpu_budget(options.max_cpu_threads)
    manifests = tuple(Path(value).expanduser().resolve() for value in patch_manifests)
    if not manifests or len(set(manifests)) != len(manifests):
        raise ValueError("patch manifests must be non-empty and unique")
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    checkpoint = Path(m7_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    plan = Path(corpus_plan).expanduser().resolve() if corpus_plan else None
    if plan is not None and not plan.is_file():
        raise FileNotFoundError(plan)
    output = Path(output_path).expanduser().resolve()
    identity = _mining_identity(
        patch_manifests=manifests,
        checkpoint=checkpoint,
        corpus_plan=plan,
        options=options,
    )
    identity_sha256 = _canonical_sha256(identity)
    state_path = output / "state.json"
    journal = output / "scores.jsonl"
    if output.exists():
        if not state_path.is_file() or not journal.is_file():
            raise ValueError(f"{output}: incomplete pathology mining output")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema") != M7_PATHOLOGY_STATE_SCHEMA:
            raise ValueError("unsupported pathology mining state schema")
        if state.get("identity") != identity:
            raise ValueError("pathology mining identity changed")
    else:
        output.mkdir(parents=True)
        journal.touch()
        state = {
            "schema": M7_PATHOLOGY_STATE_SCHEMA,
            "state": "mining",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "durable_scores": 0,
            "journal_bytes": 0,
            "source_rows_observed": 0,
            "eligible_rows_observed": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        _atomic_json(state_path, state)
    existing = _load_durable_scores(journal, state)
    predictor: M7LocalPatchPredictor | None = None
    batch: list[tuple[Path, int, dict[str, Any], np.ndarray, np.ndarray]] = []
    observed = {"rows": 0, "eligible": 0}
    scored_this_run = 0

    def flush_batch() -> None:
        nonlocal predictor, scored_this_run
        if not batch:
            return
        if predictor is None:
            predictor = M7LocalPatchPredictor(
                checkpoint,
                device=torch.device(options.device),
                amp_dtype=options.amp_dtype,
                threshold=options.threshold,
            )
        predictions = predictor.predict([item[3] for item in batch])
        scores: list[dict[str, Any]] = []
        for (manifest, row_index, row, _image, target), prediction in zip(
            batch,
            predictions,
            strict=True,
        ):
            score = score_m7_prediction(target, prediction)
            scores.append(
                {
                    "schema": M7_PATHOLOGY_SCORE_SCHEMA,
                    "mining_identity_sha256": identity_sha256,
                    "patch_id": str(row["patch_id"]),
                    "archive_sha256": str(row["archive_sha256"]),
                    "record_id": str(row["record_id"]),
                    "scroll_id": str(row["scroll_id"]),
                    "split": str(row["split"]),
                    "source_manifest": str(manifest),
                    "source_row_index": row_index,
                    **score,
                }
            )
        _append_scores(
            journal=journal,
            state_path=state_path,
            state=state,
            scores=scores,
            observed=observed,
        )
        for score in scores:
            existing[str(score["patch_id"])] = score
        scored_this_run += len(scores)
        print(
            f"m7 pathology {len(existing):,}: "
            f"+{scored_this_run:,} this run, "
            f"last={scores[-1]['patch_id']}",
            flush=True,
        )
        batch.clear()

    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as stream:
            for row_index, line in enumerate(stream):
                observed["rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{manifest}:{row_index + 1}: invalid patch row"
                    ) from error
                if row.get("schema") != "crossres-voxel-patch-v1":
                    raise ValueError(f"{manifest}: unsupported patch schema")
                if str(row.get("split")) not in options.splits:
                    continue
                if bool(row.get("has_baseline")):
                    continue
                observed["eligible"] += 1
                patch_id = str(row.get("patch_id", ""))
                archive_sha256 = str(row.get("archive_sha256", ""))
                if not patch_id or len(archive_sha256) != 64:
                    raise ValueError(f"{manifest}: invalid patch identity")
                prior = existing.get(patch_id)
                if prior is not None:
                    if (
                        prior.get("archive_sha256") != archive_sha256
                        or prior.get("record_id") != row.get("record_id")
                        or prior.get("split") != row.get("split")
                    ):
                        raise ValueError(
                            f"pathology source identity changed for {patch_id}"
                        )
                    continue
                if (
                    options.max_patches is not None
                    and scored_this_run + len(batch) >= options.max_patches
                ):
                    continue
                image, target = _load_patch_arrays(manifest, row)
                batch.append((manifest, row_index, row, image, target))
                if len(batch) >= options.batch_size:
                    flush_batch()
    flush_batch()
    state["source_rows_observed"] = observed["rows"]
    state["eligible_rows_observed"] = observed["eligible"]
    state["updated_at"] = _utc_now()
    if options.expected_count is not None:
        if observed["eligible"] > options.expected_count:
            raise ValueError(
                f"observed {observed['eligible']:,} eligible rows, expected at most "
                f"{options.expected_count:,}"
            )
        if len(existing) == options.expected_count:
            if observed["eligible"] != options.expected_count:
                raise ValueError("complete score count lacks matching source rows")
            state["state"] = "complete"
            state["completed_at"] = _utc_now()
            state["source_manifest_sha256"] = {
                str(path): _sha256(path) for path in manifests
            }
        else:
            state["state"] = "mining"
    _atomic_json(state_path, state)
    if int(state["durable_scores"]) != len(existing):
        raise RuntimeError("durable pathology score count drifted")
    return {
        "output": str(output),
        "state": state["state"],
        "durable_scores": len(existing),
        "scored_this_run": scored_this_run,
        "source_rows_observed": observed["rows"],
        "eligible_rows_observed": observed["eligible"],
        "expected_count": options.expected_count,
        "identity_sha256": identity_sha256,
    }
