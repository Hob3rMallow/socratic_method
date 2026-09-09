"""Uniform (or weighted) state-dict averages of sample-milestone checkpoints.

The student is an InstanceNorm network without batch statistics, so a plain
parameter average is a well-defined single network. The averaged payload keeps
the milestone checkpoint surface (``model_config`` + ``model`` + sample
counters) so ``load_voxel_checkpoint``, the frozen audit, blind reports and
grid inference accept it unchanged. ``requested_samples``/``cumulative_samples``
are the maximum over members: no exposure beyond that count contributed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

AVERAGE_CONTRACT = "uniform-state-dict-average-fp32-v1"
AVERAGE_CHECKPOINT_KIND = "milestone-average"
MEMBER_CHECKPOINT_KIND = "sample-milestone"
RECEIPT_SCHEMA = "crossres-voxel-checkpoint-average-v1"
DEFAULT_ALLOWED_IDENTITY_KEYS = ("options.seed",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_paths(value: Any, dotted: Iterable[str]) -> Any:
    stripped = copy.deepcopy(value)
    for path in dotted:
        parts = path.split(".")
        node = stripped
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return stripped


def load_member(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: checkpoint payload must be an object")
    if payload.get("checkpoint_kind") != MEMBER_CHECKPOINT_KIND:
        raise ValueError(
            f"{path}: only {MEMBER_CHECKPOINT_KIND!r} checkpoints may be averaged"
        )
    for key in ("model_config", "model", "requested_samples", "cumulative_samples"):
        if key not in payload:
            raise ValueError(f"{path}: checkpoint lacks {key!r}")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise TypeError(f"{path}: checkpoint model state is empty")
    return payload


def normalized_weights(count: int, weights: Sequence[float] | None) -> list[float]:
    if count <= 0:
        raise ValueError("an average needs at least one member")
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("one weight per member is required")
    values = [float(value) for value in weights]
    if any(value <= 0 for value in values):
        raise ValueError("weights must be positive")
    total = sum(values)
    return [value / total for value in values]


class _Accumulator:
    def __init__(self, allowed_identity_keys: Sequence[str]) -> None:
        self.allowed = tuple(allowed_identity_keys)
        self.first: dict[str, Any] | None = None
        self.sums: dict[str, torch.Tensor] = {}
        self.dtypes: dict[str, torch.dtype] = {}
        self.constants: dict[str, torch.Tensor] = {}
        self.identities: list[Any] = []
        self.members: list[dict[str, Any]] = []

    def add(self, path: Path, payload: dict[str, Any], weight: float) -> None:
        state = payload["model"]
        if self.first is None:
            self.first = payload
            for key, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{path}: model state {key!r} is not a tensor")
                self.dtypes[key] = tensor.dtype
        else:
            for key in ("model_config", "snapshot_checkpoint_contract", "initialization"):
                if payload.get(key) != self.first.get(key):
                    raise ValueError(f"{path}: {key} differs between members")
            if set(state) != set(self.first["model"]):
                raise ValueError(f"{path}: model state keys differ between members")
            first_identity = _strip_paths(self.first.get("identity"), self.allowed)
            if _strip_paths(payload.get("identity"), self.allowed) != first_identity:
                raise ValueError(
                    f"{path}: run identity differs beyond the allowed keys "
                    f"{list(self.allowed)}"
                )
        for key in sorted(state):
            tensor = state[key]
            reference = self.first["model"][key]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(f"{path}: model state {key!r} shape/dtype differs")
            if tensor.is_floating_point():
                contribution = tensor.detach().to(torch.float32) * weight
                if key in self.sums:
                    self.sums[key] = self.sums[key] + contribution
                else:
                    self.sums[key] = contribution
            else:
                if key in self.constants:
                    if not torch.equal(self.constants[key], tensor):
                        raise ValueError(
                            f"{path}: non-floating state {key!r} differs between members"
                        )
                else:
                    self.constants[key] = tensor.detach().clone()
        self.identities.append(payload.get("identity"))
        self.members.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "requested_samples": int(payload["requested_samples"]),
                "cumulative_samples": int(payload["cumulative_samples"]),
                "epoch": payload.get("epoch"),
                "weight": weight,
            }
        )

    def payload(self) -> dict[str, Any]:
        if self.first is None:
            raise ValueError("no members were added")
        averaged: dict[str, torch.Tensor] = {}
        for key in sorted(self.first["model"]):
            if key in self.sums:
                averaged[key] = self.sums[key].to(self.dtypes[key]).contiguous()
            else:
                averaged[key] = self.constants[key].contiguous()
        identical = all(identity == self.identities[0] for identity in self.identities)
        identity = (
            self.identities[0]
            if identical
            else _strip_paths(self.identities[0], self.allowed)
        )
        requested = max(member["requested_samples"] for member in self.members)
        cumulative = max(member["cumulative_samples"] for member in self.members)
        epochs = [member["epoch"] for member in self.members if member["epoch"] is not None]
        epoch = max(int(value) for value in epochs) if epochs else None
        result: dict[str, Any] = {
            "checkpoint_kind": AVERAGE_CHECKPOINT_KIND,
            "average_contract": AVERAGE_CONTRACT,
            "snapshot_checkpoint_contract": self.first.get("snapshot_checkpoint_contract"),
            "requested_samples": requested,
            "cumulative_samples": cumulative,
            "epoch": epoch,
            "best_score": None,
            "best_trained_score": None,
            "model_config": copy.deepcopy(self.first["model_config"]),
            "model": averaged,
            "initialization": copy.deepcopy(self.first.get("initialization")),
            "identity": identity,
            "average_of": copy.deepcopy(self.members),
            "source_sha256": {member["path"]: member["sha256"] for member in self.members},
            "metrics": {
                "epoch": epoch,
                "train": {"cumulative_samples": float(cumulative)},
                "val": {},
            },
        }
        if not identical:
            result["member_identities"] = copy.deepcopy(self.identities)
            result["identity_differs_in"] = list(self.allowed)
        return result


def average_checkpoints(
    member_paths: Sequence[Path],
    weights: Sequence[float] | None = None,
    *,
    allowed_identity_keys: Sequence[str] = DEFAULT_ALLOWED_IDENTITY_KEYS,
) -> dict[str, Any]:
    """Stream the members one at a time into a float32 average."""
    paths = [Path(path).expanduser().resolve() for path in member_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("member checkpoints must be unique")
    normalized = normalized_weights(len(paths), weights)
    accumulator = _Accumulator(allowed_identity_keys)
    for path, weight in zip(paths, normalized, strict=True):
        if not path.is_file():
            raise FileNotFoundError(path)
        accumulator.add(path, load_member(path), weight)
    return accumulator.payload()


def receipt_for(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "checkpoint_kind": payload["checkpoint_kind"],
        "average_contract": payload["average_contract"],
        "requested_samples": payload["requested_samples"],
        "cumulative_samples": payload["cumulative_samples"],
        "members": payload["average_of"],
        "model_config": payload["model_config"],
        "created_utc": datetime.now(UTC).isoformat(),
    }


def write_average_checkpoint(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    receipt = receipt_for(payload, output)
    receipt_path = output.with_name(output.name + ".json")
    receipt_temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    receipt_temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(receipt_temporary, receipt_path)
    return receipt


def members_match_receipt(receipt: dict[str, Any], member_paths: Sequence[Path]) -> bool:
    recorded = [(m["path"], m["sha256"]) for m in receipt.get("members", [])]
    actual = [
        (str(Path(p).expanduser().resolve()), sha256_file(Path(p)))
        for p in member_paths
    ]
    return recorded == actual
