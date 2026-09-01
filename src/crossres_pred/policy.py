from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from .schema import (
    PairRecord,
    canonical_scroll_id,
    ensure_scroll_disjoint,
    normalized_split,
)


class PolicyError(ValueError):
    """A manifest attempted to cross a prize-safety boundary."""


@dataclass(frozen=True)
class DataPolicy:
    profile: str
    forbidden_fine_scrolls: frozenset[str]
    allowed_scrolls: frozenset[str]
    split_scrolls: dict[str, frozenset[str]]
    source: Path | None = None

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, source: Path | None = None
    ) -> DataPolicy:
        try:
            version = int(value.get("schema_version", 1))
        except (TypeError, ValueError) as error:
            raise PolicyError("schema_version must be an integer") from error
        if version != 1:
            raise PolicyError(f"unsupported policy schema_version {version}")
        profile = str(value.get("profile", "")).strip()
        if profile not in {"prize-safe", "research"}:
            raise PolicyError("profile must be 'prize-safe' or 'research'")
        raw_policy = value.get("policy")
        if not isinstance(raw_policy, dict):
            raise PolicyError("configuration is missing [policy]")
        raw_forbidden = raw_policy.get("forbid_fine_for_scrolls", [])
        raw_allowed = raw_policy.get("allow_scrolls", [])
        if not isinstance(raw_forbidden, list) or not isinstance(raw_allowed, list):
            raise PolicyError("policy scroll lists must be arrays")
        forbidden = frozenset(canonical_scroll_id(str(item)) for item in raw_forbidden)
        allowed = frozenset(canonical_scroll_id(str(item)) for item in raw_allowed)
        raw_splits = value.get("splits", {})
        if not isinstance(raw_splits, dict):
            raise PolicyError("[splits] must be a table")
        splits: dict[str, frozenset[str]] = {}
        for name, items in raw_splits.items():
            if not isinstance(items, list):
                raise PolicyError(f"split {name!r} must be an array")
            splits[str(name)] = frozenset(
                canonical_scroll_id(str(item)) for item in items
            )
        memberships: dict[str, list[str]] = {}
        for split, scrolls in splits.items():
            for scroll in scrolls:
                memberships.setdefault(scroll, []).append(split)
        overlap = {
            scroll: names for scroll, names in memberships.items() if len(names) > 1
        }
        if overlap:
            detail = ", ".join(
                f"{scroll}={sorted(names)}" for scroll, names in sorted(overlap.items())
            )
            raise PolicyError(f"policy splits overlap: {detail}")
        if profile == "prize-safe" and "PHerc1203" not in forbidden:
            raise PolicyError(
                "prize-safe profile must forbid PHerc1203 higher-resolution data"
            )
        return cls(
            profile=profile,
            forbidden_fine_scrolls=forbidden,
            allowed_scrolls=allowed,
            split_scrolls=splits,
            source=source,
        )

    @classmethod
    def load(cls, path: str | Path) -> DataPolicy:
        source = Path(path).resolve()
        with source.open("rb") as stream:
            value = tomllib.load(stream)
        return cls.from_dict(value, source=source)

    def validate_record(self, record: PairRecord) -> None:
        if self.allowed_scrolls and record.scroll_id not in self.allowed_scrolls:
            raise PolicyError(
                f"{record.record_id}: {record.scroll_id} is not allowed by "
                f"{self.profile} policy"
            )
        if record.scroll_id in self.forbidden_fine_scrolls:
            raise PolicyError(
                f"{record.record_id}: fine scan {record.fine.scan_id!r} for "
                f"{record.scroll_id} is sealed by {self.profile} policy"
            )
        expected: str | None = None
        for split_name, scrolls in self.split_scrolls.items():
            if record.scroll_id in scrolls:
                expected = normalized_split(split_name)
                break
        if self.split_scrolls and expected is None:
            raise PolicyError(
                f"{record.record_id}: {record.scroll_id} has no policy split assignment"
            )
        record_split = normalized_split(record.split)
        if expected and expected != record_split:
            raise PolicyError(
                f"{record.record_id}: split is {record.split!r}, but policy assigns "
                f"{record.scroll_id} to {expected!r}"
            )

    def validate_records(self, records: Iterable[PairRecord]) -> None:
        materialized = list(records)
        ensure_scroll_disjoint(materialized)
        for record in materialized:
            self.validate_record(record)
