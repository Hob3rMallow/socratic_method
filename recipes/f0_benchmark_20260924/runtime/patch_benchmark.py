#!/usr/bin/env python
"""Frozen-benchmark comparison harness: released M7, F0 (and exploratory variants), HercUNet v0.

Runs ONE model family over the 689 validation rows of the frozen v14p2 manifest and records, per row and
per configuration, exact threshold counts, plus (optionally) the uint8 probability map for later CPU metrics.
Every model sees the same 192^3 patch image; every count is taken against the same registered fine-teacher
target with the ignore label 2 excluded, exactly as ``crossres-voxel audit-checkpoint`` counts it.

Counts recorded per configuration and row, at the audit's 17 thresholds (0.10 .. 0.90):
  strict     TP / FP / FN over known voxels (the audit's ThresholdCounts; macro Dice = mean of the two
             per-scroll pooled Dice values)
  tolerant   the audit's asymmetric surface precision/recall (TolerantThresholdCounts) at tolerance 2 -
             reproduced exactly - and, for fairness to thin/offset predictions, also at tolerances 1, 3, 4
  dilated    TP / FP / FN after dilating the thresholded prediction by r = 1, 2, 3 voxels (6-connected),
             the remedy HercUNet's write-up proposes for its medial target

Families
  engine    one crossres voxel checkpoint (released M7 in the student container, F0, soups). Needs
            PYTHONPATH to include socratic_method/src. Eight mirror forwards per row, batch 1, bf16 autocast,
            the flip order and arithmetic of crossres_pred.voxel.inference._predict_probability. Configs:
              <label>_flat      identity forward only (the no-TTA audit)
              <label>_tta       mean softmax over the eight flips (the shipped F0 recipe / audit --tta)
              <label>_ttalogit  softmax of the mean logits (exploratory; HercUNet's TTA arithmetic)
  hercunet  HercUNet v0 through its own inference functions (hercunet.infer.*). In-window Jacobi recurrence,
            which is also its DAgger training recurrence: pass 0 reads prev = 0, pass p reads the uint8
            quantised probability of pass p-1 divided by 255, exactly the buffer semantics of
            hercunet.infer.jacobi_refine. A 192^3 patch is exactly one of its windows, so the Gaussian
            blend reduces to sigmoid(l1 - l0). Configs: <label>_p0 .. <label>_p{passes-1}.
            --norm shipped     their inference normalisation (raw CT, dataset_fingerprint statistics)
            --norm trainmatch  their training normalisation (per-window 1st-99th percentile stretch to
                               uint8, then the plans' CTNormalization) - what the network was trained on
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import queue
import sys
import threading
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THRESHOLDS = np.asarray([float(v) for v in np.linspace(0.10, 0.90, 17)], dtype=np.float32)
TOLERANCES = (1, 2, 3, 4)
DILATIONS = (1, 2, 3)
SCHEMA = "socratic-f0-benchmark-patch-counts-v1"


# --------------------------------------------------------------------------------------------- rows
def load_rows(manifest: Path) -> list[dict]:
    base = manifest.parent
    rows = []
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            path = Path(row["path"])
            path = path if path.is_absolute() else base / path
            rows.append(
                {
                    "patch_id": row["patch_id"],
                    "scroll": row["scroll_id"],
                    "record": row["record_id"],
                    "path": str(path),
                    "cluster": row.get("support_anchor_chunk_zyx"),
                    "origin_zyx": row.get("origin_zyx"),
                    "source": row.get("supervision_source"),
                }
            )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def prefetch(rows: list[dict], depth: int = 3):
    """Yield (row, image_u8, target_u8) with background loading."""
    out: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        for row in rows:
            with np.load(row["path"], allow_pickle=False) as archive:
                image = np.asarray(archive["image"])
                target = np.asarray(archive["target_u8"])
            out.put((row, image, target))
        out.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = out.get()
        if item is None:
            return
        yield item


# ------------------------------------------------------------------------------------------ metrics
def cross_max(x: torch.Tensor, pad_value: float) -> torch.Tensor:
    """Max over the 6-connected neighbourhood plus self (scipy generate_binary_structure(3, 1))."""
    p = F.pad(x[None, None], (1, 1, 1, 1, 1, 1), value=pad_value)[0, 0]
    r = p[1:-1, 1:-1, 1:-1]
    r = torch.maximum(r, p[:-2, 1:-1, 1:-1])
    r = torch.maximum(r, p[2:, 1:-1, 1:-1])
    r = torch.maximum(r, p[1:-1, :-2, 1:-1])
    r = torch.maximum(r, p[1:-1, 2:, 1:-1])
    r = torch.maximum(r, p[1:-1, 1:-1, :-2])
    r = torch.maximum(r, p[1:-1, 1:-1, 2:])
    return r


class RowScorer:
    """Exact per-row counts against one target (ignore label 2 excluded)."""

    def __init__(self, target: np.ndarray, device: torch.device) -> None:
        t = torch.from_numpy(target.astype(np.int16)).to(device)
        self.valid = t != 2
        self.truth = (t == 1) & self.valid
        self.thr = torch.from_numpy(THRESHOLDS).to(device)[:, None]
        self.truth_v = self.truth[self.valid]
        self.known = int(self.valid.sum())
        self.positive = int(self.truth.sum())
        # scipy binary_dilation(truth, cross, iterations=k, mask=valid): growth only through valid voxels
        near = self.truth.float()
        validf = self.valid.float()
        self.near_v = {}
        for k in range(1, max(TOLERANCES) + 1):
            near = torch.maximum(near, cross_max(near, 0.0) * validf)
            if k in TOLERANCES:
                self.near_v[k] = (near > 0.5)[self.valid]

    def _strict(self, field: torch.Tensor) -> dict:
        scores = field[self.valid]
        pred = scores[None, :] >= self.thr
        truth = self.truth_v[None, :]
        return {
            "tp": (pred & truth).sum(1).tolist(),
            "fp": (pred & ~truth).sum(1).tolist(),
            "fn": (~pred & truth).sum(1).tolist(),
        }

    def measure(self, probability: torch.Tensor) -> dict:
        prob = probability.float()
        result = {"known": self.known, "positive": self.positive, "strict": self._strict(prob)}
        scores = prob[self.valid]
        pred = scores[None, :] >= self.thr
        predicted = pred.sum(1)
        # audit TolerantThresholdCounts: recovered = iterated cross max of where(valid, p, -1), cval -1
        recovered = torch.where(self.valid, prob, torch.full_like(prob, -1.0))
        tolerant = {}
        for k in range(1, max(TOLERANCES) + 1):
            recovered = cross_max(recovered, -1.0)
            if k in TOLERANCES:
                recall_scores = recovered[self.truth]
                tolerant[str(k)] = {
                    "matched_prediction": (pred & self.near_v[k][None, :]).sum(1).tolist(),
                    "predicted": predicted.tolist(),
                    "matched_truth": (recall_scores[None, :] >= self.thr).sum(1).tolist(),
                    "truth": int(recall_scores.numel()),
                }
        result["tolerant"] = tolerant
        dilated = {}
        field = prob
        for r in range(1, max(DILATIONS) + 1):
            field = cross_max(field, 0.0)  # max-filter of p == dilation of (p >= t) for every t
            if r in DILATIONS:
                dilated[str(r)] = self._strict(field)
        result["dilated"] = dilated
        return result


# --------------------------------------------------------------------------------------- families
class EngineFamily:
    def __init__(self, checkpoint: Path, label: str, device: torch.device) -> None:
        from crossres_pred.voxel.inference import load_voxel_checkpoint
        from crossres_pred.voxel.patches import normalize_m7_ct
        from crossres_pred.voxel.resources import assert_cuda_power_limit

        assert_cuda_power_limit(device)
        self.model, payload = load_voxel_checkpoint(checkpoint, device=device)
        self.model.eval()
        self.normalize = normalize_m7_ct
        self.device = device
        self.label = label
        self.identity = {
            "family": "engine",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "amp_dtype": "bfloat16",
            "flip_order": "itertools.product((False, True), repeat=3) over (z, y, x)",
            "epoch": payload.get("epoch"),
        }
        self.configs = [f"{label}_flat", f"{label}_tta", f"{label}_ttalogit"]

    @torch.no_grad()
    def predict(self, image_u8: np.ndarray) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.normalize(image_u8))[None, None].to(self.device)
        prob_total = None
        logit_total = None
        flat = None
        for flips in product((False, True), repeat=3):
            dims = tuple(index + 2 for index, enabled in enumerate(flips) if enabled)
            value = torch.flip(image, dims) if dims else image
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = self.model.full_resolution_logits(value)
                probability = torch.softmax(logits, dim=1)[:, 1]
            if dims:
                probability = torch.flip(probability, tuple(d - 1 for d in dims))
                logits = torch.flip(logits, dims)
            if flat is None:
                flat = probability.float().clone()
            prob_total = probability.float() if prob_total is None else prob_total + probability.float()
            logit_total = logits.float() if logit_total is None else logit_total + logits.float()
        tta = prob_total / 8
        ttalogit = torch.softmax(logit_total / 8, dim=1)[:, 1]
        return {
            f"{self.label}_flat": flat[0],
            f"{self.label}_tta": tta[0],
            f"{self.label}_ttalogit": ttalogit[0],
        }


class HercUNetFamily:
    def __init__(self, model_dir: Path, label: str, device: torch.device, norm: str, passes: int, tta: str) -> None:
        from hercunet.infer import jacobi_refine as JR
        from hercunet.infer import nnunet_infer as NI

        self.JR, self.NI = JR, NI
        self.net, cfg, _lm, pm, _dsj, num_in = NI.load_affinity_net(str(model_dir), "checkpoint_best.pth", "cuda", n_orient=6)
        if num_in != 8 or list(cfg.patch_size) != [192, 192, 192]:
            raise SystemExit(f"unexpected HercUNet geometry: in={num_in} patch={cfg.patch_size}")
        self.device = device
        self.label = label
        self.norm = norm
        self.passes = passes
        self.variants = JR._resolve_tta(tta)
        self.fingerprint_norm = NI.ct_norm_params(str(model_dir))
        plans = json.loads((model_dir / "plans.json").read_text(encoding="utf-8"))
        props = plans["foreground_intensity_properties_per_channel"]["0"]
        self.plans_norm = (
            float(props["percentile_00_5"]),
            float(props["percentile_99_5"]),
            float(props["mean"]),
            float(props["std"]),
        )
        checkpoint = model_dir / "fold_0" / "checkpoint_best.pth"
        self.identity = {
            "family": "hercunet",
            "model_dir": str(model_dir),
            "checkpoint_sha256": sha256(checkpoint),
            "norm": norm,
            "fingerprint_norm_lo_hi_mean_std": self.fingerprint_norm,
            "plans_norm_lo_hi_mean_std": self.plans_norm,
            "passes": passes,
            "tta": tta,
            "tta_variants": len(self.variants),
            "autocast": "float16 (as hercunet.infer.jacobi_refine)",
            "recurrence": "in-window Jacobi; prev_p = round(255 * prob_{p-1}) / 255; pass 0 prev = 0",
            "blend": "single 192^3 window: sigmoid(clamp(l1 - l0, -30, 30)) == their Gaussian-blend output",
        }
        self.configs = [f"{label}_p{p}" for p in range(passes)]

    def normalize(self, image_u8: np.ndarray) -> np.ndarray:
        if self.norm == "shipped":
            lo, hi, mean, std = self.fingerprint_norm
            return self.NI.norm_ct(image_u8, lo, hi, mean, std)
        if self.norm == "trainmatch":
            g = np.asarray(image_u8, np.float32)
            lo_p, hi_p = np.percentile(g, 1), np.percentile(g, 99)
            stretched = np.clip((g - lo_p) / max(hi_p - lo_p, 1e-3), 0.0, 1.0)
            ct8 = (stretched * 255.0).round().astype(np.uint8)
            lo, hi, mean, std = self.plans_norm
            return self.NI.norm_ct(ct8, lo, hi, mean, std)
        raise SystemExit(f"unknown --norm {self.norm}")

    @torch.no_grad()
    def predict(self, image_u8: np.ndarray) -> dict[str, torch.Tensor]:
        ct = torch.from_numpy(self.normalize(image_u8).astype(np.float32)).to(self.device)
        prev = torch.zeros_like(ct)
        out = {}
        for p in range(self.passes):
            xb = torch.stack([ct, prev])[None]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = self.JR._tta_seg_logits(self.net, xb, True, self.variants)
                diff = (logits[:, 1] - logits[:, 0]).float()
            probability = torch.sigmoid(diff.clamp(-30.0, 30.0))[0]
            u8 = probability.mul(255.0).round().clamp(0, 255)
            quantised = u8 / 255.0
            out[f"{self.label}_p{p}"] = quantised
            prev = quantised
        return out


# ------------------------------------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=("engine", "hercunet"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path, help="engine: voxel checkpoint")
    parser.add_argument("--model-dir", type=Path, help="hercunet: model folder (plans.json, fold_0/)")
    parser.add_argument("--norm", default="shipped", choices=("shipped", "trainmatch"))
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--tta", default="none", help="hercunet: their --tta grammar (none, mirror, all, ...)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--store", nargs="*", default=[], help="configs whose uint8 probability maps are kept")
    parser.add_argument("--limit", type=int, default=0, help="pilot: first N rows only")
    parser.add_argument("--every", type=int, default=1, help="pilot: every Nth row")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scroll", default=None, help="only validation rows of this scroll (e.g. PHerc0500P2)")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    rows = load_rows(args.manifest)
    if args.scroll:
        rows = [row for row in rows if row["scroll"] == args.scroll]
    if args.every > 1:
        rows = rows[:: args.every]
    if args.limit:
        rows = rows[: args.limit]
    run_dir = args.work / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.family == "engine":
        family = EngineFamily(args.checkpoint, args.label, device)
    else:
        family = HercUNetFamily(args.model_dir, args.label, device, args.norm, args.passes, args.tta)
    unknown = sorted(set(args.store) - set(family.configs))
    if unknown:
        raise SystemExit(f"--store names unknown configs {unknown}; available {family.configs}")
    identity = {
        "schema": SCHEMA,
        "label": args.label,
        "configs": family.configs,
        "model": family.identity,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "rows_requested": len(rows),
        "thresholds": THRESHOLDS.tolist(),
        "tolerances": list(TOLERANCES),
        "dilations": list(DILATIONS),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }
    identity_path = run_dir / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        for key in ("configs", "manifest_sha256", "thresholds"):
            if previous.get(key) != identity.get(key):
                raise SystemExit(f"resume identity mismatch on {key}")
        if previous["model"].get("checkpoint_sha256") != identity["model"].get("checkpoint_sha256"):
            raise SystemExit("resume identity mismatch on checkpoint")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    counts_path = run_dir / "rows.jsonl"
    done = set()
    if counts_path.exists():
        with counts_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(json.loads(line)["patch_id"])
    todo = [row for row in rows if row["patch_id"] not in done]
    print(f"[bench] {args.label}: {len(rows)} rows, {len(done)} already done, {len(todo)} to run on "
          f"{identity['device']}", flush=True)
    for config in args.store:
        (run_dir / "probs" / config).mkdir(parents=True, exist_ok=True)
    writer = futures.ThreadPoolExecutor(max_workers=6)
    pending: list[futures.Future] = []

    def store(path: Path, array: np.ndarray) -> None:
        temporary = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(temporary, p8=array)
        os.replace(temporary, path)

    start = time.time()
    with counts_path.open("a", encoding="utf-8") as sink:
        for index, (row, image, target) in enumerate(prefetch(todo), 1):
            if image.shape != (192, 192, 192) or target.shape != image.shape:
                raise SystemExit(f"{row['patch_id']}: unexpected shape {image.shape}")
            scorer = RowScorer(target, device)
            fields = family.predict(image)
            record = {"patch_id": row["patch_id"], "scroll": row["scroll"], "record": row["record"],
                      "cluster": row["cluster"], "configs": {}}
            for config, field in fields.items():
                record["configs"][config] = scorer.measure(field)
                if config in args.store:
                    p8 = field.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    pending.append(writer.submit(store, run_dir / "probs" / config / f"{row['patch_id']}.npz", p8))
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if index % 10 == 0 or index == len(todo):
                rate = index / max(time.time() - start, 1e-9)
                print(f"[bench] {args.label}: {index}/{len(todo)} rows {rate * 60:.1f} rows/min "
                      f"eta {(len(todo) - index) / max(rate, 1e-9) / 60:.1f} min", flush=True)
            pending = [f for f in pending if not f.done() or f.result() is not None]
    for job in pending:
        job.result()
    writer.shutdown()
    print(f"[bench] {args.label}: DONE in {(time.time() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
