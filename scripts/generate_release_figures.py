from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import distance_transform_edt, maximum_filter

BACKGROUND = (3, 7, 18)
PANEL = (8, 13, 24)
PANEL_ALT = (12, 19, 34)
GRID = (36, 48, 68)
TEXT = (248, 250, 252)
MUTED = (148, 163, 184)
M7_CYAN = (55, 220, 255)
TEACHER_YELLOW = (250, 204, 21)
STUDENT_PINK = (255, 55, 190)
SHARED_GREEN = (80, 255, 130)
MEDIAL_PURPLE = (167, 139, 250)
PIN_ORANGE = (251, 146, 60)

REPORT_RELATIVE = Path(
    "output/crossres_data/"
    "m7_xr_v31_duration_ladder_human_report_20260831"
)
FLIP_RELATIVE = Path(
    "output/crossres_data/"
    "m7_xr_v31_duration_ladder_flip_audit_20260901/audit.json"
)
CONNECTIVITY_RELATIVE = Path(
    "output/crossres_data/"
    "pherc0139_training_dynamic_medial_connectivity_v28b_fullowner_20260831"
)
ATLAS_RELATIVE = Path(
    "output/crossres_data/coarse_teacher_atlas_v16_250k/pherc0139"
)
CT_RELATIVE = Path(
    "PHerc0139-full/"
    "20250728140407-9.362um-1.2m-113keV-masked.zarr"
)
M7_RELATIVE = Path(
    "PHerc0139-full/"
    "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
LINE_FITTER_RELATIVE = Path("output/fixup_4x5x5/report_assets")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [Path(r"C:\Windows\Fonts\seguisb.ttf"), Path("DejaVuSans-Bold.ttf")]
        if bold
        else [Path(r"C:\Windows\Fonts\segoeui.ttf"), Path("DejaVuSans.ttf")]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    raise RuntimeError("no usable TrueType font found")


FONT_22 = _font(22)
FONT_24 = _font(24)
FONT_26 = _font(26)
FONT_28 = _font(28)
FONT_30 = _font(30)
FONT_32 = _font(32, bold=True)
FONT_36 = _font(36, bold=True)
FONT_42 = _font(42, bold=True)


@dataclass
class SourceTracker:
    root: Path
    files: set[Path] = field(default_factory=set)
    zarr_sources: list[dict[str, Any]] = field(default_factory=list)

    def record(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.files.add(resolved)
        return resolved

    def image(self, path: Path, mode: str) -> Image.Image:
        return Image.open(self.record(path)).convert(mode)

    def json(self, path: Path) -> dict[str, Any]:
        return _json(self.record(path))

    def zarr(self, path: Path, *, identity: Path, description: str) -> Any:
        try:
            import zarr
        except ImportError as error:  # pragma: no cover - environment guard
            raise RuntimeError("install the zarr extra to generate method figures") from error
        identity = self.record(identity)
        store = path.resolve()
        self.zarr_sources.append(
            {
                "store": str(store),
                "identity_file": str(identity),
                "identity_sha256": _sha256(identity),
                "description": description,
            }
        )
        value = zarr.open(store, mode="r")
        if hasattr(value, "array_keys"):
            value = value["0"]
        return value

    def manifest_inputs(self) -> list[dict[str, Any]]:
        values = []
        for path in sorted(self.files, key=lambda item: str(item).lower()):
            try:
                relative = path.relative_to(self.root)
                display = str(relative).replace("\\", "/")
            except ValueError:
                display = str(path)
            values.append(
                {"path": display, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
        return values


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.autocontrast(image.convert("L")), dtype=np.uint8)


def _mask(image: Image.Image) -> np.ndarray:
    # Report masks use display-safe near-black/near-white values (5 and 242),
    # not literal 0 and 255. Threshold at the midpoint so the dark canvas is
    # never mistaken for foreground.
    return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def _plain_ct(ct: np.ndarray) -> Image.Image:
    return Image.fromarray(np.repeat(ct[..., None], 3, axis=-1), mode="RGB")


def _overlay(ct: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.repeat(ct[..., None], 3, axis=-1).astype(np.float32) * 0.32
    rgb[mask] = np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _probability_overlay(
    ct: np.ndarray, probability: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    rgb = np.repeat(ct[..., None], 3, axis=-1).astype(np.float32) * 0.28
    alpha = np.clip(probability.astype(np.float32), 0.0, 1.0)[..., None] * 0.92
    rgb = rgb * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _difference(
    ct: np.ndarray, student: np.ndarray, reference: np.ndarray
) -> Image.Image:
    rgb = np.repeat(ct[..., None], 3, axis=-1).astype(np.float32) * 0.18
    rgb[student & reference] = SHARED_GREEN
    rgb[~student & reference] = TEACHER_YELLOW
    rgb[student & ~reference] = STUDENT_PINK
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _multi_mask(
    ct: np.ndarray,
    layers: Iterable[tuple[np.ndarray, tuple[int, int, int]]],
) -> Image.Image:
    rgb = np.repeat(ct[..., None], 3, axis=-1).astype(np.float32) * 0.22
    for mask, color in layers:
        rgb[mask] = color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _new(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), BACKGROUND)


def _rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=GRID, width=2)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.NEAREST)


def _cell(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    image: Image.Image,
    *,
    label: str | None = None,
    detail: str | None = None,
    accent: tuple[int, int, int] | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    _rounded_panel(draw, box)
    x0, y0, x1, y1 = box
    text_height = 48 if label else 12
    if label:
        draw.text((x0 + 16, y0 + 10), label, fill=TEXT, font=FONT_26)
        if accent is not None:
            draw.rounded_rectangle(
                (x1 - 42, y0 + 15, x1 - 16, y0 + 39), radius=6, fill=accent
            )
    detail_height = 38 if detail else 10
    inner = (x0 + 10, y0 + text_height, x1 - 10, y1 - detail_height)
    canvas.paste(_fit_image(image, (inner[2] - inner[0], inner[3] - inner[1])), inner[:2])
    if detail:
        draw.text((x0 + 14, y1 - 31), detail, fill=MUTED, font=FONT_22)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int, entries: list[tuple[str, tuple[int, int, int]]]) -> None:
    cursor = x
    for label, color in entries:
        draw.rounded_rectangle((cursor, y + 3, cursor + 22, y + 25), radius=5, fill=color)
        cursor += 32
        draw.text((cursor, y), label, fill=MUTED, font=FONT_22)
        cursor += int(draw.textlength(label, font=FONT_22)) + 30


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True, dpi=(300, 300))


class ReportData:
    def __init__(self, tracker: SourceTracker, report_root: Path) -> None:
        self.tracker = tracker
        self.root = report_root.resolve()
        self.report = tracker.json(self.root / "report.json")
        self.locked_rows = {int(row["rank"]): row for row in self.report["locked16_rows"]}
        self.pherc_rows = list(self.report["pherc1447_fixed18_rows"])

    def locked(self, rank: int) -> dict[str, Any]:
        base = self.root / "assets" / "locked16" / f"rank_{rank:03d}"
        row = self.locked_rows[rank]
        model = next(item for item in row["models"] if int(item["samples"]) == 8192)
        return {
            "row": row,
            "ct": _gray(self.tracker.image(base / "ct.png", "L")),
            "m7": _mask(self.tracker.image(base / "m7.png", "L")),
            "teacher": _mask(self.tracker.image(base / "teacher.png", "L")),
            "student": _mask(
                self.tracker.image(base / "n008192" / "t0.45.png", "L")
            ),
            "detail": model["panels"]["0.45"]["detail"],
        }

    def pherc(self, coordinate: int, cube_id: str | None = None) -> dict[str, Any]:
        matches = [
            row
            for row in self.pherc_rows
            if int(row["global_coordinate"]) == coordinate
            and (cube_id is None or row["cube_id"] == cube_id)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one PHerc1447 view for {cube_id=} {coordinate=}")
        return self.pherc_row(matches[0])

    def pherc_row(self, row: dict[str, Any]) -> dict[str, Any]:
        model = next(item for item in row["models"] if int(item["samples"]) == 8192)
        source = Path(model["panels"]["0.45"]["src"])
        view = self.root / source.parent.parent
        return {
            "row": row,
            "ct": _gray(self.tracker.image(view / "ct.png", "L")),
            "m7": _mask(self.tracker.image(view / "m7.png", "L")),
            "reference": _mask(self.tracker.image(view / "v15.png", "L")),
            "student": _mask(
                self.tracker.image(view / "n008192" / "t0.45.png", "L")
            ),
            "detail": model["panels"]["0.45"]["detail"],
        }


def _teaser(data: ReportData, output: Path) -> None:
    blind = data.pherc(12032, "z12032_y03968_x03072")
    locked = data.locked(23)
    canvas = _new(2400, 1020)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (42, 22),
        "Blind anti-blob · PHerc1447 z=12032",
        fill=TEXT,
        font=FONT_36,
    )
    blind_panels = (
        ("coarse CT", _plain_ct(blind["ct"]), None),
        (
            "released M7",
            _overlay(blind["ct"], blind["m7"], M7_CYAN),
            M7_CYAN,
        ),
        (
            "raw M7-XR",
            _overlay(blind["ct"], blind["student"], STUDENT_PINK),
            STUDENT_PINK,
        ),
    )
    for index, (label, image, accent) in enumerate(blind_panels):
        x = 415 + index * 525
        _cell(canvas, (x, 72, x + 490, 430), image, label=label, accent=accent)

    draw.line((42, 458, 2358, 458), fill=GRID, width=3)
    draw.text(
        (42, 482),
        "Locked longitudinal growth · PHerc0139 rank 23",
        fill=TEXT,
        font=FONT_36,
    )
    locked_panels = (
        ("coarse CT", _plain_ct(locked["ct"]), None),
        (
            "released M7",
            _overlay(locked["ct"], locked["m7"], M7_CYAN),
            M7_CYAN,
        ),
        (
            "projected teacher",
            _overlay(locked["ct"], locked["teacher"], TEACHER_YELLOW),
            TEACHER_YELLOW,
        ),
        (
            "raw M7-XR",
            _overlay(locked["ct"], locked["student"], STUDENT_PINK),
            STUDENT_PINK,
        ),
    )
    for index, (label, image, accent) in enumerate(locked_panels):
        x = 190 + index * 510
        _cell(canvas, (x, 532, x + 480, 890), image, label=label, accent=accent)

    draw.text(
        (42, 944),
        "same raw checkpoint · 8,192 samples · T=0.45 · no blend",
        fill=MUTED,
        font=FONT_28,
    )
    _legend(
        draw,
        1310,
        942,
        [
            ("released M7", M7_CYAN),
            ("teacher", TEACHER_YELLOW),
            ("raw student", STUDENT_PINK),
        ],
    )
    _save(canvas, output / "teaser.png")


def _registered_examples(data: ReportData, output: Path) -> None:
    ranks = (23, 37, 45, 64)
    width, height = 2400, 1830
    canvas = _new(width, height)
    draw = ImageDraw.Draw(canvas)
    headers = ("coarse CT", "released M7", "projected teacher", "raw student", "teacher comparison")
    accents = (None, M7_CYAN, TEACHER_YELLOW, STUDENT_PINK, SHARED_GREEN)
    left, gap = 230, 18
    col_width = 412
    for index, (header, accent) in enumerate(zip(headers, accents, strict=True)):
        x = left + index * (col_width + gap)
        draw.text((x + 12, 36), header, fill=TEXT, font=FONT_30)
        if accent:
            draw.rounded_rectangle((x + col_width - 34, 40, x + col_width - 10, 64), radius=5, fill=accent)
    for row_index, rank in enumerate(ranks):
        item = data.locked(rank)
        y = 92 + row_index * 410
        draw.text((35, y + 62), "PHerc0139", fill=MUTED, font=FONT_24)
        draw.text((35, y + 98), f"rank {rank}", fill=TEXT, font=FONT_36)
        draw.text((35, y + 144), item["detail"], fill=MUTED, font=FONT_22)
        images = (
            _plain_ct(item["ct"]),
            _overlay(item["ct"], item["m7"], M7_CYAN),
            _overlay(item["ct"], item["teacher"], TEACHER_YELLOW),
            _overlay(item["ct"], item["student"], STUDENT_PINK),
            _difference(item["ct"], item["student"], item["teacher"]),
        )
        for column, image in enumerate(images):
            x = left + column * (col_width + gap)
            _cell(canvas, (x, y, x + col_width, y + 380), image)
    _legend(
        draw,
        250,
        1757,
        [
            ("shared", SHARED_GREEN),
            ("teacher only", TEACHER_YELLOW),
            ("student only", STUDENT_PINK),
        ],
    )
    draw.text((1440, 1757), "8,192 samples · T=0.45 · no blend", fill=MUTED, font=FONT_24)
    _save(canvas, output / "figure_2_registered_examples.png")


def _case_block(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    note: str,
    ct: np.ndarray,
    m7: np.ndarray,
    reference: np.ndarray,
    student: np.ndarray,
) -> None:
    draw = ImageDraw.Draw(canvas)
    _rounded_panel(draw, box)
    x0, y0, x1, y1 = box
    draw.text((x0 + 20, y0 + 16), title, fill=TEXT, font=FONT_32)
    draw.text((x0 + 20, y0 + 58), note, fill=MUTED, font=FONT_22)
    labels = ("CT", "released M7", "reference", "raw student", "comparison")
    accents = (None, M7_CYAN, TEACHER_YELLOW, STUDENT_PINK, SHARED_GREEN)
    images = (
        _plain_ct(ct),
        _overlay(ct, m7, M7_CYAN),
        _overlay(ct, reference, TEACHER_YELLOW),
        _overlay(ct, student, STUDENT_PINK),
        _difference(ct, student, reference),
    )
    gap = 12
    inner_width = x1 - x0 - 40
    cell_width = (inner_width - 4 * gap) // 5
    for index, (label, accent, image) in enumerate(zip(labels, accents, images, strict=True)):
        x = x0 + 20 + index * (cell_width + gap)
        _cell(canvas, (x, y0 + 96, x + cell_width, y1 - 20), image, label=label, accent=accent)


def _failure_cases(data: ReportData, output: Path) -> None:
    rank26 = data.locked(26)
    rank64 = data.locked(64)
    p12153 = data.pherc(12153, "z12032_y03968_x03072")
    p12256 = data.pherc(12256, "z12160_y04224_x02944")
    canvas = _new(2400, 1430)
    cases = [
        ((35, 35, 1185, 700), "Locked rank 26 · scalar topology exception", "intended structures recovered; reported 5 CC vs teacher 4", rank26["ct"], rank26["m7"], rank26["teacher"], rank26["student"]),
        ((1215, 35, 2365, 700), "Locked rank 64 · mild thinning / drift", "bottom-left strip remains thinner than desired", rank64["ct"], rank64["m7"], rank64["teacher"], rank64["student"]),
        ((35, 730, 1185, 1395), "Blind z=12153 · fragmented lower-left line", "v15 is a comparison model, not ground truth", p12153["ct"], p12153["m7"], p12153["reference"], p12153["student"]),
        ((1215, 730, 2365, 1395), "Blind z=12256 · residual undergrowth", "visible structure remains only partly recovered", p12256["ct"], p12256["m7"], p12256["reference"], p12256["student"]),
    ]
    for box, title, note, ct, m7, reference, student in cases:
        _case_block(
            canvas,
            box,
            title=title,
            note=note,
            ct=ct,
            m7=m7,
            reference=reference,
            student=student,
        )
    _save(canvas, output / "figure_4_failure_cases.png")


def _gallery_block(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    detail: str,
    images: tuple[tuple[str, Image.Image, tuple[int, int, int] | None], ...],
) -> None:
    draw = ImageDraw.Draw(canvas)
    _rounded_panel(draw, box)
    x0, y0, x1, y1 = box
    draw.text((x0 + 22, y0 + 18), title, fill=TEXT, font=FONT_32)
    draw.text((x0 + 22, y0 + 59), detail, fill=MUTED, font=FONT_22)
    gap = 14
    inner_width = x1 - x0 - 44
    cell_width = (inner_width - gap * (len(images) - 1)) // len(images)
    for index, (label, image, accent) in enumerate(images):
        x = x0 + 22 + index * (cell_width + gap)
        _cell(
            canvas,
            (x, y0 + 95, x + cell_width, y1 - 22),
            image,
            label=label,
            accent=accent,
        )


def _gallery_locked(data: ReportData, output: Path) -> None:
    ranks = (23, 24, 37, 39, 41, 43, 44, 45)
    canvas = _new(2400, 2700)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 28), "Locked growth gallery", fill=TEXT, font=FONT_42)
    draw.text(
        (42, 82),
        "the raw student follows thin longitudinal teacher geometry without a blend",
        fill=MUTED,
        font=FONT_26,
    )
    for index, rank in enumerate(ranks):
        item = data.locked(rank)
        column, row = index % 2, index // 2
        x0, y0 = 35 + column * 1180, 145 + row * 625
        _gallery_block(
            canvas,
            (x0, y0, x0 + 1150, y0 + 590),
            title=f"PHerc0139 · locked rank {rank}",
            detail=f"{item['detail']} · 8,192 samples · T=0.45",
            images=(
                ("coarse CT", _plain_ct(item["ct"]), None),
                ("released M7", _overlay(item["ct"], item["m7"], M7_CYAN), M7_CYAN),
                ("projected teacher", _overlay(item["ct"], item["teacher"], TEACHER_YELLOW), TEACHER_YELLOW),
                ("raw student", _overlay(item["ct"], item["student"], STUDENT_PINK), STUDENT_PINK),
            ),
        )
    _legend(
        draw,
        45,
        2650,
        [
            ("released M7", M7_CYAN),
            ("teacher", TEACHER_YELLOW),
            ("raw student", STUDENT_PINK),
        ],
    )
    draw.text((1510, 2648), "all panels use identical registered crops", fill=MUTED, font=FONT_24)
    _save(canvas, output / "gallery_locked_growth.png")


def _gallery_blind(data: ReportData, output: Path) -> None:
    selections = (
        (12032, "z12032_y03968_x03072"),
        (12043, "z12032_y03968_x03072"),
        (12045, "z12032_y03968_x03072"),
        (12050, "z12032_y03968_x03072"),
        (12056, "z12032_y03968_x03072"),
        (12171, "z12160_y04224_x02944"),
        (12224, "z12160_y03968_x03200"),
        (12864, "z12800_y04096_x03456"),
    )
    canvas = _new(2400, 2700)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 28), "Blind anti-blob gallery", fill=TEXT, font=FONT_42)
    draw.text(
        (42, 82),
        "one raw checkpoint removes released-M7 foreground inflation while retaining usable sheets",
        fill=MUTED,
        font=FONT_26,
    )
    for index, (coordinate, cube_id) in enumerate(selections):
        item = data.pherc(coordinate, cube_id)
        column, row = index % 2, index // 2
        x0, y0 = 35 + column * 1180, 145 + row * 625
        _gallery_block(
            canvas,
            (x0, y0, x0 + 1150, y0 + 590),
            title=f"PHerc1447 · z={coordinate}",
            detail=f"{cube_id} · {item['detail']} · T=0.45",
            images=(
                ("coarse CT", _plain_ct(item["ct"]), None),
                ("released M7", _overlay(item["ct"], item["m7"], M7_CYAN), M7_CYAN),
                ("raw student", _overlay(item["ct"], item["student"], STUDENT_PINK), STUDENT_PINK),
            ),
        )
    _legend(draw, 45, 2650, [("released M7", M7_CYAN), ("raw student", STUDENT_PINK)])
    draw.text((1450, 2648), "blind six-cube corpus · no fine teacher", fill=MUTED, font=FONT_24)
    _save(canvas, output / "gallery_blind_antiblob.png")


def _recolor_fitter_asset(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    m7_overlay = (red > green + 12) & (green > blue + 12) & (red > 70)
    added_pixels = (green > red + 28) & (green > blue + 28)
    rgb[m7_overlay] = M7_CYAN
    rgb[added_pixels] = SHARED_GREEN
    return Image.fromarray(rgb, mode="RGB")


def _line_fitter_examples(
    tracker: SourceTracker,
    source_root: Path,
    output: Path,
) -> None:
    fixup_root = source_root / "output" / "fixup_4x5x5"
    assets = source_root / LINE_FITTER_RELATIVE
    report = tracker.json(fixup_root / "run4" / "fixup_report.json")
    tracker.json(assets / "manifest.json")
    tracker.record(source_root / "CHANGELOG.md")

    site = _recolor_fitter_asset(
        tracker.image(assets / "run4_site_z4377.png", "RGB")
    )
    far_site = _recolor_fitter_asset(
        tracker.image(assets / "run4_far1_z4840.png", "RGB")
    )
    persistence = _recolor_fitter_asset(
        tracker.image(assets / "track_persistence.png", "RGB")
    )
    connected = tracker.image(
        assets / "val_CONNECTED_join_z4353_a3471_2913_b3473_2911.png",
        "RGB",
    )
    separate = tracker.image(
        assets / "val_SEPARATE_join_z4652_a3304_2826_b3306_2823.png",
        "RGB",
    )
    # These audit frames contain a labeled inset at the right edge. Letterbox
    # them to the tall paper cell so ImageOps.fit does not crop that evidence.
    connected = ImageOps.pad(
        connected, (375, 394), method=Image.Resampling.NEAREST, color=PANEL
    )
    separate = ImageOps.pad(
        separate, (375, 394), method=Image.Resampling.NEAREST, color=PANEL
    )

    canvas = _new(2400, 1280)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (42, 22),
        "Released M7 → independently gated additive repair",
        fill=TEXT,
        font=FONT_42,
    )
    draw.text(
        (42, 78),
        "the comparison starts from the released state of the art; green pixels are the only edits",
        fill=MUTED,
        font=FONT_26,
    )
    _cell(
        canvas,
        (40, 130, 1180, 600),
        site,
        label="accepted tracked gap · z=4377",
        detail="released M7 before | +38 pixels after",
        accent=SHARED_GREEN,
    )
    _cell(
        canvas,
        (1220, 130, 2360, 600),
        far_site,
        label="accepted far-tier gap · z=4840",
        detail="released M7 before | +64 pixels after",
        accent=SHARED_GREEN,
    )
    _cell(
        canvas,
        (40, 645, 1490, 1125),
        persistence,
        label="connection persistence across z=4669–4672",
        detail="one candidate survives the cross-slice support gate",
        accent=MEDIAL_PURPLE,
    )
    _cell(
        canvas,
        (1530, 645, 1925, 1125),
        connected,
        label="fine CT · connected",
        detail="accepted geometry",
        accent=SHARED_GREEN,
    )
    _cell(
        canvas,
        (1965, 645, 2360, 1125),
        separate,
        label="fine CT · separate",
        detail="audited failure",
        accent=PIN_ORANGE,
    )
    _legend(
        draw,
        45,
        1184,
        [("released M7", M7_CYAN), ("new additive pixels", SHARED_GREEN)],
    )
    draw.text(
        (970, 1182),
        f"{report['kept']}/{report['joins']} joins · "
        f"{report['painted_px']:,} px · 92.5% fine-CT precision · "
        "0 full-turn fusions",
        fill=TEXT,
        font=FONT_24,
    )
    _save(canvas, output / "figure_3_line_fitter_examples.png")


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], stop: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = start
    x1, y1 = stop
    draw.line((x0, y0, x1 - 18, y1), fill=color, width=5)
    draw.polygon([(x1, y1), (x1 - 24, y1 - 13), (x1 - 24, y1 + 13)], fill=color)


def _method_overview(data: ReportData, output: Path) -> None:
    item = data.locked(23)
    canvas = _new(2400, 900)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 24), "TRAINING ONLY", fill=TEACHER_YELLOW, font=FONT_32)
    draw.line((42, 68, 1580, 68), fill=(93, 78, 22), width=3)
    draw.text((1640, 24), "DEPLOYED", fill=STUDENT_PINK, font=FONT_32)
    draw.line((1640, 68, 2355, 68), fill=(93, 31, 77), width=3)

    _cell(canvas, (40, 105, 290, 700), _plain_ct(item["ct"]), label="9.362 µm CT")
    _cell(
        canvas,
        (310, 105, 560, 700),
        _overlay(item["ct"], item["m7"], M7_CYAN),
        label="released M7",
        detail="frozen state of art",
        accent=M7_CYAN,
    )
    _cell(
        canvas,
        (580, 105, 830, 700),
        _overlay(item["ct"], item["teacher"], TEACHER_YELLOW),
        label="fine teacher",
        detail="training only",
        accent=TEACHER_YELLOW,
    )
    _arrow(draw, (840, 402), (900, 402), TEACHER_YELLOW)

    train_box = (900, 105, 1580, 700)
    _rounded_panel(draw, train_box)
    draw.text((930, 132), "M7-initialized student", fill=TEXT, font=FONT_36)
    draw.text(
        (930, 182),
        "one ordinary 3D residual-encoder nnU-Net",
        fill=MUTED,
        font=FONT_24,
    )
    losses = [
        ("soft occupancy + Dice", TEACHER_YELLOW, "match projected fine evidence"),
        ("medial crest recall", MEDIAL_PURPLE, "grow along center geometry"),
        ("background separation", PIN_ORANGE, "do not buy recall with girth"),
        ("dynamic widest path", SHARED_GREEN, "raise one viable longitudinal route"),
        ("M7 KL + preservation", M7_CYAN, "retain supported M7 behavior"),
    ]
    y = 235
    for label, color, note in losses:
        draw.rounded_rectangle((935, y, 965, y + 30), radius=6, fill=color)
        draw.text((982, y - 2), label, fill=TEXT, font=FONT_24)
        draw.text((1250, y), note, fill=MUTED, font=FONT_22)
        y += 64
    draw.rounded_rectangle(
        (930, 578, 1550, 666),
        radius=12,
        fill=PANEL_ALT,
        outline=M7_CYAN,
        width=3,
    )
    draw.text(
        (952, 604),
        "optimizer step → global relative-L2 trust projection",
        fill=TEXT,
        font=FONT_22,
    )

    _arrow(draw, (1590, 402), (1640, 402), STUDENT_PINK)
    _cell(
        canvas,
        (1640, 105, 1890, 700),
        _overlay(item["ct"], item["student"], STUDENT_PINK),
        label="raw student",
        detail="T=0.45 · no blend",
        accent=STUDENT_PINK,
    )
    _arrow(draw, (1900, 402), (1950, 402), MUTED)
    _rounded_panel(draw, (1950, 105, 2360, 700))
    draw.text((1978, 142), "optional", fill=MUTED, font=FONT_26)
    draw.text((1978, 182), "2D line fitter", fill=TEXT, font=FONT_32)
    draw.text((1978, 248), "additive only", fill=SHARED_GREEN, font=FONT_26)
    draw.text((1978, 292), "separately gated", fill=MUTED, font=FONT_24)
    draw.text((1978, 336), "not part of", fill=MUTED, font=FONT_24)
    draw.text((1978, 374), "the learned model", fill=MUTED, font=FONT_24)
    draw.line((1978, 480, 2332, 480), fill=GRID, width=2)
    draw.text((1978, 510), "No teacher", fill=TEXT, font=FONT_24)
    draw.text((1978, 550), "No M7 blend", fill=TEXT, font=FONT_24)
    draw.text((1978, 590), "One checkpoint", fill=TEXT, font=FONT_24)
    draw.text(
        (40, 765),
        "4,096 PHerc0139 atlas rows × 2 passes = 8,192 samples",
        fill=MUTED,
        font=FONT_28,
    )
    draw.text(
        (40, 812),
        "released M7 is both the frozen initializer and the state-of-art visual baseline",
        fill=M7_CYAN,
        font=FONT_26,
    )
    draw.text((1630, 765), "inference boundary", fill=MUTED, font=FONT_28)
    _save(canvas, output / "figure_1_method_overview.png")


def _centers_and_radii(ct: np.ndarray, hard: np.ndarray, crest: np.ndarray) -> Image.Image:
    scale = 6
    image = _overlay(ct, hard, TEACHER_YELLOW).resize(
        (ct.shape[1] * scale, ct.shape[0] * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    radius = distance_transform_edt(hard)
    candidates = [
        (float(radius[y, x]), int(y), int(x))
        for y, x in np.argwhere(crest)
        if radius[y, x] >= 1.0
    ]
    chosen: list[tuple[float, int, int]] = []
    for value in sorted(candidates, reverse=True):
        _, y, x = value
        if all((y - py) ** 2 + (x - px) ** 2 >= 64 for _, py, px in chosen):
            chosen.append(value)
        if len(chosen) == 10:
            break
    for value, y, x in chosen:
        r = max(1.0, value) * scale
        cx, cy = x * scale + scale / 2, y * scale + scale / 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=MEDIAL_PURPLE, width=3)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=PIN_ORANGE)
    return image


def _pins_and_anchors(
    ct: np.ndarray,
    event: np.ndarray,
    pin_values: np.ndarray,
    free_mask: np.ndarray,
) -> Image.Image:
    scale = 6
    image = _multi_mask(
        ct,
        [
            (event, MEDIAL_PURPLE),
            ((pin_values & 1) > 0, M7_CYAN),
            ((pin_values & 2) > 0, PIN_ORANGE),
        ],
    ).resize((ct.shape[1] * scale, ct.shape[0] * scale), Image.Resampling.NEAREST)
    # Free anchors coincide with the contact pins in this event. A small green
    # center preserves both encodings: cyan/orange outer cells identify the two
    # component contacts, while green identifies unit-capacity anchors.
    draw = ImageDraw.Draw(image)
    for y, x in np.argwhere(free_mask):
        cx, cy = int(x) * scale + scale // 2, int(y) * scale + scale // 2
        draw.rectangle((cx - 1, cy - 1, cx + 1, cy + 1), fill=SHARED_GREEN)
    return image


def _crossres_supervision(tracker: SourceTracker, source_root: Path, output: Path) -> None:
    connectivity = source_root / CONNECTIVITY_RELATIVE
    atlas = source_root / ATLAS_RELATIVE
    connectivity_state = connectivity / "connectivity_state.json"
    atlas_state = atlas / "atlas_state.json"
    event_ids = tracker.zarr(
        connectivity / "event_ids.zarr",
        identity=connectivity_state,
        description="dynamic medial event IDs",
    )
    pins = tracker.zarr(
        connectivity / "pin_membership.zarr",
        identity=connectivity_state,
        description="dynamic medial component-contact pin bitsets",
    )
    free = tracker.zarr(
        connectivity / "free_anchors.zarr",
        identity=connectivity_state,
        description="dynamic medial unit-capacity anchors",
    )
    teacher_q = tracker.zarr(
        atlas / "teacher_q.zarr",
        identity=atlas_state,
        description="projected fine-teacher soft occupancy",
    )
    teacher_crest = tracker.zarr(
        atlas / "teacher_crest.zarr",
        identity=atlas_state,
        description="projected fine-teacher medial crest",
    )
    teacher_crest_valid = tracker.zarr(
        atlas / "teacher_crest_valid.zarr",
        identity=atlas_state,
        description="medial crest validity",
    )
    target_valid = tracker.zarr(
        atlas / "target_valid.zarr",
        identity=atlas_state,
        description="projected occupancy validity",
    )
    ct_volume = tracker.zarr(
        source_root / CT_RELATIVE,
        identity=source_root / CT_RELATIVE / ".zattrs",
        description="PHerc0139 9.362 µm CT",
    )
    m7_volume = tracker.zarr(
        source_root / M7_RELATIVE,
        identity=source_root / M7_RELATIVE / ".zattrs",
        description="released M7 PHerc0139 segmentation",
    )

    z, y0, y1, x0, x1 = 5738, 2556, 2612, 2428, 2500
    ct = np.asarray(ct_volume[z, y0:y1, x0:x1], dtype=np.uint8)
    ct = np.asarray(ImageOps.autocontrast(Image.fromarray(ct, mode="L")), dtype=np.uint8)
    q_block = np.asarray(
        teacher_q[z - 2 : z + 3, y0 - 2 : y1 + 2, x0 - 2 : x1 + 2],
        dtype=np.float32,
    ) / 255.0
    valid_block = np.asarray(
        target_valid[z - 2 : z + 3, y0 - 2 : y1 + 2, x0 - 2 : x1 + 2]
    ) > 0
    crest_block = np.asarray(
        teacher_crest[z - 2 : z + 3, y0 - 2 : y1 + 2, x0 - 2 : x1 + 2]
    ) > 0
    crest_valid_block = np.asarray(
        teacher_crest_valid[z - 2 : z + 3, y0 - 2 : y1 + 2, x0 - 2 : x1 + 2]
    ) > 0
    positive = crest_block | ((q_block >= 0.5) & valid_block & ~crest_valid_block)
    shell_block = (
        (maximum_filter(positive.astype(np.uint8), size=(5, 5, 5)) > 0)
        & valid_block
        & (q_block <= 0.1)
        & ~positive
    )
    q = q_block[2, 2:-2, 2:-2]
    crest = crest_block[2, 2:-2, 2:-2]
    shell = shell_block[2, 2:-2, 2:-2]
    hard = q >= 0.5
    m7 = np.asarray(m7_volume[z, y0:y1, x0:x1]) > 0
    event = np.asarray(event_ids[z, y0:y1, x0:x1]) == 138
    pin_values = np.asarray(pins[z, y0:y1, x0:x1], dtype=np.uint8)
    free_mask = np.asarray(free[z, y0:y1, x0:x1]) > 0

    shell_image = _multi_mask(ct, [(hard, TEACHER_YELLOW), (shell, PIN_ORANGE)])
    crest_image = _multi_mask(ct, [(hard, (105, 88, 16)), (crest, MEDIAL_PURPLE)])
    corridor_image = _multi_mask(ct, [(m7, M7_CYAN), (event, MEDIAL_PURPLE)])
    pins_image = _pins_and_anchors(ct, event, pin_values, free_mask)
    panels = [
        ("coarse CT", _plain_ct(ct), None, "z=5738 · 9.362 µm"),
        ("soft occupancy q", _probability_overlay(ct, q, TEACHER_YELLOW), TEACHER_YELLOW, "projected fine teacher"),
        ("centers + radii", _centers_and_radii(ct, hard, crest), MEDIAL_PURPLE, "crest centers; local EDT radii"),
        ("medial crest C", crest_image, MEDIAL_PURPLE, "sparse longitudinal target"),
        ("separation shell S", shell_image, PIN_ORANGE, "radius 2; q ≤ 0.1"),
        ("released M7", _overlay(ct, m7, M7_CYAN), M7_CYAN, "disconnected fragments"),
        ("teacher surface", _overlay(ct, hard, TEACHER_YELLOW), TEACHER_YELLOW, "q ≥ 0.5 for display"),
        ("event 138 corridor", corridor_image, MEDIAL_PURPLE, "125 voxels · 44 steps"),
        ("pins + free anchors", pins_image, SHARED_GREEN, "cyan/orange pin sets; green anchors"),
    ]
    canvas = _new(2400, 1050)
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 22), "Teacher center/radius supervision", fill=TEACHER_YELLOW, font=FONT_36)
    draw.text((40, 66), "de-blob pressure acts around the crest, not across the whole blob", fill=MUTED, font=FONT_24)
    draw.text((40, 527), "Dynamic max–min connectivity", fill=SHARED_GREEN, font=FONT_36)
    draw.text((40, 571), "one viable path is selected inside the actual teacher-medial corridor", fill=MUTED, font=FONT_24)
    col_w, gap = 448, 20
    for index, (label, image, accent, detail) in enumerate(panels[:5]):
        x = 40 + index * (col_w + gap)
        _cell(canvas, (x, 108, x + col_w, 500), image, label=label, detail=detail, accent=accent)
    for index, (label, image, accent, detail) in enumerate(panels[5:]):
        x = 40 + index * (col_w + gap)
        _cell(canvas, (x, 613, x + col_w, 1005), image, label=label, detail=detail, accent=accent)
    summary_x = 40 + 4 * (col_w + gap)
    _rounded_panel(draw, (summary_x, 613, summary_x + col_w, 1005))
    draw.text((summary_x + 24, 640), "loss target", fill=TEXT, font=FONT_30)
    draw.text((summary_x + 24, 694), "raise the weakest", fill=MUTED, font=FONT_24)
    draw.text((summary_x + 24, 733), "model probability on", fill=MUTED, font=FONT_24)
    draw.text((summary_x + 24, 772), "the best pin-to-pin", fill=MUTED, font=FONT_24)
    draw.text((summary_x + 24, 811), "path toward p = 0.2", fill=SHARED_GREEN, font=FONT_26)
    draw.line((summary_x + 24, 862, summary_x + col_w - 24, 862), fill=GRID, width=2)
    draw.text((summary_x + 24, 888), "No fixed route.", fill=TEXT, font=FONT_24)
    draw.text((summary_x + 24, 925), "No reward for girth.", fill=TEXT, font=FONT_24)
    _save(canvas, output / "figure_crossres_supervision.png")


def _locked_grid(data: ReportData, output: Path) -> None:
    ranks = sorted(data.locked_rows)
    width, row_height, header = 2250, 270, 110
    canvas = _new(width, header + row_height * len(ranks) + 70)
    draw = ImageDraw.Draw(canvas)
    headers = ("coarse CT", "released M7", "projected teacher", "raw student", "teacher comparison")
    left, col_w, gap = 185, 390, 14
    for index, label in enumerate(headers):
        draw.text((left + index * (col_w + gap) + 10, 42), label, fill=TEXT, font=FONT_26)
    for row_index, rank in enumerate(ranks):
        item = data.locked(rank)
        y = header + row_index * row_height
        draw.text((25, y + 76), f"rank {rank}", fill=TEXT, font=FONT_30)
        draw.text((25, y + 116), item["detail"], fill=MUTED, font=FONT_22)
        images = (
            _plain_ct(item["ct"]),
            _overlay(item["ct"], item["m7"], M7_CYAN),
            _overlay(item["ct"], item["teacher"], TEACHER_YELLOW),
            _overlay(item["ct"], item["student"], STUDENT_PINK),
            _difference(item["ct"], item["student"], item["teacher"]),
        )
        for column, image in enumerate(images):
            x = left + column * (col_w + gap)
            _cell(canvas, (x, y + 8, x + col_w, y + row_height - 8), image)
    _legend(draw, 195, canvas.height - 48, [("shared", SHARED_GREEN), ("teacher only", TEACHER_YELLOW), ("student only", STUDENT_PINK)])
    draw.text((1490, canvas.height - 50), "8,192 samples · T=0.45 · raw masks", fill=MUTED, font=FONT_24)
    _save(canvas, output / "locked16_release_grid.png")


def _pherc_grid(data: ReportData, output: Path) -> None:
    rows = data.pherc_rows
    width, row_height, header = 2350, 315, 110
    canvas = _new(width, header + row_height * len(rows) + 70)
    draw = ImageDraw.Draw(canvas)
    headers = ("coarse CT", "released M7", "v15 reference", "raw student", "reference comparison")
    left, col_w, gap = 245, 390, 14
    for index, label in enumerate(headers):
        draw.text((left + index * (col_w + gap) + 10, 42), label, fill=TEXT, font=FONT_26)
    for row_index, row in enumerate(rows):
        item = data.pherc_row(row)
        y = header + row_index * row_height
        draw.text((22, y + 68), row["cube_id"].split("_")[0], fill=MUTED, font=FONT_22)
        draw.text((22, y + 103), f"z={int(row['global_coordinate'])}", fill=TEXT, font=FONT_30)
        draw.text((22, y + 143), item["detail"], fill=MUTED, font=FONT_22)
        images = (
            _plain_ct(item["ct"]),
            _overlay(item["ct"], item["m7"], M7_CYAN),
            _overlay(item["ct"], item["reference"], TEACHER_YELLOW),
            _overlay(item["ct"], item["student"], STUDENT_PINK),
            _difference(item["ct"], item["student"], item["reference"]),
        )
        for column, image in enumerate(images):
            x = left + column * (col_w + gap)
            _cell(canvas, (x, y + 8, x + col_w, y + row_height - 8), image)
    _legend(draw, 255, canvas.height - 48, [("shared", SHARED_GREEN), ("v15 only", TEACHER_YELLOW), ("student only", STUDENT_PINK)])
    draw.text((1530, canvas.height - 50), "v15 is a comparison, not truth", fill=MUTED, font=FONT_24)
    _save(canvas, output / "pherc1447_release_grid.png")


def _plot_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    x_values: list[float],
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    y_label: str,
) -> tuple[Any, Any]:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=GRID, width=3)
    draw.line((x0, y0, x0, y1), fill=GRID, width=3)
    for value in y_ticks:
        py = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
        draw.line((x0, py, x1, py), fill=(25, 35, 52), width=2)
        draw.text((x0 - 78, py - 12), f"{value:.2f}", fill=MUTED, font=FONT_22)
    for value in x_values:
        px = x0 + (value - x_values[0]) / (x_values[-1] - x_values[0]) * (x1 - x0)
        draw.text((px - 20, y1 + 14), f"{value:.2f}", fill=MUTED, font=FONT_22)
    draw.text((x0, y0 - 45), y_label, fill=TEXT, font=FONT_28)
    return (
        lambda value: x0 + (value - x_values[0]) / (x_values[-1] - x_values[0]) * (x1 - x0),
        lambda value: y1 - (value - y_min) / (y_max - y_min) * (y1 - y0),
    )


def _duration_selection(tracker: SourceTracker, source_root: Path, output: Path) -> None:
    audit = tracker.json(source_root / FLIP_RELATIVE)
    rows = [
        row
        for row in audit["ranked_summary"]
        if 0.38 <= float(row["threshold"]) <= 0.50
    ]
    samples = (1024, 2048, 3072, 4096, 8192)
    colors = (M7_CYAN, TEACHER_YELLOW, MEDIAL_PURPLE, SHARED_GREEN, STUDENT_PINK)
    thresholds = sorted({float(row["threshold"]) for row in rows})
    canvas = _new(2200, 960)
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 24), "Why 8,192 samples at T=0.45?", fill=TEXT, font=FONT_42)
    draw.text((55, 76), "literal teacher resemblance and blind anti-blob behavior answer different questions", fill=MUTED, font=FONT_26)
    left_box = (140, 190, 1030, 790)
    right_box = (1260, 190, 2150, 790)
    x_left, y_left = _plot_axes(
        draw,
        left_box,
        x_values=thresholds,
        y_min=0.21,
        y_max=0.27,
        y_ticks=[0.22, 0.23, 0.24, 0.25, 0.26],
        y_label="locked-16 FLIP mean · lower is closer",
    )
    x_right, y_right = _plot_axes(
        draw,
        right_box,
        x_values=thresholds,
        y_min=0.75,
        y_max=1.35,
        y_ticks=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        y_label="PHerc1447 foreground / v15 reference",
    )
    band_top, band_bottom = y_right(1.10), y_right(0.90)
    draw.rectangle((right_box[0], band_top, right_box[2], band_bottom), fill=(12, 42, 31))
    draw.text((right_box[0] + 18, band_top + 10), "bounded foreground band", fill=SHARED_GREEN, font=FONT_22)
    for sample, color in zip(samples, colors, strict=True):
        series = sorted(
            (row for row in rows if int(row["samples"]) == sample),
            key=lambda row: float(row["threshold"]),
        )
        flip_points = [
            (x_left(float(row["threshold"])), y_left(float(row["flip_mean_macro"])))
            for row in series
        ]
        ratio_points = [
            (
                x_right(float(row["threshold"])),
                y_right(float(row["pherc1447_foreground_ratio_vs_v15"])),
            )
            for row in series
            if row["pherc1447_foreground_ratio_vs_v15"] is not None
        ]
        if len(flip_points) > 1:
            draw.line(flip_points, fill=color, width=4, joint="curve")
        if len(ratio_points) > 1:
            draw.line(ratio_points, fill=color, width=4, joint="curve")
        for px, py in flip_points:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        for px, py in ratio_points:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    selected = next(
        row
        for row in rows
        if int(row["samples"]) == 8192 and float(row["threshold"]) == 0.45
    )
    for px, py in [
        (x_left(0.45), y_left(float(selected["flip_mean_macro"]))),
        (x_right(0.45), y_right(float(selected["pherc1447_foreground_ratio_vs_v15"]))),
    ]:
        draw.ellipse((px - 13, py - 13, px + 13, py + 13), outline=TEXT, width=4)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=STUDENT_PINK)
    _legend(draw, 120, 850, [(f"{sample:,}", color) for sample, color in zip(samples, colors, strict=True)])
    draw.text((1415, 848), "white ring = selected release", fill=TEXT, font=FONT_24)
    draw.text((1415, 886), "FLIP-only winner: 4,096 / 0.42", fill=MUTED, font=FONT_22)
    _save(canvas, output / "figure_duration_selection.png")


def generate(source_root: Path, output: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    tracker = SourceTracker(source_root)
    report_root = source_root / REPORT_RELATIVE
    data = ReportData(tracker, report_root)

    _teaser(data, output)
    _registered_examples(data, output)
    _failure_cases(data, output)
    _line_fitter_examples(tracker, source_root, output)
    _gallery_locked(data, output)
    _gallery_blind(data, output)
    _method_overview(data, output)
    _crossres_supervision(tracker, source_root, output)
    _locked_grid(data, output)
    _pherc_grid(data, output)
    _duration_selection(tracker, source_root, output)

    generated = []
    for path in sorted(output.glob("*.png"), key=lambda item: item.name):
        generated.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    manifest = {
        "schema": "socratic-method-release-figures-v1",
        "selection": {
            "checkpoint_samples": 8192,
            "operating_threshold": 0.45,
            "model_composition": "raw-student-only-no-m7-blend-no-teacher",
        },
        "palette": {
            "background": BACKGROUND,
            "released_m7": M7_CYAN,
            "teacher_or_reference": TEACHER_YELLOW,
            "raw_student": STUDENT_PINK,
            "shared": SHARED_GREEN,
            "medial": MEDIAL_PURPLE,
            "pins_or_shell": PIN_ORANGE,
        },
        "discrete_inputs": tracker.manifest_inputs(),
        "zarr_sources": tracker.zarr_sources,
        "generated": generated,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate data-backed M7-XR release and paper figures"
    )
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\work\vesuvius-c"))
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "submissions" / "2026-09" / "figures",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = generate(args.source_root, args.output)
    print(json.dumps({"generated": manifest["generated"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
