from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from crossres_pred.voxel.grid_growth import grow_probability_grid


def test_probability_grid_growth_writes_a_provenanced_subset(tmp_path: Path) -> None:
    input_grid = tmp_path / "input"
    (input_grid / "cubes_PRED").mkdir(parents=True)
    (input_grid / "probability").mkdir()
    cube_id = "z00016_y00016_x00016"
    shape = (16, 16, 16)
    seed = np.zeros(shape, dtype=np.uint8)
    seed[8, 8, 2:5] = 255
    probability = np.zeros(shape, dtype=np.float16)
    probability[seed != 0] = 0.9
    probability[8, 8, 5:10] = 0.44
    tifffile.imwrite(input_grid / "cubes_PRED" / f"{cube_id}.tif", seed)
    tifffile.imwrite(input_grid / "probability" / f"{cube_id}.tif", probability)
    (input_grid / "cubes_PRED" / "present.json").write_text(
        json.dumps([cube_id]) + "\n",
        encoding="utf-8",
    )
    (input_grid / "source_manifest.json").write_text(
        json.dumps({"chunk_size": 16}) + "\n",
        encoding="utf-8",
    )
    (input_grid / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "synthetic-grid",
                "target_cube_ids": [cube_id],
                "options": {"threshold": 0.5},
                "checkpoint": {"sha256": "synthetic-checkpoint"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "grown"
    result = grow_probability_grid(
        input_grid=input_grid,
        output_path=output,
        support_threshold=0.4,
        max_steps=6,
        halo=6,
        workers=1,
        max_cpu_threads=2,
        target_cube_ids=[cube_id],
    )

    assert result == output.resolve()
    grown = tifffile.imread(output / "cubes_PRED" / f"{cube_id}.tif") != 0
    assert np.all(grown[8, 8, 2:10])
    growth = json.loads((output / "growth.json").read_text(encoding="utf-8"))
    assert growth["schema"] == "crossres-probability-ridge-growth-v1"
    assert growth["aggregate"]["added_positive"] == 5
    assert growth["seam_reconciliation"]["removed_added_voxels"] == 0
    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["parent_grid"] == str(input_grid.resolve())
    assert provenance["checkpoint"]["sha256"] == "synthetic-checkpoint"
    assert provenance["target_cube_ids"] == [cube_id]
