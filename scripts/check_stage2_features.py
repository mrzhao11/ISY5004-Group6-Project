from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REQUIRED_FIELDS = {
    "sequence_id",
    "speed_mean",
    "speed_var",
    "displacement",
    "bbox_scale_change",
    "trajectory_length",
    "center_jitter",
    "motion_direction",
}


def main() -> int:
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/jaad")
    feature_path = processed_dir / "trajectory_features.csv"
    if not feature_path.exists():
        raise SystemExit(f"Feature file not found: {feature_path}")

    with feature_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        available_fields = set(reader.fieldnames or [])

    missing_fields = sorted(REQUIRED_FIELDS - available_fields)
    print(
        json.dumps(
            {
                "processed_dir": str(processed_dir),
                "row_count": len(rows),
                "missing_required_fields": missing_fields,
                "sample_sequence_ids": [row["sequence_id"] for row in rows[:3]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
