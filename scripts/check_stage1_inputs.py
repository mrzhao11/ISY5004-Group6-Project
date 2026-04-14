from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/jaad")
    manifest_path = processed_dir / "cleaned_sequences_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    missing_crop_dirs = 0
    incomplete_sequences = 0

    for row in rows:
        crop_dir = processed_dir / row["crop_dir"] if row["crop_dir"] else None
        expected_frames = int(row["frame_count"])
        if crop_dir is None or not crop_dir.exists():
            missing_crop_dirs += 1
            continue
        actual_frames = len(list(crop_dir.glob("*.jpg")))
        if actual_frames != expected_frames:
            incomplete_sequences += 1

    print(
        json.dumps(
            {
                "processed_dir": str(processed_dir),
                "sequence_count": len(rows),
                "missing_crop_dirs": missing_crop_dirs,
                "incomplete_sequences": incomplete_sequences,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
