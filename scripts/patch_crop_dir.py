from __future__ import annotations

import csv
import sys
from pathlib import Path


def _progress(current: int, total: int, width: int = 40) -> None:
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def main() -> int:
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/jaad")
    manifest_path = processed_dir / "cleaned_sequences_manifest.csv"

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    total = len(rows)

    patched = 0
    for i, row in enumerate(rows, 1):
        _progress(i, total)
        sequence_id = row["sequence_id"]
        crop_dir = processed_dir / "crops" / sequence_id
        if crop_dir.exists():
            count = len(list(crop_dir.glob("*.jpg")))
            row["crop_dir"] = str(crop_dir.relative_to(processed_dir))
            row["crops_exported"] = count
            patched += 1

    print()
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total sequences : {len(rows)}")
    print(f"Patched         : {patched}")
    print(f"Still empty     : {len(rows) - patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
