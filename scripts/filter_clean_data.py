from __future__ import annotations

import csv
import sys
from pathlib import Path


def _progress(current: int, total: int, label: str = "", width: int = 40) -> None:
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label} [{bar}] {current}/{total}", end="", flush=True)


def main() -> int:
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/jaad")
    manifest_path = processed_dir / "cleaned_sequences_manifest.csv"
    metadata_path = processed_dir / "cleaned_metadata.csv"

    for p in (manifest_path, metadata_path):
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    # ── Step 1: find incomplete/missing crop sequences from manifest ────────
    print("Step 1: checking manifest for incomplete crops...")
    manifest_rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    bad_sequence_ids: set[str] = set()

    for row in manifest_rows:
        crop_dir_str = row.get("crop_dir", "").strip()
        expected = int(row.get("frame_count", 0))
        exported = int(row.get("crops_exported", 0))
        if not crop_dir_str or exported != expected:
            bad_sequence_ids.add(row["sequence_id"])

    print(f"  Incomplete/missing crop sequences: {len(bad_sequence_ids)}")

    # ── Step 2: filter manifest ─────────────────────────────────────────────
    print("Step 2: filtering manifest...")
    filtered_manifest = []
    manifest_removed_unknown = 0
    manifest_removed_crops = 0

    for i, row in enumerate(manifest_rows, 1):
        _progress(i, len(manifest_rows), "  manifest", width=35)
        if row["sequence_id"] in bad_sequence_ids:
            manifest_removed_crops += 1
            continue
        if (
            row["sequence_behavior_label"] == "unknown"
            or row["sequence_crossing_label"] == "unknown"
            or row["sequence_look_label"] == "unknown"
        ):
            manifest_removed_unknown += 1
            continue
        filtered_manifest.append(row)

    print(f"\n  Removed (incomplete crops): {manifest_removed_crops}")
    print(f"  Removed (unknown labels)  : {manifest_removed_unknown}")
    print(f"  Remaining sequences       : {len(filtered_manifest)}")

    # ── Step 3: filter metadata ─────────────────────────────────────────────
    print("Step 3: filtering metadata...")
    metadata_rows = list(csv.DictReader(metadata_path.open("r", encoding="utf-8")))
    filtered_metadata = []
    metadata_removed = 0

    for i, row in enumerate(metadata_rows, 1):
        _progress(i, len(metadata_rows), "  metadata", width=35)
        if (
            row["behavior_label"] == "unknown"
            or row["crossing_label"] == "unknown"
            or row["look_label"] == "unknown"
        ):
            metadata_removed += 1
            continue
        filtered_metadata.append(row)

    print(f"\n  Removed (unknown labels): {metadata_removed}")
    print(f"  Remaining rows          : {len(filtered_metadata)}")

    # ── Step 4: write output ────────────────────────────────────────────────
    print("Step 4: writing output files...")
    out_manifest = processed_dir / "filtered_sequences_manifest.csv"
    out_metadata = processed_dir / "filtered_metadata.csv"

    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(filtered_manifest)

    with out_metadata.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(filtered_metadata)

    print(f"\nDone.")
    print(f"  {out_manifest}")
    print(f"  {out_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
