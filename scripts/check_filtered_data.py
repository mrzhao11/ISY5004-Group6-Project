from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/jaad")
    manifest_path = processed_dir / "filtered_sequences_manifest.csv"
    metadata_path = processed_dir / "filtered_metadata.csv"

    for p in (manifest_path, metadata_path):
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    manifest_rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    metadata_rows = list(csv.DictReader(metadata_path.open("r", encoding="utf-8")))

    # ── check manifest ───────────────────────────────────────────────────────
    print(f"=== filtered_sequences_manifest.csv  ({len(manifest_rows)} sequences) ===")

    unknown_manifest = [
        r for r in manifest_rows
        if r["sequence_behavior_label"] == "unknown"
        or r["sequence_crossing_label"] == "unknown"
        or r["sequence_look_label"] == "unknown"
    ]
    bad_crops = [
        r for r in manifest_rows
        if not r.get("crop_dir", "").strip()
        or int(r.get("crops_exported", 0)) != int(r.get("frame_count", 0))
    ]

    print(f"  Unknown labels       : {len(unknown_manifest)}")
    print(f"  Incomplete crops     : {len(bad_crops)}")

    print("\n  Label distribution:")
    for col in ("sequence_behavior_label", "sequence_crossing_label", "sequence_look_label"):
        counter = Counter(r[col] for r in manifest_rows)
        print(f"    [{col}]")
        for v, c in counter.most_common():
            print(f"      {v!r:<25} {c:>8}")

    # ── check metadata ─���───────────────────────────���─────────────────────────
    print(f"\n=== filtered_metadata.csv  ({len(metadata_rows)} rows) ===")

    unknown_metadata = [
        r for r in metadata_rows
        if r["behavior_label"] == "unknown"
        or r["crossing_label"] == "unknown"
        or r["look_label"] == "unknown"
    ]
    print(f"  Unknown labels       : {len(unknown_metadata)}")

    print("\n  Label distribution:")
    for col in ("behavior_label", "crossing_label", "look_label"):
        counter = Counter(r[col] for r in metadata_rows)
        print(f"    [{col}]")
        for v, c in counter.most_common():
            print(f"      {v!r:<25} {c:>8}")

    # ── summary ───────────────────────────���──────────────────────────────────
    print("\n=== Summary ===")
    issues = len(unknown_manifest) + len(bad_crops) + len(unknown_metadata)
    if issues == 0:
        print("  All clean.")
    else:
        print(f"  {issues} issue(s) found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
