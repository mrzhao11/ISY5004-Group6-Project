"""
Analyse JAAD ground-truth: per-video frame count, per-video pedestrian count,
and average pedestrians visible per frame. Outputs:

  - JSON with full per-video stats
  - Console summary table
  - Recommended representative clips for the multi-clip latency sweep
    (p25 / p50 / p75 / p90 by frame count)
"""

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

ANN_DIR = Path("data/raw/jaad/annotations")
FRAMES_DIR = Path("data/raw/jaad/images")
OUT_PATH = Path("results/jaad_distribution.json")


def parse_video(xml_path: Path):
    """Return (n_frames, unique_pedestrians, avg_peds_per_frame, max_peds_per_frame)."""
    root = ET.parse(xml_path).getroot()
    ped_tracks = [t for t in root.findall("track") if t.get("label") == "pedestrian"]
    if not ped_tracks:
        return None

    frame_to_peds = defaultdict(set)
    unique_ids = set()
    for track in ped_tracks:
        # Find this track's stable pedestrian id from any box <attribute name="id">
        ped_id = None
        for box in track.findall("box"):
            for attr in box.findall("attribute"):
                if attr.get("name") == "id":
                    ped_id = attr.text
                    break
            if ped_id:
                break
        if ped_id is None:
            ped_id = f"unknown_{id(track)}"
        unique_ids.add(ped_id)

        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            f = int(box.get("frame", 0))
            frame_to_peds[f].add(ped_id)

    n_frames = max(frame_to_peds.keys()) + 1 if frame_to_peds else 0
    counts_per_frame = [len(frame_to_peds.get(f, set())) for f in range(n_frames)]
    avg_peds = float(np.mean(counts_per_frame)) if counts_per_frame else 0.0
    max_peds = max(counts_per_frame) if counts_per_frame else 0
    return n_frames, len(unique_ids), avg_peds, max_peds


def main():
    rows = []
    for xml in sorted(ANN_DIR.glob("video_*.xml")):
        vid = xml.stem
        # frame count: prefer the actual JPGs (matches what the pipeline uses)
        frames_dir = FRAMES_DIR / vid
        if frames_dir.is_dir():
            jpg_frames = len(list(frames_dir.glob("frame_*.jpg")))
        else:
            jpg_frames = 0
        parsed = parse_video(xml)
        if parsed is None:
            continue
        ann_frames, n_unique, avg_p, max_p = parsed
        rows.append({
            "video":          vid,
            "frames_jpg":     jpg_frames,
            "frames_ann":     ann_frames,
            "unique_peds":    n_unique,
            "avg_peds_per_frame": round(avg_p, 2),
            "max_peds_in_frame":  max_p,
        })

    fr = np.array([r["frames_jpg"] for r in rows if r["frames_jpg"] > 0])
    pe = np.array([r["unique_peds"] for r in rows])
    ap = np.array([r["avg_peds_per_frame"] for r in rows])
    mp = np.array([r["max_peds_in_frame"] for r in rows])

    summary = {
        "n_videos": len(rows),
        "frames_per_clip": {
            "min": int(fr.min()), "p25": int(np.percentile(fr, 25)),
            "median": int(np.median(fr)), "mean": float(round(fr.mean(), 1)),
            "p75": int(np.percentile(fr, 75)), "p90": int(np.percentile(fr, 90)),
            "max": int(fr.max()),
        },
        "unique_peds_per_clip": {
            "min": int(pe.min()), "p25": int(np.percentile(pe, 25)),
            "median": int(np.median(pe)), "mean": float(round(pe.mean(), 2)),
            "p75": int(np.percentile(pe, 75)), "p90": int(np.percentile(pe, 90)),
            "max": int(pe.max()),
        },
        "avg_peds_per_frame": {
            "min":   float(round(ap.min(), 2)),
            "p25":   float(round(np.percentile(ap, 25), 2)),
            "median":float(round(np.percentile(ap, 50), 2)),
            "mean":  float(round(ap.mean(), 2)),
            "p75":   float(round(np.percentile(ap, 75), 2)),
            "p90":   float(round(np.percentile(ap, 90), 2)),
            "max":   float(round(ap.max(), 2)),
        },
        "max_peds_in_frame": {
            "min": int(mp.min()), "p25": int(np.percentile(mp, 25)),
            "median": int(np.median(mp)), "mean": float(round(mp.mean(), 2)),
            "p75": int(np.percentile(mp, 75)), "p90": int(np.percentile(mp, 90)),
            "max": int(mp.max()),
        },
    }

    # Pick representative clips: closest to each frame-count percentile.
    targets = {"p25": summary["frames_per_clip"]["p25"],
               "p50": summary["frames_per_clip"]["median"],
               "p75": summary["frames_per_clip"]["p75"],
               "p90": summary["frames_per_clip"]["p90"]}
    representative = {}
    for label, tgt in targets.items():
        best = min(rows, key=lambda r: abs(r["frames_jpg"] - tgt) if r["frames_jpg"] > 0 else 1e9)
        representative[label] = {
            "video":        best["video"],
            "frames":       best["frames_jpg"],
            "unique_peds":  best["unique_peds"],
            "avg_peds":     best["avg_peds_per_frame"],
            "target":       tgt,
        }

    out = {
        "summary":        summary,
        "representative": representative,
        "per_video":      rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"Videos parsed: {summary['n_videos']}")
    print()
    print("Frame count per clip:")
    for k, v in summary["frames_per_clip"].items():
        print(f"  {k:>6}: {v}")
    print()
    print("Unique pedestrians per clip (ground truth):")
    for k, v in summary["unique_peds_per_clip"].items():
        print(f"  {k:>6}: {v}")
    print()
    print("Avg pedestrians visible per frame:")
    for k, v in summary["avg_peds_per_frame"].items():
        print(f"  {k:>6}: {v}")
    print()
    print("Max pedestrians in a single frame:")
    for k, v in summary["max_peds_in_frame"].items():
        print(f"  {k:>6}: {v}")
    print()
    print("Representative clips for multi-clip latency sweep:")
    for label, r in representative.items():
        print(f"  {label} (target {r['target']} frames):  {r['video']}  "
              f"frames={r['frames']}  peds={r['unique_peds']}  "
              f"avg_peds/frame={r['avg_peds']:.2f}")
    print()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
