from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".tar.bz2"}


@dataclass
class AnnotationRecord:
    video_id: str
    pedestrian_id: str
    frame_id: int
    xtl: float
    ytl: float
    width: float
    height: float
    behavior_label: str
    crossing_label: str
    source_file: str
    raw_attributes: Dict[str, str]


def _normalize_name(value: Optional[str], default: str) -> str:
    if not value:
        return default
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or default


def _safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _safe_int(value: Optional[str], default: int = -1) -> int:
    try:
        return int(float(value)) if value is not None else default
    except ValueError:
        return default


def _extract_bbox(node: ET.Element) -> Tuple[float, float, float, float]:
    if {"xtl", "ytl", "xbr", "ybr"} <= set(node.attrib):
        xtl = _safe_float(node.attrib.get("xtl"))
        ytl = _safe_float(node.attrib.get("ytl"))
        xbr = _safe_float(node.attrib.get("xbr"))
        ybr = _safe_float(node.attrib.get("ybr"))
        return xtl, ytl, max(0.0, xbr - xtl), max(0.0, ybr - ytl)

    if {"left", "top", "width", "height"} <= set(node.attrib):
        return (
            _safe_float(node.attrib.get("left")),
            _safe_float(node.attrib.get("top")),
            max(0.0, _safe_float(node.attrib.get("width"))),
            max(0.0, _safe_float(node.attrib.get("height"))),
        )

    if {"x", "y", "w", "h"} <= set(node.attrib):
        return (
            _safe_float(node.attrib.get("x")),
            _safe_float(node.attrib.get("y")),
            max(0.0, _safe_float(node.attrib.get("w"))),
            max(0.0, _safe_float(node.attrib.get("h"))),
        )

    return 0.0, 0.0, 0.0, 0.0


def _collect_attributes(node: ET.Element) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for child in node:
        key = child.attrib.get("name") or child.attrib.get("label") or child.tag
        if key:
            attrs[key.lower()] = (child.text or "").strip()
    return attrs


def _is_pedestrian(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in ("ped", "person"))


def _parse_track_boxes(path: Path, root: ET.Element, video_id: str) -> List[AnnotationRecord]:
    records: List[AnnotationRecord] = []
    for track in root.iter("track"):
        label = track.attrib.get("label") or track.attrib.get("name") or ""
        if label and not _is_pedestrian(label):
            continue

        track_attrs = _collect_attributes(track)
        track_ped_id = (
            track_attrs.get("id")
            or track_attrs.get("pedestrian_id")
            or track.attrib.get("id")
            or f"ped_{len(records):04d}"
        )

        for box in track.findall("box"):
            attrs = track_attrs | _collect_attributes(box)
            frame_id = _safe_int(box.attrib.get("frame") or box.attrib.get("frame_id"))
            xtl, ytl, width, height = _extract_bbox(box)
            pedestrian_id = _normalize_name(
                attrs.get("id") or attrs.get("pedestrian_id") or track_ped_id,
                "ped_unknown",
            )
            behavior_label = (
                attrs.get("action")
                or attrs.get("behavior")
                or attrs.get("pose")
                or attrs.get("state")
                or "unknown"
            )
            crossing_label = (
                attrs.get("crossing")
                or attrs.get("cross")
                or attrs.get("intention")
                or "unknown"
            )
            records.append(
                AnnotationRecord(
                    video_id=video_id,
                    pedestrian_id=pedestrian_id,
                    frame_id=frame_id,
                    xtl=xtl,
                    ytl=ytl,
                    width=width,
                    height=height,
                    behavior_label=behavior_label,
                    crossing_label=crossing_label,
                    source_file=str(path),
                    raw_attributes=attrs,
                )
            )
    return records


def parse_annotation_file(path: Path) -> List[AnnotationRecord]:
    root = ET.parse(path).getroot()
    video_id = _normalize_name(root.attrib.get("name") or root.attrib.get("video") or path.stem, path.stem)
    records = _parse_track_boxes(path, root, video_id)
    return [record for record in records if record.frame_id >= 0]


def discover_annotation_files(raw_dir: Path) -> List[Path]:
    candidates = []
    for path in raw_dir.rglob("*.xml"):
        if "annotation" in str(path).lower() or "annot" in str(path).lower():
            candidates.append(path)
    if candidates:
        return sorted(candidates)
    return sorted(raw_dir.rglob("*.xml"))


def prepare_jaad_source(raw_dir: Path, archive_path: Optional[Path], download_url: Optional[str]) -> Optional[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = raw_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    if archive_path is None and not download_url:
        return None

    resolved_archive = archive_path
    if download_url:
        filename = Path(urllib.parse.urlparse(download_url).path).name or "jaad_archive.zip"
        resolved_archive = archives_dir / filename
        if not resolved_archive.exists():
            urllib.request.urlretrieve(download_url, resolved_archive)

    if resolved_archive is None or not resolved_archive.exists():
        raise FileNotFoundError("JAAD archive was not found after download/extract preparation.")

    shutil.unpack_archive(str(resolved_archive), str(raw_dir))
    return resolved_archive


def load_records(raw_dir: Path) -> List[AnnotationRecord]:
    records: List[AnnotationRecord] = []
    for annotation_file in discover_annotation_files(raw_dir):
        records.extend(parse_annotation_file(annotation_file))
    return records


def clean_records(records: Sequence[AnnotationRecord]) -> List[AnnotationRecord]:
    cleaned: List[AnnotationRecord] = []
    dedupe_seen = set()
    for record in sorted(records, key=lambda item: (item.video_id, item.pedestrian_id, item.frame_id)):
        if record.width <= 1 or record.height <= 1:
            continue
        key = (record.video_id, record.pedestrian_id, record.frame_id)
        if key in dedupe_seen:
            continue
        dedupe_seen.add(key)
        cleaned.append(record)
    return cleaned


def _frame_number_from_name(path: Path) -> Optional[int]:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def build_frame_index(raw_dir: Path) -> Dict[str, Dict[int, Path]]:
    index: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for image_path in raw_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame_id = _frame_number_from_name(image_path)
        if frame_id is None:
            continue
        parts = image_path.parts
        video_candidates = [part for part in parts if part not in {"images", "clips", "frames", "raw", "jaad"}]
        video_id = _normalize_name(video_candidates[-2] if len(video_candidates) >= 2 else image_path.parent.name, image_path.parent.name)
        index[video_id][frame_id] = image_path
    return index


def split_continuous_runs(records: Sequence[AnnotationRecord]) -> List[List[AnnotationRecord]]:
    runs: List[List[AnnotationRecord]] = []
    current_run: List[AnnotationRecord] = []
    previous_frame: Optional[int] = None

    for record in records:
        if previous_frame is None or record.frame_id == previous_frame + 1:
            current_run.append(record)
        else:
            if current_run:
                runs.append(current_run)
            current_run = [record]
        previous_frame = record.frame_id

    if current_run:
        runs.append(current_run)
    return runs


def _majority_value(values: Iterable[str]) -> str:
    filtered = [value for value in values if value and value != "unknown"]
    if not filtered:
        return "unknown"
    return Counter(filtered).most_common(1)[0][0]


def _motion_direction(dx: float, dy: float) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _compute_sequence_features(window: Sequence[AnnotationRecord]) -> Dict[str, float | str]:
    centers = [(item.xtl + item.width / 2.0, item.ytl + item.height / 2.0) for item in window]
    areas = [item.width * item.height for item in window]
    deltas = []
    for first, second in zip(centers, centers[1:]):
        deltas.append((second[0] - first[0], second[1] - first[1]))
    speeds = [math.hypot(dx, dy) for dx, dy in deltas]

    total_dx = centers[-1][0] - centers[0][0] if len(centers) > 1 else 0.0
    total_dy = centers[-1][1] - centers[0][1] if len(centers) > 1 else 0.0
    speed_mean = sum(speeds) / len(speeds) if speeds else 0.0
    speed_var = sum((speed - speed_mean) ** 2 for speed in speeds) / len(speeds) if speeds else 0.0
    displacement = math.hypot(total_dx, total_dy)
    area_change = ((areas[-1] - areas[0]) / areas[0]) if areas and areas[0] else 0.0
    center_jitter = sum(
        math.hypot(center[0] - sum(point[0] for point in centers) / len(centers), center[1] - sum(point[1] for point in centers) / len(centers))
        for center in centers
    ) / len(centers)

    return {
        "speed_mean": round(speed_mean, 4),
        "speed_var": round(speed_var, 4),
        "displacement": round(displacement, 4),
        "bbox_scale_change": round(area_change, 4),
        "trajectory_length": round(sum(speeds), 4),
        "center_jitter": round(center_jitter, 4),
        "motion_direction": _motion_direction(total_dx, total_dy),
    }


def export_crop_sequence(
    frame_index: Dict[str, Dict[int, Path]],
    window: Sequence[AnnotationRecord],
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    video_frames = frame_index.get(window[0].video_id, {})
    for record in window:
        source_image = video_frames.get(record.frame_id)
        if source_image is None:
            continue
        with Image.open(source_image) as image:
            left = max(0, int(record.xtl))
            top = max(0, int(record.ytl))
            right = min(image.width, int(record.xtl + record.width))
            bottom = min(image.height, int(record.ytl + record.height))
            if right <= left or bottom <= top:
                continue
            crop = image.crop((left, top, right, bottom))
            crop.save(output_dir / f"frame_{record.frame_id:06d}.jpg")
            exported += 1
    return exported


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def export_dataset(
    records: Sequence[AnnotationRecord],
    raw_dir: Path,
    processed_dir: Path,
    window_size: int,
    stride: int,
    skip_crops: bool = False,
    max_sequences: Optional[int] = None,
) -> Dict[str, int]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame_index = {} if skip_crops else build_frame_index(raw_dir)

    manifest_rows: List[Dict[str, object]] = []
    track_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    cleaned_rows = [asdict(record) | {"raw_attributes": json.dumps(record.raw_attributes, ensure_ascii=False)} for record in records]

    by_actor: Dict[Tuple[str, str], List[AnnotationRecord]] = defaultdict(list)
    for record in records:
        by_actor[(record.video_id, record.pedestrian_id)].append(record)

    sequence_count = 0
    for (video_id, pedestrian_id), actor_records in sorted(by_actor.items()):
        runs = split_continuous_runs(sorted(actor_records, key=lambda item: item.frame_id))
        for run in runs:
            if len(run) < window_size:
                continue
            for start_index in range(0, len(run) - window_size + 1, stride):
                if max_sequences is not None and sequence_count >= max_sequences:
                    break
                window = run[start_index : start_index + window_size]
                sequence_id = f"{video_id}__{pedestrian_id}__{window[0].frame_id:06d}_{window[-1].frame_id:06d}"
                crop_dir = processed_dir / "crops" / sequence_id
                track_path = processed_dir / "tracks" / f"{sequence_id}.csv"

                exported_crops = 0
                if not skip_crops:
                    exported_crops = export_crop_sequence(frame_index, window, crop_dir)

                track_rows_for_sequence = []
                for record in window:
                    center_x = round(record.xtl + record.width / 2.0, 4)
                    center_y = round(record.ytl + record.height / 2.0, 4)
                    row = {
                        "sequence_id": sequence_id,
                        "video_id": record.video_id,
                        "pedestrian_id": record.pedestrian_id,
                        "frame_id": record.frame_id,
                        "xtl": round(record.xtl, 4),
                        "ytl": round(record.ytl, 4),
                        "width": round(record.width, 4),
                        "height": round(record.height, 4),
                        "center_x": center_x,
                        "center_y": center_y,
                        "behavior_label": record.behavior_label,
                        "crossing_label": record.crossing_label,
                    }
                    track_rows_for_sequence.append(row)
                track_rows.extend(track_rows_for_sequence)
                _write_csv(track_path, track_rows_for_sequence, track_rows_for_sequence[0].keys())

                features = _compute_sequence_features(window)
                feature_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "video_id": video_id,
                        "pedestrian_id": pedestrian_id,
                        **features,
                    }
                )

                manifest_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "video_id": video_id,
                        "pedestrian_id": pedestrian_id,
                        "start_frame": window[0].frame_id,
                        "end_frame": window[-1].frame_id,
                        "window_size": window_size,
                        "stride": stride,
                        "sequence_behavior_label": _majority_value(record.behavior_label for record in window),
                        "sequence_crossing_label": _majority_value(record.crossing_label for record in window),
                        "frame_count": len(window),
                        "crop_dir": str(crop_dir.relative_to(processed_dir)) if exported_crops else "",
                        "bbox_track_path": str(track_path.relative_to(processed_dir)),
                        "crops_exported": exported_crops,
                    }
                )
                sequence_count += 1
            if max_sequences is not None and sequence_count >= max_sequences:
                break
        if max_sequences is not None and sequence_count >= max_sequences:
            break

    _write_csv(
        processed_dir / "cleaned_metadata.csv",
        cleaned_rows,
        [
            "video_id",
            "pedestrian_id",
            "frame_id",
            "xtl",
            "ytl",
            "width",
            "height",
            "behavior_label",
            "crossing_label",
            "source_file",
            "raw_attributes",
        ],
    )
    if manifest_rows:
        _write_csv(processed_dir / "cleaned_sequences_manifest.csv", manifest_rows, manifest_rows[0].keys())
        _write_csv(processed_dir / "trajectory_features.csv", feature_rows, feature_rows[0].keys())
    else:
        _write_csv(
            processed_dir / "cleaned_sequences_manifest.csv",
            [],
            [
                "sequence_id",
                "video_id",
                "pedestrian_id",
                "start_frame",
                "end_frame",
                "window_size",
                "stride",
                "sequence_behavior_label",
                "sequence_crossing_label",
                "frame_count",
                "crop_dir",
                "bbox_track_path",
                "crops_exported",
            ],
        )
        _write_csv(
            processed_dir / "trajectory_features.csv",
            [],
            [
                "sequence_id",
                "video_id",
                "pedestrian_id",
                "speed_mean",
                "speed_var",
                "displacement",
                "bbox_scale_change",
                "trajectory_length",
                "center_jitter",
                "motion_direction",
            ],
        )

    _write_json(processed_dir / "sequence_index.json", manifest_rows)
    _write_json(
        processed_dir / "data_dictionary.json",
        {
            "cleaned_sequences_manifest": {
                "grain": "one row per sequence",
                "required_fields": [
                    "sequence_id",
                    "video_id",
                    "pedestrian_id",
                    "start_frame",
                    "end_frame",
                    "window_size",
                    "sequence_behavior_label",
                    "sequence_crossing_label",
                    "crop_dir",
                    "bbox_track_path",
                ],
            },
            "bbox_or_track_sequence": {
                "grain": "one row per frame within a sequence",
                "required_fields": [
                    "sequence_id",
                    "frame_id",
                    "xtl",
                    "ytl",
                    "width",
                    "height",
                    "center_x",
                    "center_y",
                ],
            },
            "trajectory_features": {
                "grain": "one row per sequence",
                "required_fields": [
                    "sequence_id",
                    "speed_mean",
                    "speed_var",
                    "displacement",
                    "bbox_scale_change",
                    "trajectory_length",
                    "center_jitter",
                    "motion_direction",
                ],
            },
        },
    )

    return {
        "cleaned_records": len(records),
        "sequences": len(manifest_rows),
        "tracks": len(track_rows),
        "features": len(feature_rows),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare JAAD data for Stage 1 training inputs and Stage 2 features.")
    parser.add_argument("--raw-dir", default="data/raw/jaad", help="Root directory containing JAAD annotations and frames.")
    parser.add_argument("--processed-dir", default="data/processed/jaad", help="Directory where processed artifacts will be stored.")
    parser.add_argument("--window-size", type=int, default=16, help="Fixed sequence length for sliding-window generation.")
    parser.add_argument("--stride", type=int, default=4, help="Stride for sliding-window generation.")
    parser.add_argument("--archive-path", type=Path, help="Optional local JAAD archive to extract into the raw directory.")
    parser.add_argument("--download-url", help="Optional direct JAAD archive URL to download before extraction.")
    parser.add_argument("--skip-crops", action="store_true", help="Skip crop export if image frames are unavailable.")
    parser.add_argument("--max-sequences", type=int, help="Optional cap for quick smoke tests.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    prepare_jaad_source(raw_dir, args.archive_path, args.download_url)
    records = clean_records(load_records(raw_dir))
    stats = export_dataset(
        records=records,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        window_size=args.window_size,
        stride=args.stride,
        skip_crops=args.skip_crops,
        max_sequences=args.max_sequences,
    )

    print(json.dumps({"raw_dir": str(raw_dir), "processed_dir": str(processed_dir), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
