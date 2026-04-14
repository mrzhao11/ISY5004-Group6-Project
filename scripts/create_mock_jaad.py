from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_mock_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)

    annotations_dir = root / "annotations"
    images_dir = root / "images" / "video_001"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    annotation = ET.Element("annotations", attrib={"name": "video_001"})
    track = ET.SubElement(annotation, "track", attrib={"id": "1", "label": "pedestrian"})

    for frame_id in range(1, 21):
        xtl = 20 + frame_id * 3
        ytl = 35 + frame_id
        xbr = xtl + 28
        ybr = ytl + 56

        box = ET.SubElement(
            track,
            "box",
            attrib={
                "frame": str(frame_id),
                "xtl": str(xtl),
                "ytl": str(ytl),
                "xbr": str(xbr),
                "ybr": str(ybr),
            },
        )
        ET.SubElement(box, "attribute", attrib={"name": "pedestrian_id"}).text = "ped_001"
        ET.SubElement(box, "attribute", attrib={"name": "action"}).text = "walking" if frame_id < 12 else "looking"
        ET.SubElement(box, "attribute", attrib={"name": "crossing"}).text = "crossing"

        image = Image.new("RGB", (160, 120), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle((xtl, ytl, xbr, ybr), outline=(220, 30, 30), width=3)
        draw.text((8, 8), f"frame {frame_id}", fill=(0, 0, 0))
        image.save(images_dir / f"frame_{frame_id:06d}.jpg")

    tree = ET.ElementTree(annotation)
    tree.write(annotations_dir / "video_001_annotations.xml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    target = PROJECT_ROOT / "data" / "raw" / "mock_jaad"
    build_mock_dataset(target)
    print(target)
