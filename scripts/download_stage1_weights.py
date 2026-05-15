from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/1IiszJk3zGOtEuLpUmqabveZRgALjpM6B"
REQUIRED_FILES = (
    "action_sequence_swin3d_t.pt",
    "look_frame_swin_t.pt",
    "person_detector.pt",
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Stage 1 weights from the team Google Drive folder into models/stage1."
    )
    parser.add_argument(
        "--folder-url",
        type=str,
        default=DEFAULT_FOLDER_URL,
        help="Public Google Drive folder URL containing the three Stage 1 weight files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/stage1"),
        help="Directory where the downloaded weight files will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite files even if they already exist locally.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce download output from gdown.",
    )
    return parser


def _import_gdown():
    try:
        import gdown
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "gdown is required to download the Google Drive folder.\n"
            "Install dependencies with `pip install -r requirements.txt` "
            "or `pip install gdown` and rerun the script."
        ) from exc
    return gdown


def _missing_required_files(output_dir: Path) -> list[str]:
    return [filename for filename in REQUIRED_FILES if not (output_dir / filename).is_file()]


def _print_prepared_files(output_dir: Path) -> None:
    for filename in REQUIRED_FILES:
        print(f"- {output_dir / filename}")


def _copy_required_files(download_root: Path, output_dir: Path, force: bool) -> list[Path]:
    copied: list[Path] = []
    for filename in REQUIRED_FILES:
        matches = list(download_root.rglob(filename))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {filename!r} in the downloaded Google Drive folder contents."
            )
        source = matches[0]
        destination = output_dir / filename
        if destination.exists() and not force:
            print(f"skip existing: {destination}")
            copied.append(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"saved: {destination}")
        copied.append(destination)
    return copied


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_files = _missing_required_files(args.output_dir)

    if not args.force and not missing_files:
        print(f"all required Stage 1 weights already exist in {args.output_dir}; skipping download")
        _print_prepared_files(args.output_dir)
        return

    if missing_files:
        print(f"missing Stage 1 weights: {', '.join(missing_files)}")

    gdown = _import_gdown()

    with tempfile.TemporaryDirectory(prefix="stage1_weights_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"downloading folder: {args.folder_url}")
        downloaded = gdown.download_folder(
            url=args.folder_url,
            output=str(tmp_path),
            quiet=args.quiet,
            use_cookies=False,
        )
        if not downloaded:
            raise SystemExit("No files were downloaded from the Google Drive folder.")
        copied = _copy_required_files(tmp_path, args.output_dir, force=args.force)

    print(f"done: prepared {len(copied)} files in {args.output_dir}")
    for path in copied:
        print(f"- {path}")


if __name__ == "__main__":
    main()
