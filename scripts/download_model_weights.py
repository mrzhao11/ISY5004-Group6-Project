from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


DEFAULT_STAGE1_FOLDER_URL = "https://drive.google.com/drive/folders/1IiszJk3zGOtEuLpUmqabveZRgALjpM6B"
DEFAULT_STAGE2_FOLDER_URL = "https://drive.google.com/drive/folders/1yLeYN9UsO-rwn8AHA9cNJdX08m-RuovX"

STAGE1_REQUIRED_FILES = (
    "action_sequence_swin3d_t.pt",
    "look_frame_swin_t.pt",
    "person_detector.pt",
)

STAGE2_REQUIRED_FILES = (
    "trajectory_motion_tcl.pt",
    "trajectory_motion_stage1_raw_logit_seed7_tcl.pt",
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download model weights from the team Google Drive folders into models/stage1 and models/stage2."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "stage1", "stage2"],
        default="all",
        help="Which model weight group to prepare.",
    )
    parser.add_argument(
        "--stage1-folder-url",
        type=str,
        default=DEFAULT_STAGE1_FOLDER_URL,
        help="Public Google Drive folder URL containing the three Stage 1 weight files.",
    )
    parser.add_argument(
        "--stage2-folder-url",
        type=str,
        default=DEFAULT_STAGE2_FOLDER_URL,
        help="Public Google Drive folder URL containing the two Stage 2 weight files.",
    )
    parser.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=Path("models/stage1"),
        help="Directory where Stage 1 weight files will be stored.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        type=Path,
        default=Path("models/stage2"),
        help="Directory where Stage 2 weight files will be stored.",
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


def _missing_required_files(output_dir: Path, required_files: tuple[str, ...]) -> list[str]:
    return [filename for filename in required_files if not (output_dir / filename).is_file()]


def _print_prepared_files(output_dir: Path, required_files: tuple[str, ...]) -> None:
    for filename in required_files:
        print(f"- {output_dir / filename}")


def _copy_required_files(
    download_root: Path,
    output_dir: Path,
    required_files: tuple[str, ...],
    force: bool,
) -> list[Path]:
    copied: list[Path] = []
    for filename in required_files:
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


def _prepare_weight_group(
    *,
    label: str,
    folder_url: str,
    output_dir: Path,
    required_files: tuple[str, ...],
    force: bool,
    quiet: bool,
    gdown=None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_files = _missing_required_files(output_dir, required_files)
    if not force and not missing_files:
        print(f"all required {label} weights already exist in {output_dir}; skipping download")
        _print_prepared_files(output_dir, required_files)
        return [output_dir / filename for filename in required_files]

    if missing_files:
        print(f"missing {label} weights: {', '.join(missing_files)}")
    if gdown is None:
        gdown = _import_gdown()

    with tempfile.TemporaryDirectory(prefix=f"{label.lower().replace(' ', '_')}_weights_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"downloading {label} folder: {folder_url}")
        downloaded = gdown.download_folder(
            url=folder_url,
            output=str(tmp_path),
            quiet=quiet,
            use_cookies=False,
        )
        if not downloaded:
            raise SystemExit(f"No files were downloaded from the {label} Google Drive folder.")
        copied = _copy_required_files(tmp_path, output_dir, required_files, force=force)

    print(f"done: prepared {len(copied)} {label} files in {output_dir}")
    for path in copied:
        print(f"- {path}")
    return copied


def main() -> None:
    args = build_argparser().parse_args()

    selected_groups = []
    if args.stage in {"all", "stage1"}:
        selected_groups.append(
            (
                "Stage 1",
                args.stage1_folder_url,
                args.stage1_output_dir,
                STAGE1_REQUIRED_FILES,
            )
        )
    if args.stage in {"all", "stage2"}:
        selected_groups.append(
            (
                "Stage 2",
                args.stage2_folder_url,
                args.stage2_output_dir,
                STAGE2_REQUIRED_FILES,
            )
        )

    needs_download = any(
        args.force or _missing_required_files(output_dir, required_files)
        for _, _, output_dir, required_files in selected_groups
    )
    gdown = _import_gdown() if needs_download else None

    prepared = []
    for label, folder_url, output_dir, required_files in selected_groups:
        prepared.extend(
            _prepare_weight_group(
                label=label,
                folder_url=folder_url,
                output_dir=output_dir,
                required_files=required_files,
                force=args.force,
                quiet=args.quiet,
                gdown=gdown,
            )
        )

    print(f"prepared {len(prepared)} model weight files")


if __name__ == "__main__":
    main()
