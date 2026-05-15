# JAAD Data Pipeline

This repository now includes a Stage 1-oriented JAAD preprocessing pipeline that turns raw annotations into sequence-level training inputs and trajectory features for Stage 2.

## Expected Raw Layout

The pipeline looks for JAAD assets under `data/raw/jaad` by default.

Recommended layout:

```text
data/raw/jaad/
  annotations/
    *.xml
  images/
    <video_id>/
      frame_000001.jpg
      frame_000002.jpg
```

You can also point the script to another root with `--raw-dir`.

## Outputs

The preprocessing step writes these artifacts to `data/processed/jaad`:

- `cleaned_metadata.csv`: cleaned frame-level pedestrian metadata
- `cleaned_sequences_manifest.csv`: one row per fixed-length pedestrian sequence
- `tracks/<sequence_id>.csv`: bbox and center-point sequence for each sequence
- `trajectory_features.csv`: one row per sequence with reusable trajectory features
- `sequence_index.json`: JSON index for downstream integrations
- `data_dictionary.json`: field-level contract for B/C/D handoff
- `crops/<sequence_id>/`: cropped pedestrian frames for CNN+LSTM input

## Commands

Prepare a mock JAAD-style sample:

```bash
python scripts/create_mock_jaad.py
```

Run the real preprocessing pipeline:

```bash
python scripts/prepare_jaad.py --raw-dir data/raw/jaad --processed-dir data/processed/jaad
```

Quick smoke test with the mock sample:

```bash
python scripts/prepare_jaad.py --raw-dir data/raw/mock_jaad --processed-dir data/processed/mock_jaad --window-size 8 --stride 4
python scripts/check_stage1_inputs.py data/processed/mock_jaad
python scripts/check_stage2_features.py data/processed/mock_jaad
```

If you only have annotations and want to defer crop export:

```bash
python scripts/prepare_jaad.py --skip-crops
```

If you already have a JAAD archive:

```bash
python scripts/prepare_jaad.py --archive-path path/to/jaad.zip
```

If you have a direct archive URL:

```bash
python scripts/prepare_jaad.py --download-url https://example.com/jaad.zip
```

## Sequence Contract

`cleaned_sequences_manifest.csv` contains the Stage 1 handoff contract:

- `sequence_id`
- `video_id`
- `pedestrian_id`
- `start_frame`
- `end_frame`
- `window_size`
- `stride`
- `sequence_behavior_label`
- `sequence_crossing_label`
- `frame_count`
- `crop_dir`
- `bbox_track_path`
- `crops_exported`

`tracks/<sequence_id>.csv` contains one row per frame:

- `sequence_id`
- `frame_id`
- `xtl`
- `ytl`
- `width`
- `height`
- `center_x`
- `center_y`
- `behavior_label`
- `crossing_label`

`trajectory_features.csv` contains one row per sequence:

- `sequence_id`
- `speed_mean`
- `speed_var`
- `displacement`
- `bbox_scale_change`
- `trajectory_length`
- `center_jitter`
- `motion_direction`

## Notes

- v1 uses fixed sliding windows to keep Stage 1 training inputs uniform.
- The parser is intentionally permissive for JAAD-style XML track annotations.
- When image frames are missing, the pipeline still exports manifest, tracks, and features.
