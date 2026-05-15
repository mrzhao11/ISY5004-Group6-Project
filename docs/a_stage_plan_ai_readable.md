# A Stage Implementation Plan

## Title

JAAD Data Processing and Stage 1 Training Input Preparation

## Goal

Build a stable preprocessing pipeline that converts raw JAAD data into standardized outputs for:

- Stage 1 behavior modeling with CNN + LSTM
- Stage 2 crossing-risk modeling with reusable trajectory features
- Team-wide integration through consistent file paths, field names, and sequence identifiers

This stage is responsible for data preparation only, not model training.

## Scope

Included:

- JAAD raw data intake and directory setup
- Annotation parsing and metadata cleaning
- Pedestrian fixed-length sequence generation
- Cropped pedestrian frame export
- Bounding-box / track sequence export
- Basic trajectory feature extraction
- Output contract documentation for B, C, and D

Not included:

- Stage 1 model training
- Stage 2 model training
- Scene-level advanced feature engineering
- Event-based variable-length sequence segmentation

## Assumptions

- Real JAAD data may not be present yet.
- The primary downstream consumer is Stage 1 training.
- Sequence output must include both image crops and bbox/coordinate sequences.
- Version 1 uses fixed sliding windows.
- Version 1 focuses on simple, robust, explainable trajectory features.

## Pipeline Overview

```text
Raw JAAD
-> annotation parsing
-> cleaned metadata
-> continuous pedestrian tracks
-> fixed sliding windows
-> sequence manifest + crop frames + track files
-> trajectory feature table
-> handoff to B / C / D
```

## Input Specification

Default raw data root:

`data/raw/jaad`

Expected structure:

```text
data/raw/jaad/
  annotations/
    *.xml
  images/
    <video_id>/
      frame_000001.jpg
      frame_000002.jpg
      ...
```

Minimum required source fields from JAAD annotations:

- `video_id`
- `pedestrian_id`
- `frame_id`
- bounding box coordinates
- behavior-related label
- crossing-related label
- optional raw attributes for traceability

## Processing Rules

### 1. Raw Data Preparation

- Place original JAAD assets under `data/raw/jaad`.
- If needed, support archive extraction or direct archive download before processing.
- Keep processed outputs under `data/processed/jaad`.

### 2. Cleaning Rules

- Remove rows with invalid or near-zero bounding boxes.
- Remove duplicated `(video_id, pedestrian_id, frame_id)` records.
- Normalize pedestrian IDs into stable machine-readable values.
- Preserve source annotation path and raw attributes for debugging.
- Only keep records that can participate in continuous pedestrian tracks.

### 3. Continuous Track Construction

- Group records by `video_id + pedestrian_id`.
- Sort by `frame_id`.
- Split tracks whenever frame continuity breaks.
- Only keep track segments with enough frames to form at least one full sequence window.

### 4. Sequence Generation

Use fixed sliding windows.

Default logic:

- `window_size`: fixed number of frames
- `stride`: fixed step between adjacent sequences
- each exported sequence must have the same length

Each sequence must produce:

- one manifest row
- one bbox/track CSV
- one crop directory containing cropped pedestrian frames
- one trajectory feature row

Sequence ID format:

`<video_id>__<pedestrian_id>__<start_frame>_<end_frame>`

## Output Contracts

### 1. Cleaned Sequence Manifest

File:

`data/processed/jaad/cleaned_sequences_manifest.csv`

Grain:

One row per sequence

Required fields:

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

Primary downstream use:

- B reads sequence-level training samples
- D uses it as the top-level integration index

### 2. BBox / Track Sequence

Files:

`data/processed/jaad/tracks/<sequence_id>.csv`

Grain:

One row per frame inside one exported sequence

Required fields:

- `sequence_id`
- `video_id`
- `pedestrian_id`
- `frame_id`
- `xtl`
- `ytl`
- `width`
- `height`
- `center_x`
- `center_y`
- `behavior_label`
- `crossing_label`

Primary downstream use:

- B can use coordinates as auxiliary inputs or explainability signals
- C can reuse the temporal track data if needed

### 3. Trajectory Features

File:

`data/processed/jaad/trajectory_features.csv`

Grain:

One row per sequence

Required fields:

- `sequence_id`
- `video_id`
- `pedestrian_id`
- `speed_mean`
- `speed_var`
- `displacement`
- `bbox_scale_change`
- `trajectory_length`
- `center_jitter`
- `motion_direction`

Primary downstream use:

- C uses these features for Stage 2 training and feature fusion

### 4. Supporting Outputs

- `data/processed/jaad/cleaned_metadata.csv`
- `data/processed/jaad/sequence_index.json`
- `data/processed/jaad/data_dictionary.json`
- `data/processed/jaad/crops/<sequence_id>/frame_*.jpg`

## Feature Definitions

Version 1 trajectory features should remain simple and explainable:

- `speed_mean`: average frame-to-frame movement magnitude
- `speed_var`: variance of frame-to-frame movement magnitude
- `displacement`: net movement from first frame center to last frame center
- `bbox_scale_change`: relative change in bbox area across the window
- `trajectory_length`: sum of stepwise movement magnitudes
- `center_jitter`: average deviation of center points from the window mean center
- `motion_direction`: coarse dominant direction inferred from net displacement

## Interface Responsibilities

### Handoff to B

Provide:

- sequence manifest
- cropped frame sequences
- per-frame bbox track files
- sequence-level labels and metadata

Expectation:

- B should be able to load fixed-length sequence samples directly for CNN + LSTM training

### Handoff to C

Provide:

- trajectory feature table keyed by `sequence_id`
- sequence time window identifiers
- pedestrian and video identifiers for later joins

Expectation:

- C should be able to merge trajectory features with behavior outputs and scene signals

### Handoff to D

Provide:

- data dictionary
- stable file paths
- stable naming rules
- stable sample granularity

Expectation:

- D should be able to integrate A/B/C outputs without redefining file formats

## Validation Plan

### Smoke Test

Use a small mock or subset sample to verify:

- annotations can be parsed
- cleaned metadata is generated
- sequences are produced with uniform length
- crops and tracks match the same pedestrian
- trajectory features are exported

### Consistency Checks

- every manifest row references a valid track file
- every sequence has the expected number of frames
- crop directories match manifest entries
- track files and feature rows share the same `sequence_id`

### Downstream Checks

- B can read sequence manifest plus crops as fixed-length inputs
- C can read `trajectory_features.csv` directly without extra reshaping

## Current Repo Mapping

Implementation entrypoints currently aligned to this plan:

- preprocessing logic: `backend/jaad_dataset.py`
- CLI entrypoint: `scripts/prepare_jaad.py`
- mock sample generator: `scripts/create_mock_jaad.py`
- B-side validation: `scripts/check_stage1_inputs.py`
- C-side validation: `scripts/check_stage2_features.py`
- interface documentation: `docs/jaad_data_pipeline.md`

## Suggested Execution Order

1. Prepare or download JAAD into `data/raw/jaad`
2. Run preprocessing to generate cleaned outputs
3. Inspect manifest, tracks, features, and crop directories
4. Let B validate fixed-length sequence loading
5. Let C validate direct feature-table loading
6. Freeze the output contract for team integration

## Success Criteria

This stage is complete when:

- raw JAAD can be converted into standardized processed outputs
- every exported sequence has a stable ID and fixed length
- B can consume the image sequences directly
- C can consume the trajectory features directly
- D can rely on the documented file contract for integration
