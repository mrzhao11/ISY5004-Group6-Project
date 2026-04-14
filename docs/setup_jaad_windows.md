# Windows Setup for JAAD Preprocessing

This guide provides the full command-line workflow for:

- creating a dedicated `conda` environment
- installing project dependencies
- validating the local preprocessing pipeline with mock data
- downloading JAAD annotations and clips
- extracting video frames
- generating processed outputs for Stage 1 and Stage 2 handoff

Environment context assumed by this guide:

- OS: Windows
- shell: PowerShell
- project path: `E:\文件\学习\硕士\项目三\ISY5004-Group6-Project`

Official references used for JAAD access:

- JAAD repository: `https://github.com/ykotseruba/JAAD`
- JAAD dataset page: `https://data.nvision2.eecs.yorku.ca/JAAD_dataset/`

## 1. Create a Conda Environment

```powershell
conda create -n isy5004-jaad python=3.11 -y
conda activate isy5004-jaad
```

If `conda activate` does not work in PowerShell:

```powershell
conda init powershell
```

Close the current PowerShell window, reopen it, then run:

```powershell
conda activate isy5004-jaad
```

## 2. Enter the Project and Install Dependencies

```powershell
cd "E:\文件\学习\硕士\项目三\ISY5004-Group6-Project"
pip install -r requirements.txt
pip install gdown
conda install -c conda-forge ffmpeg -y
```

## 3. Smoke Test the A-Stage Pipeline with Mock Data

```powershell
python scripts/create_mock_jaad.py
python scripts/prepare_jaad.py --raw-dir data/raw/mock_jaad --processed-dir data/processed/mock_jaad --window-size 8 --stride 4
python scripts/check_stage1_inputs.py data/processed/mock_jaad
python scripts/check_stage2_features.py data/processed/mock_jaad
```

## 4. Download JAAD Annotations

Clone the official JAAD 2.0 annotation repository:

```powershell
git clone -b JAAD_2.0 https://github.com/ykotseruba/JAAD.git data/raw/jaad_repo
```

Create the working directories for this project:

```powershell
New-Item -ItemType Directory -Force -Path data\raw\jaad\annotations | Out-Null
New-Item -ItemType Directory -Force -Path data\raw\jaad\images | Out-Null
New-Item -ItemType Directory -Force -Path data\raw\jaad\JAAD_clips | Out-Null
```

Copy annotations into the project data directory:

```powershell
Copy-Item -Recurse -Force data\raw\jaad_repo\annotations\* data\raw\jaad\annotations\
```

## 5. Download JAAD Video Clips

Try the official YorkU dataset link first:

```powershell
Invoke-WebRequest -Uri "https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip" -OutFile "data\raw\jaad\JAAD_clips.zip"
```

If the YorkU download fails, use the Google Drive mirror referenced by the official repository:

```powershell
gdown --fuzzy "https://drive.google.com/file/d/1HCFLBO9fJutCKG11FtjKfdLvME6Qe_5L/view?usp=drive_link" -O "data\raw\jaad\JAAD_clips.zip"
```

Extract the downloaded archive:

```powershell
Expand-Archive -Path "data\raw\jaad\JAAD_clips.zip" -DestinationPath "data\raw\jaad\JAAD_clips" -Force
```

## 6. Convert JAAD Video Clips into Frames

If clips are nested in subdirectories, use the recursive version below:

```powershell
Get-ChildItem "data\raw\jaad\JAAD_clips" -Recurse -Filter *.mp4 | ForEach-Object {
    $videoId = $_.BaseName
    $outDir = Join-Path "data\raw\jaad\images" $videoId
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    ffmpeg -i $_.FullName -start_number 1 (Join-Path $outDir "frame_%06d.jpg")
}
```

If the archive extracts all mp4 files directly into one folder, this shorter command also works:

```powershell
Get-ChildItem "data\raw\jaad\JAAD_clips" -Filter *.mp4 | ForEach-Object {
    $videoId = $_.BaseName
    $outDir = Join-Path "data\raw\jaad\images" $videoId
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    ffmpeg -i $_.FullName -start_number 1 (Join-Path $outDir "frame_%06d.jpg")
}
```

## 7. Run the Real JAAD Preprocessing Pipeline

```powershell
python scripts/prepare_jaad.py --raw-dir data/raw/jaad --processed-dir data/processed/jaad --window-size 16 --stride 4
```

## 8. Validate the Outputs

```powershell
python scripts/check_stage1_inputs.py data/processed/jaad
python scripts/check_stage2_features.py data/processed/jaad
```

## 9. Inspect the Output Artifacts

```powershell
Get-ChildItem data\processed\jaad
Get-ChildItem data\processed\jaad\tracks | Select-Object -First 5
Get-ChildItem data\processed\jaad\crops | Select-Object -First 5
```

Expected core outputs:

- `data/processed/jaad/cleaned_metadata.csv`
- `data/processed/jaad/cleaned_sequences_manifest.csv`
- `data/processed/jaad/trajectory_features.csv`
- `data/processed/jaad/tracks/`
- `data/processed/jaad/crops/`
- `data/processed/jaad/sequence_index.json`
- `data/processed/jaad/data_dictionary.json`

## 10. Full Command List

You can copy and run the following full workflow step by step:

```powershell
conda create -n isy5004-jaad python=3.11 -y
conda activate isy5004-jaad
cd "E:\文件\学习\硕士\项目三\ISY5004-Group6-Project"
pip install -r requirements.txt
pip install gdown
conda install -c conda-forge ffmpeg -y

python scripts/create_mock_jaad.py
python scripts/prepare_jaad.py --raw-dir data/raw/mock_jaad --processed-dir data/processed/mock_jaad --window-size 8 --stride 4
python scripts/check_stage1_inputs.py data/processed/mock_jaad
python scripts/check_stage2_features.py data/processed/mock_jaad

git clone -b JAAD_2.0 https://github.com/ykotseruba/JAAD.git data/raw/jaad_repo
New-Item -ItemType Directory -Force -Path data\raw\jaad\annotations | Out-Null
New-Item -ItemType Directory -Force -Path data\raw\jaad\images | Out-Null
New-Item -ItemType Directory -Force -Path data\raw\jaad\JAAD_clips | Out-Null
Copy-Item -Recurse -Force data\raw\jaad_repo\annotations\* data\raw\jaad\annotations\

Invoke-WebRequest -Uri "https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip" -OutFile "data\raw\jaad\JAAD_clips.zip"
# If the command above fails, use:
# gdown --fuzzy "https://drive.google.com/file/d/1HCFLBO9fJutCKG11FtjKfdLvME6Qe_5L/view?usp=drive_link" -O "data\raw\jaad\JAAD_clips.zip"

Expand-Archive -Path "data\raw\jaad\JAAD_clips.zip" -DestinationPath "data\raw\jaad\JAAD_clips" -Force

Get-ChildItem "data\raw\jaad\JAAD_clips" -Recurse -Filter *.mp4 | ForEach-Object {
    $videoId = $_.BaseName
    $outDir = Join-Path "data\raw\jaad\images" $videoId
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    ffmpeg -i $_.FullName -start_number 1 (Join-Path $outDir "frame_%06d.jpg")
}

python scripts/prepare_jaad.py --raw-dir data/raw/jaad --processed-dir data/processed/jaad --window-size 16 --stride 4
python scripts/check_stage1_inputs.py data/processed/jaad
python scripts/check_stage2_features.py data/processed/jaad
```

## 11. Notes

- These commands only change your local workspace unless you later commit and push them.
- If JAAD download links change in the future, check the official repository first.
- The preprocessing script can still run with annotations only by adding `--skip-crops`, but Stage 1 image training inputs require extracted frames.
