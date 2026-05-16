FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy<2.0" \
        opencv-python-headless \
        ultralytics
