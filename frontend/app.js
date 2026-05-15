const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("analyzeForm");
const resultPanel = document.getElementById("resultsPanel");
const stage1Status = document.getElementById("stage1Status");
const stage2Status = document.getElementById("stage2Status");
const demoVideoSelect = document.getElementById("demoVideoSelect");
const videoFileInput = document.getElementById("videoFile");
const filePickerButton = document.getElementById("filePickerButton");
const selectedFileName = document.getElementById("selectedFileName");
const rawVideoPreview = document.getElementById("rawVideoPreview");
const rawVideoCaption = document.getElementById("rawVideoCaption");
let uploadedVideoUrl = null;

function setStatuses(stage1, stage2) {
  stage1Status.className = `status ${stage1}`;
  stage2Status.className = `status ${stage2}`;

  stage1Status.textContent = stage1 === "done" ? "Completed" : stage1 === "running" ? "Running" : "Waiting";
  stage2Status.textContent =
    stage2 === "done" ? "Completed" : stage2 === "running" ? "Running" : "Waiting";
}

function renderResult(data, sourceLabel) {
  const features = data.stage2_risk.feature_summary || {};
  const windows = data.stage1_behavior.windows || [];
  const primary = data.stage1_behavior.primary_window || null;
  const previews = data.stage1_behavior.crop_previews || [];
  const windowsMarkup = windows
    .slice(0, 6)
    .map(
      (window) => `
        <li>
          Window ${window.window_index}: frames ${window.start_frame}-${window.end_frame},
          action ${window.action_label} (${Number(window.action_confidence).toFixed(2)}),
          look ${window.look_label} (${Number(window.look_confidence).toFixed(2)})
        </li>
      `
    )
    .join("");
  const previewMarkup = previews
    .map(
      (preview) => `
        <figure class="crop-preview">
          <img src="${preview.image_data}" alt="Extracted pedestrian crop from frame ${preview.frame_id}" />
          <figcaption>Frame ${preview.frame_id}</figcaption>
        </figure>
      `
    )
    .join("");
  const crossingProbability =
    data.stage2_risk.crossing_probability === null || data.stage2_risk.crossing_probability === undefined
      ? "N/A"
      : Number(data.stage2_risk.crossing_probability).toFixed(3);
  const riskLevel = data.stage2_risk.risk_level || "N/A";
  const riskClass = riskLevel.toLowerCase() === "high" ? "risk-high" : "risk-low";

  resultPanel.innerHTML = `
    <h2>Analysis Output</h2>
    <p class="muted">Data Source: ${sourceLabel}</p>

    <div class="result-grid">
      <div class="kpi">
        <span>Action Label</span>
        <strong>${data.stage1_behavior.action_label}</strong>
      </div>
      <div class="kpi">
        <span>Look Label</span>
        <strong>${data.stage1_behavior.look_label}</strong>
      </div>
      <div class="kpi">
        <span>Pedestrian ID</span>
        <strong>${data.stage1_behavior.pedestrian_id}</strong>
      </div>
      <div class="kpi">
        <span>Crossing Risk</span>
        <strong class="${riskClass}">${riskLevel}</strong>
      </div>
      <div class="kpi">
        <span>Crossing Probability</span>
        <strong>${crossingProbability}</strong>
      </div>
    </div>

    <p class="result-line"><strong>Action Confidence:</strong> ${Number(data.stage1_behavior.action_confidence || 0).toFixed(3)}</p>
    <p class="result-line"><strong>Look Confidence:</strong> ${Number(data.stage1_behavior.look_confidence || 0).toFixed(3)}</p>
    <p class="result-line"><strong>Windows Analyzed:</strong> ${data.stage1_behavior.windows_analyzed} (track frames: ${data.stage1_behavior.track_frame_count})</p>
    ${
      primary
        ? `<p class="result-line"><strong>Primary Window:</strong> #${primary.window_index}, frames ${primary.start_frame}-${primary.end_frame}</p>`
        : ""
    }

    <h3>Crossing Prediction</h3>
    <p class="result-line"><strong>Status:</strong> ${data.stage2_risk.status}</p>
    <p class="result-line"><strong>Message:</strong> ${data.stage2_risk.message}</p>

    <h3>Extracted Pedestrian Crops</h3>
    <div class="preview-grid">${previewMarkup || '<p class="muted">No crop previews returned.</p>'}</div>

    <ul class="feature-list">
      <li>Action Confidence Feature: ${features.action_confidence ?? "N/A"}</li>
      <li>Look Confidence Feature: ${features.look_confidence ?? "N/A"}</li>
      <li>Mean Base Crossing Probability: ${features.mean_base_prob_crossing ?? "N/A"}</li>
      <li>Mean Behavior-Aux Crossing Probability: ${features.mean_stage1_aux_prob_crossing ?? "N/A"}</li>
      <li>Stage 2 Windows Analyzed: ${features.stage2_windows_analyzed ?? "N/A"}</li>
    </ul>

    <h3>Window Summaries</h3>
    <ul class="feature-list">${windowsMarkup || "<li>No windows returned.</li>"}</ul>
  `;
}

function setRawVideoPreview(src, caption) {
  rawVideoPreview.src = src;
  rawVideoCaption.textContent = caption;
  rawVideoPreview.load();
}

function clearUploadedVideoUrl() {
  if (uploadedVideoUrl) {
    URL.revokeObjectURL(uploadedVideoUrl);
    uploadedVideoUrl = null;
  }
}

function previewDemoVideo(filename) {
  if (!filename) {
    setRawVideoPreview("", "Select a demo video or upload a local video to preview the raw input.");
    return;
  }
  clearUploadedVideoUrl();
  setRawVideoPreview(
    `${API_BASE}/api/v1/demo-videos/${encodeURIComponent(filename)}`,
    `Previewing demo video: ${filename}`
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const videoFile = videoFileInput.files[0];
  const demoFilename = demoVideoSelect.value;
  if (!videoFile && !demoFilename) {
    resultPanel.innerHTML = '<h2>Analysis Output</h2><p class="muted">Please select a demo video or upload a video file first.</p>';
    return;
  }

  setStatuses("running", "waiting");

  try {
    let response;
    let sourceLabel;
    if (videoFile) {
      const payload = new FormData();
      payload.append("video", videoFile);
      response = await fetch(`${API_BASE}/api/v1/analyze-video`, {
        method: "POST",
        body: payload,
      });
      sourceLabel = "Uploaded Video";
    } else {
      response = await fetch(`${API_BASE}/api/v1/analyze-demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: demoFilename }),
      });
      sourceLabel = `Demo Video: ${demoFilename}`;
    }

    if (!response.ok) {
      throw new Error(`API response status: ${response.status}`);
    }

    setStatuses("done", "running");
    const data = await response.json();
    setStatuses("done", data.stage2_risk.status === "completed" ? "done" : "waiting");
    renderResult(data, sourceLabel);
  } catch (error) {
    setStatuses("done", "done");
    resultPanel.innerHTML = `
      <h2>Analysis Output</h2>
      <p class="muted">The backend could not complete video inference. Check that Docker is running, weights are downloaded, and the uploaded video is valid.</p>
    `;
  }
});

videoFileInput.addEventListener("change", () => {
  const file = videoFileInput.files[0];
  selectedFileName.textContent = file?.name || "No file selected";
  if (!file) {
    previewDemoVideo(demoVideoSelect.value);
    return;
  }
  clearUploadedVideoUrl();
  uploadedVideoUrl = URL.createObjectURL(file);
  setRawVideoPreview(uploadedVideoUrl, `Previewing uploaded video: ${file.name}`);
});

filePickerButton.addEventListener("click", () => {
  videoFileInput.click();
});

demoVideoSelect.addEventListener("change", () => {
  if (!videoFileInput.files[0]) {
    previewDemoVideo(demoVideoSelect.value);
  }
});

async function loadDemoVideos() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/demo-videos`);
    if (!response.ok) {
      throw new Error(`API response status: ${response.status}`);
    }
    const videos = await response.json();
    if (!videos.length) {
      demoVideoSelect.innerHTML = '<option value="">No demo videos available</option>';
      return;
    }
    demoVideoSelect.innerHTML = videos
      .map((video) => `<option value="${video.filename}">${video.label}</option>`)
      .join("");
    previewDemoVideo(videos[0].filename);
  } catch (error) {
    demoVideoSelect.innerHTML = '<option value="">Demo videos unavailable</option>';
  }
}

loadDemoVideos();
