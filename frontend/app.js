const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("analyzeForm");
const resultPanel = document.getElementById("resultsPanel");
const stage1Status = document.getElementById("stage1Status");
const stage2Status = document.getElementById("stage2Status");

function setStatuses(stage1, stage2) {
  stage1Status.className = `status ${stage1}`;
  stage2Status.className = `status ${stage2}`;

  stage1Status.textContent = stage1 === "done" ? "Completed" : stage1 === "running" ? "Running" : "Waiting";
  stage2Status.textContent = stage2 === "done" ? "Completed" : stage2 === "running" ? "Running" : "Waiting";
}

function getRecommendation(probability) {
  if (probability >= 0.7) {
    return "Immediate caution recommended: reduce speed and prepare to brake.";
  }
  if (probability >= 0.4) {
    return "Maintain controlled speed and closely monitor pedestrian movement.";
  }
  return "Low crossing intent detected. Continue standard monitoring.";
}

function renderResult(data, sourceLabel) {
  const probability = Number(data.stage2_risk.crossing_probability || 0);
  const probabilityPct = Math.round(probability * 100);
  const recommendation = getRecommendation(probability);
  const features = data.stage2_risk.feature_summary || {};

  resultPanel.innerHTML = `
    <h2>Current Assessment</h2>
    <p class="muted">Data Source: ${sourceLabel}</p>

    <div class="result-grid">
      <div class="kpi">
        <span>Behavior Label</span>
        <strong>${data.stage1_behavior.label}</strong>
      </div>
      <div class="kpi">
        <span>Behavior Confidence</span>
        <strong>${data.stage1_behavior.confidence}</strong>
      </div>
      <div class="kpi">
        <span>Risk Level</span>
        <strong>${data.stage2_risk.risk_level}</strong>
      </div>
    </div>

    <p class="result-line"><strong>Crossing Probability:</strong> ${probabilityPct}%</p>
    <div class="risk-track">
      <div class="risk-fill" style="width: ${probabilityPct}%;"></div>
    </div>

    <p class="result-line"><strong>Decision Guidance:</strong> ${recommendation}</p>

    <ul class="feature-list">
      <li>Trajectory Speed Norm: ${features.trajectory_speed_norm ?? "N/A"}</li>
      <li>Behavior Confidence Feature: ${features.behavior_confidence ?? "N/A"}</li>
      <li>Scene Context Flag: ${features.scene_context_flag ?? "N/A"}</li>
      <li>Request ID: ${data.request_id}</li>
    </ul>
  `;
}

function buildLocalDemoResponse(videoPath, pedestrianId, includeContext) {
  const base = Math.abs((videoPath + pedestrianId).length % 30) / 100;
  const crossingProbability = Math.min(0.92, Math.max(0.2, 0.52 + base + (includeContext ? 0.1 : -0.04)));

  let riskLevel = "Low";
  if (crossingProbability >= 0.7) {
    riskLevel = "High";
  } else if (crossingProbability >= 0.4) {
    riskLevel = "Medium";
  }

  return {
    request_id: `demo-${Date.now()}`,
    stage1_behavior: {
      label: crossingProbability > 0.65 ? "looking" : "standing",
      confidence: Number((0.72 + base).toFixed(2)),
      temporal_window: 16,
    },
    stage2_risk: {
      crossing_probability: Number(crossingProbability.toFixed(2)),
      risk_level: riskLevel,
      feature_summary: {
        trajectory_speed_norm: 0.52,
        behavior_confidence: Number((0.72 + base).toFixed(2)),
        scene_context_flag: includeContext ? 1 : 0,
      },
    },
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const videoPath = document.getElementById("videoPath").value.trim();
  const pedestrianId = document.getElementById("pedestrianId").value.trim() || "ped_001";
  const includeContext = document.getElementById("includeContext").checked;

  setStatuses("running", "waiting");

  const payload = {
    video_path: videoPath,
    pedestrian_id: pedestrianId,
    include_context: includeContext,
  };

  try {
    const response = await fetch(`${API_BASE}/api/v1/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API response status: ${response.status}`);
    }

    setStatuses("done", "running");
    const data = await response.json();
    setStatuses("done", "done");
    renderResult(data, "Backend Inference API");
  } catch (error) {
    const fallback = buildLocalDemoResponse(videoPath, pedestrianId, includeContext);
    setStatuses("done", "done");
    renderResult(fallback, "Local Demo Inference");
  }
});
