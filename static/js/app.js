/**
 * AI Crop Recommendation System — Frontend Logic
 * Handles: form submission, charts, chatbot, voice, presets
 */

"use strict";

// ── Crop Icons (FA class) & Colors ───────────────────────────────────────────
const CROP_ICON = {
  wheat:      { icon: "fa-wheat-awn",   bg: "#92400e" },
  rice:       { icon: "fa-bowl-rice",   bg: "#1e40af" },
  cotton:     { icon: "fa-cloud",       bg: "#374151" },
  sugarcane:  { icon: "fa-fire",        bg: "#991b1b" },
  maize:      { icon: "fa-sun",         bg: "#854d0e" },
  barley:     { icon: "fa-seedling",    bg: "#166534" },
  pulses:     { icon: "fa-circle-dot",  bg: "#5b21b6" },
  vegetables: { icon: "fa-leaf",        bg: "#065f46" },
  fruits:     { icon: "fa-apple-whole", bg: "#9f1239" },
};

const CROP_COLORS = {
  wheat:      "#d97706",
  rice:       "#1d4ed8",
  cotton:     "#374151",
  sugarcane:  "#dc2626",
  maize:      "#ca8a04",
  barley:     "#16a34a",
  pulses:     "#7c3aed",
  vegetables: "#0f766e",
  fruits:     "#be123c",
};

// ── Presets ──────────────────────────────────────────────────────────────────
const PRESETS = {
  wheat:      { temp:22, humidity:55, rainfall:60,  ph:6.5, moisture:40, N:80,  P:40, K:40 },
  rice:       { temp:28, humidity:82, rainfall:200, ph:6.0, moisture:75, N:90,  P:45, K:45 },
  cotton:     { temp:30, humidity:60, rainfall:80,  ph:7.0, moisture:45, N:120, P:50, K:50 },
  sugarcane:  { temp:32, humidity:75, rainfall:180, ph:6.5, moisture:70, N:100, P:50, K:50 },
  maize:      { temp:26, humidity:65, rainfall:100, ph:6.2, moisture:55, N:85,  P:45, K:45 },
  barley:     { temp:18, humidity:50, rainfall:50,  ph:6.8, moisture:35, N:70,  P:35, K:35 },
  pulses:     { temp:24, humidity:58, rainfall:70,  ph:6.3, moisture:42, N:20,  P:60, K:80 },
  vegetables: { temp:25, humidity:70, rainfall:120, ph:6.4, moisture:60, N:110, P:55, K:55 },
  fruits:     { temp:27, humidity:72, rainfall:150, ph:6.6, moisture:65, N:95,  P:50, K:50 },
};

// ── Chart instances ──────────────────────────────────────────────────────────
let suitabilityChart = null, radarChart = null, elbowChart = null, distChart = null;
let voiceEnabled = true;

// ── Fill Preset ──────────────────────────────────────────────────────────────
function fillPreset(crop) {
  const p = PRESETS[crop];
  if (!p) return;
  Object.entries(p).forEach(([k, v]) => {
    const el = document.getElementById(k);
    if (el) el.value = v;
  });
  updateWeatherCards(p);
}

// ── Update Weather Cards ─────────────────────────────────────────────────────
function updateWeatherCards(d) {
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? "—";
  };
  set("wc-temp",  d.temp     ?? "—");
  set("wc-humid", d.humidity ?? "—");
  set("wc-rain",  d.rainfall ?? "—");
  set("wc-ph",    d.ph       ?? "—");
}

// ── Reset Form ───────────────────────────────────────────────────────────────
function resetForm() {
  document.getElementById("cropForm").reset();
  updateWeatherCards({});
  document.getElementById("results-section").classList.add("d-none");
}

// ── Form Submit ──────────────────────────────────────────────────────────────
document.getElementById("cropForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = {
    temp:     parseFloat(document.getElementById("temp").value),
    humidity: parseFloat(document.getElementById("humidity").value),
    rainfall: parseFloat(document.getElementById("rainfall").value),
    ph:       parseFloat(document.getElementById("ph").value),
    moisture: parseFloat(document.getElementById("moisture").value),
    N:        parseFloat(document.getElementById("N").value),
    P:        parseFloat(document.getElementById("P").value),
    K:        parseFloat(document.getElementById("K").value),
  };
  updateWeatherCards(data);
  showLoading(true);

  try {
    const res  = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const json = await res.json();
    if (json.status === "success") {
      renderResults(json.data);
    } else {
      alert("Error: " + json.message);
    }
  } catch (err) {
    alert("Network error: " + err.message);
  } finally {
    showLoading(false);
  }
});

// ── Render Results ───────────────────────────────────────────────────────────
function renderResults(d) {
  const section = document.getElementById("results-section");
  section.classList.remove("d-none");
  section.classList.add("fade-in");

  const crop     = d.recommended_crop;
  const cropData = CROP_ICON[crop] || { icon: "fa-seedling", bg: "#2d6a4f" };
  const label    = crop.charAt(0).toUpperCase() + crop.slice(1);

  // Hero — icon wrap with crop color
  const wrap = document.getElementById("res-icon-wrap");
  wrap.style.background = cropData.bg + "33";
  wrap.style.borderColor = cropData.bg + "66";
  const icon = document.getElementById("res-icon");
  icon.className = `fas ${cropData.icon} fa-3x`;
  icon.style.color = "white";

  document.getElementById("res-crop").textContent    = label;
  document.getElementById("res-cluster").textContent = `Cluster ${d.cluster_id} — K-Means Assignment`;
  document.getElementById("res-yield").innerHTML     =
    `<span class="badge" style="background:rgba(255,255,255,0.2);color:#fde68a;font-size:0.85rem;padding:6px 14px;border-radius:20px;">
      <i class="fas fa-box me-1"></i>Expected Yield: ${d.yield_estimate}
    </span>`;

  // Suitability ring
  const score = Math.min(d.suitability_score, 99.9);
  document.getElementById("res-score").textContent = score + "%";
  const ring = document.getElementById("suitabilityRing");
  const circumference = 2 * Math.PI * 50;
  ring.style.strokeDashoffset = circumference - (score / 100) * circumference;
  ring.style.transition = "stroke-dashoffset 1.2s ease";

  // Info cards
  document.getElementById("res-fertilizer").textContent = d.fertilizer;
  document.getElementById("res-irrigation").textContent  = d.irrigation;
  document.getElementById("res-season").textContent      = d.seasonal_tip;

  // Alternatives — colored icon badges
  const altDiv = document.getElementById("res-alternatives");
  altDiv.innerHTML = d.alternatives.map(a => {
    const ci = CROP_ICON[a] || { icon: "fa-seedling", bg: "#2d6a4f" };
    return `<span class="d-inline-flex align-items-center gap-2 px-3 py-2 rounded-pill fw-semibold"
              style="background:${ci.bg};color:white;font-size:0.85rem;">
      <i class="fas ${ci.icon}"></i>${a.charAt(0).toUpperCase() + a.slice(1)}
    </span>`;
  }).join("");

  // Reasons — icon-based status indicators
  const reasonsDiv = document.getElementById("res-reasons");
  reasonsDiv.innerHTML = d.reasons.map(r => {
    let cls, iconHtml;
    if (r.startsWith("OK ")) {
      cls = "reason-ok";
      iconHtml = `<i class="fas fa-circle-check reason-icon"></i>`;
      r = r.slice(3);
    } else if (r.startsWith("WARN ")) {
      cls = "reason-warn";
      iconHtml = `<i class="fas fa-triangle-exclamation reason-icon"></i>`;
      r = r.slice(5);
    } else {
      cls = "reason-bad";
      iconHtml = `<i class="fas fa-circle-xmark reason-icon"></i>`;
      r = r.startsWith("BAD ") ? r.slice(4) : r;
    }
    return `<div class="reason-item ${cls}">${iconHtml}<span>${r}</span></div>`;
  }).join("");

  // Charts
  renderSuitabilityChart(d.all_scores);
  renderRadarChart(d.input);

  // Varieties for this crop
  renderResultVarieties(d.varieties || [], crop);

  // Voice
  if (voiceEnabled) {
    speak(`Recommendation: ${label}. Suitability score: ${score} percent. ${d.fertilizer.split(".")[0]}.`);
  }

  setTimeout(() => section.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

// ── Suitability Bar Chart ────────────────────────────────────────────────────
function renderSuitabilityChart(scores) {
  const labels = Object.keys(scores).map(c => c.charAt(0).toUpperCase() + c.slice(1));
  const values = Object.values(scores);
  const colors = Object.keys(scores).map(c => CROP_COLORS[c] || "#64748b");

  if (suitabilityChart) suitabilityChart.destroy();
  suitabilityChart = new Chart(document.getElementById("suitabilityChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Suitability %",
        data: values,
        backgroundColor: colors,
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } }
      },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: v => v + "%" }, grid: { color: "#e8f5e9" } },
        x: { grid: { display: false } }
      }
    }
  });
}

// ── Radar Chart ──────────────────────────────────────────────────────────────
function renderRadarChart(input) {
  const labels  = ["Temp", "Humidity", "Rainfall", "pH", "Moisture", "N", "P", "K"];
  const maxVals = [50, 100, 500, 14, 100, 300, 200, 200];
  const userVals = [input.temp, input.humidity, input.rainfall, input.ph,
                    input.moisture, input.N, input.P, input.K]
                   .map((v, i) => Math.min(100, (v / maxVals[i]) * 100));

  if (radarChart) radarChart.destroy();
  radarChart = new Chart(document.getElementById("radarChart"), {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "Your Conditions",
        data: userVals,
        backgroundColor: "rgba(15,118,110,0.15)",
        borderColor: "#0f766e",
        borderWidth: 2.5,
        pointBackgroundColor: "#0f766e",
        pointRadius: 4,
      }]
    },
    options: {
      responsive: true,
      scales: { r: { beginAtZero: true, max: 100, ticks: { display: false }, grid: { color: "#e2e8f0" } } },
      plugins: { legend: { display: false } }
    }
  });
}

// ── Dashboard ────────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const [elbowRes, statsRes, clusterRes] = await Promise.all([
      fetch("/api/elbow").then(r => r.json()),
      fetch("/api/dataset-stats").then(r => r.json()),
      fetch("/api/clusters").then(r => r.json()),
    ]);
    if (elbowRes.status   === "success") renderElbowChart(elbowRes.data);
    if (statsRes.status   === "success") renderDistChart(statsRes.data);
    if (clusterRes.status === "success") renderClusterTable(clusterRes.data);
  } catch (err) {
    console.error("Dashboard load error:", err);
  }
}

function renderElbowChart(data) {
  if (elbowChart) elbowChart.destroy();
  elbowChart = new Chart(document.getElementById("elbowChart"), {
    type: "line",
    data: {
      labels: data.k,
      datasets: [{
        label: "Inertia (WCSS)",
        data: data.inertia,
        borderColor: "#0e7490",
        backgroundColor: "rgba(14,116,144,0.1)",
        borderWidth: 2.5,
        pointBackgroundColor: "#0e7490",
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Number of Clusters (K)" } },
        y: { title: { display: true, text: "Inertia" }, grid: { color: "#e2e8f0" } }
      }
    }
  });
}

function renderDistChart(data) {
  const crops  = Object.keys(data.crops);
  const counts = Object.values(data.crops);
  const colors = crops.map(c => CROP_COLORS[c] || "#64748b");

  if (distChart) distChart.destroy();
  distChart = new Chart(document.getElementById("distChart"), {
    type: "doughnut",
    data: {
      labels: crops.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets: [{ data: counts, backgroundColor: colors, borderWidth: 2, borderColor: "#fff" }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "right", labels: { font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} samples` } }
      }
    }
  });
}

function renderClusterTable(clusters) {
  const tbody = document.getElementById("clusterTableBody");
  tbody.innerHTML = clusters.map(c => {
    const p  = c.profile;
    const ci = CROP_ICON[c.crop] || { icon: "fa-seedling", bg: "#2d6a4f" };
    const label = c.crop.charAt(0).toUpperCase() + c.crop.slice(1);
    return `<tr>
      <td><span class="badge" style="background:#334155;color:white;">${c.cluster}</span></td>
      <td>
        <span class="d-inline-flex align-items-center gap-2">
          <span style="width:28px;height:28px;border-radius:8px;background:${ci.bg};
                       display:inline-flex;align-items:center;justify-content:center;">
            <i class="fas ${ci.icon}" style="color:white;font-size:0.75rem;"></i>
          </span>
          <strong>${label}</strong>
        </span>
      </td>
      <td>${p.temp}</td><td>${p.humidity}</td><td>${p.rainfall}</td>
      <td>${p.ph}</td><td>${p.moisture}</td>
      <td>${p.N}</td><td>${p.P}</td><td>${p.K}</td>
    </tr>`;
  }).join("");
}

// ── Chatbot ──────────────────────────────────────────────────────────────────
const CHAT_KB = {
  "wheat":      "Wheat grows best at 20-25 degrees C with moderate rainfall (50-75mm). Sow in Oct-Nov (Rabi season).",
  "rice":       "Rice needs high humidity (80%+) and 150-250mm rainfall. Transplant in June-July.",
  "cotton":     "Cotton prefers 28-35 degrees C and well-drained soil. Use drip irrigation for best results.",
  "sugarcane":  "Sugarcane needs 25-35 degrees C and 150-200mm rainfall. Takes 12 months to mature.",
  "maize":      "Maize grows in 20-30 degrees C with 80-120mm rainfall. Short duration crop (90-110 days).",
  "fertilizer": "Use NPK fertilizers based on soil test. Urea for N, DAP for P, MOP for K.",
  "irrigation": "Drip irrigation saves 40-60% water. Irrigate at critical crop growth stages.",
  "ph":         "Ideal soil pH is 6.0-7.0 for most crops. Add lime to raise pH, sulfur to lower it.",
  "npk":        "N = Nitrogen (leaf growth), P = Phosphorus (root/flower), K = Potassium (fruit/disease resistance).",
  "kmeans":     "K-Means groups similar soil/weather conditions into clusters. Each cluster maps to a crop type.",
  "cluster":    "K-Means groups similar soil/weather conditions into clusters. Each cluster maps to a crop type.",
  "disease":    "Common diseases: Wheat Rust, Rice Blast, Cotton Bollworm. Use resistant varieties and fungicides.",
  "soil":       "Healthy soil has good structure, pH 6-7, adequate NPK, and organic matter above 2%.",
  "season":     "Kharif crops: Jun-Nov (rice, cotton, maize). Rabi crops: Oct-Mar (wheat, barley, pulses).",
  "yield":      "Yield depends on variety, soil health, irrigation, and pest management. Use certified seeds.",
  "hello":      "Hello! I am AgriBot. Ask me about crops, soil, fertilizers, or irrigation!",
  "hi":         "Hi there! How can I help you with farming today?",
  "help":       "I can answer questions about: crops, soil, fertilizers, irrigation, pH, NPK, seasons, diseases.",
};

function toggleChat() {
  document.getElementById("chatWindow").classList.toggle("d-none");
}

function sendChat() {
  const input = document.getElementById("chatInput");
  const msg   = input.value.trim();
  if (!msg) return;

  appendChat(msg, "user");
  input.value = "";

  const lower = msg.toLowerCase();
  let reply = "I am not sure about that. Try asking about: crops, soil, fertilizer, irrigation, pH, NPK, or seasons.";

  for (const [key, ans] of Object.entries(CHAT_KB)) {
    if (lower.includes(key)) { reply = ans; break; }
  }

  setTimeout(() => {
    appendChat(reply, "bot");
    if (voiceEnabled) speak(reply);
  }, 400);
}

function appendChat(msg, type) {
  const div = document.createElement("div");
  div.className = `chat-msg ${type}`;
  div.textContent = msg;
  const container = document.getElementById("chatMessages");
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// ── Voice Assistant ──────────────────────────────────────────────────────────
function speak(text) {
  if (!voiceEnabled || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 0.95; utt.pitch = 1.0; utt.volume = 0.9;
  window.speechSynthesis.speak(utt);
}

function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  document.getElementById("voiceLabel").textContent = voiceEnabled ? "Voice ON" : "Voice OFF";
  if (!voiceEnabled) window.speechSynthesis?.cancel();
}

// ── Loading ──────────────────────────────────────────────────────────────────
function showLoading(show) {
  document.getElementById("loadingOverlay").classList.toggle("d-none", !show);
}

// ── Live input → weather cards ───────────────────────────────────────────────
["temp", "humidity", "rainfall", "ph"].forEach(id => {
  document.getElementById(id)?.addEventListener("input", () => {
    updateWeatherCards({
      temp:     document.getElementById("temp")?.value,
      humidity: document.getElementById("humidity")?.value,
      rainfall: document.getElementById("rainfall")?.value,
      ph:       document.getElementById("ph")?.value,
    });
  });
});

// ── Auto-load dashboard on page load ────────────────────────────────────────
window.addEventListener("load", () => {
  loadDashboard();
});

// ── Variety Guide ────────────────────────────────────────────────────────────
let allVarieties = {};

const VARIETY_ICON_MAP = {
  vegetables: { icon: "fa-leaf",        bg: "#065f46" },
  pulses:     { icon: "fa-circle-dot",  bg: "#5b21b6" },
  fruits:     { icon: "fa-apple-whole", bg: "#9f1239" },
  wheat:      { icon: "fa-wheat-awn",   bg: "#92400e" },
  rice:       { icon: "fa-bowl-rice",   bg: "#1e40af" },
  maize:      { icon: "fa-sun",         bg: "#854d0e" },
  cotton:     { icon: "fa-cloud",       bg: "#374151" },
  sugarcane:  { icon: "fa-fire",        bg: "#991b1b" },
  barley:     { icon: "fa-seedling",    bg: "#166534" },
};

async function loadVarietyGuide() {
  try {
    const res  = await fetch("/api/varieties");
    const json = await res.json();
    if (json.status === "success") {
      allVarieties = json.data;
      showGuideTab("vegetables");
    }
  } catch (err) {
    console.error("Variety guide load error:", err);
  }
}

function showGuideTab(crop) {
  // Update active tab
  document.querySelectorAll(".guide-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.crop === crop);
  });
  const container = document.getElementById("guideCardsContainer");
  const varieties = allVarieties[crop] || [];
  container.innerHTML = varieties.length
    ? varieties.map(v => buildVarietyCard(v, crop)).join("")
    : `<div class="col-12 text-muted text-center py-4">No variety data available for this crop.</div>`;
}

function buildVarietyCard(v, crop) {
  const ci = VARIETY_ICON_MAP[crop] || { icon: "fa-seedling", bg: "#2d6a4f" };
  const rows = [
    { cls: "vri-soil",      icon: "fa-mountain",       label: "Soil",       val: v.soil },
    { cls: "vri-temp",      icon: "fa-thermometer-half",label: "Temp",      val: v.temp_range },
    { cls: "vri-water",     icon: "fa-droplet",         label: "Water",     val: v.water_need },
    { cls: "vri-fert",      icon: "fa-flask",           label: "Fertilizer",val: v.fertilizer },
    { cls: "vri-duration",  icon: "fa-clock",           label: "Duration",  val: v.duration },
    { cls: "vri-yield",     icon: "fa-box",             label: "Yield",     val: v.yield },
    { cls: "vri-disease",   icon: "fa-bug",             label: "Disease",   val: v.disease },
    { cls: "vri-companion", icon: "fa-handshake",       label: "Companion", val: v.companion },
    { cls: "vri-market",    icon: "fa-store",           label: "Market",    val: v.market },
  ];
  return `
    <div class="col-md-6 col-lg-4">
      <div class="variety-card">
        <div class="variety-card-header">
          <div class="variety-icon" style="background:${ci.bg}">
            <i class="fas ${ci.icon}"></i>
          </div>
          <div>
            <div class="variety-name">${v.name}</div>
            <div class="variety-season"><i class="fas fa-calendar-days me-1"></i>${v.best_season}</div>
          </div>
        </div>
        <div class="variety-card-body">
          ${rows.map(r => `
            <div class="variety-row">
              <div class="variety-row-icon ${r.cls}"><i class="fas ${r.icon}"></i></div>
              <div><span class="variety-row-label">${r.label}: </span><span class="variety-row-val">${r.val}</span></div>
            </div>`).join("")}
        </div>
      </div>
    </div>`;
}

// Show varieties in results section after recommendation
function renderResultVarieties(varieties, crop) {
  const section = document.getElementById("variety-section");
  const container = document.getElementById("variety-cards");
  if (!varieties || varieties.length === 0) {
    section.classList.add("d-none");
    return;
  }
  section.classList.remove("d-none");
  container.innerHTML = varieties.map(v => buildVarietyCard(v, crop)).join("");
}

// ── Extend loadDashboard to also load variety guide ──────────────────────────
const _origLoad = loadDashboard;
// Override window.addEventListener load to also call loadVarietyGuide
window.addEventListener("load", () => {
  loadVarietyGuide();
});
