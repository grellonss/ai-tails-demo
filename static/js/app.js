/**
 * AI-Tails: multimodal analysis client (RGB photo + thermal photo).
 */

document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("catCanvas"), ctx = canvas.getContext("2d");
  const tCanvas = $("thermalCanvas"), tctx = tCanvas.getContext("2d");
  const CIRCUMFERENCE = 2 * Math.PI * 40;
  const LEVEL_COLOR = { good: "var(--status-good)", medium: "var(--status-medium)", poor: "var(--status-poor)" };

  const state = { rgbFile: null, rgbImg: null, thermalFile: null, thermalImg: null, result: null, chat: [] };

  // ---------------------------------------------------------------------------
  // Component status (header)
  // ---------------------------------------------------------------------------
  fetch("/api/health").then((r) => r.json()).then((h) => {
    document.querySelectorAll("#statusPills .status-pill").forEach((pill) => {
      const ok = h[pill.dataset.key];
      pill.classList.toggle("off", !ok);
      pill.title = ok ? "configured" : "not configured (see .env)";
    });
  }).catch(() => {});

  // ---------------------------------------------------------------------------
  // File upload (click, drag & drop, paste)
  // ---------------------------------------------------------------------------
  function setupDropzone(zoneId, inputId, onFile) {
    const zone = $(zoneId), input = $(inputId);
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault(); zone.classList.remove("dragover");
      if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", (e) => { if (e.target.files.length) onFile(e.target.files[0]); });
  }

  function loadImage(file, done) {
    if (!file.type.startsWith("image/")) { showError("Select an image file (JPG, PNG, WEBP)."); return; }
    const reader = new FileReader();
    reader.onload = (ev) => { const img = new Image(); img.onload = () => done(img); img.src = ev.target.result; };
    reader.readAsDataURL(file);
  }

  const setRgb = (file) => loadImage(file, (img) => {
    Object.assign(state, { rgbFile: file, rgbImg: img, result: null });
    $("canvasEmptyState").classList.add("hidden");
    drawRgb(); updateButton();
  });
  const setThermal = (file) => loadImage(file, (img) => {
    Object.assign(state, { thermalFile: file, thermalImg: img, result: null });
    $("thermalEmptyState").classList.add("hidden");
    drawThermal(); updateButton();
  });
  setupDropzone("rgbDropzone", "rgbInput", setRgb);
  setupDropzone("thermalDropzone", "thermalInput", setThermal);
  document.addEventListener("paste", (e) => {   // paste: RGB photo first, then the thermal one
    const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === "file" && i.type.startsWith("image/"));
    if (item) (state.rgbFile ? setThermal : setRgb)(item.getAsFile());
  });

  function updateButton() {
    const ready = state.rgbFile && state.thermalFile;
    $("analyzeBtn").disabled = !ready;
    $("btnText").textContent = ready ? "Analyse health status" : "Upload both photos";
  }

  // ---------------------------------------------------------------------------
  // Drawing: RGB photo with face box and landmarks, thermal photo with hotspots
  // ---------------------------------------------------------------------------
  function drawRgb() {
    if (!state.rgbImg) return;
    canvas.width = state.rgbImg.width; canvas.height = state.rgbImg.height;
    ctx.drawImage(state.rgbImg, 0, 0);
    const r = state.result;
    if (!r) return;
    const s = Math.max(1, canvas.width / 600);
    const b = r.face.box;
    ctx.save();
    ctx.strokeStyle = "#4ade80"; ctx.lineWidth = 1.5 * s; ctx.setLineDash([6 * s, 4 * s]);
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.restore();
    if ($("toggleLines").checked) drawLines(r.landmarks, s);
    if ($("toggleLandmarks").checked) drawPoints(r.landmarks, s);
  }

  // indices used by the FGS computation (see DEFAULT_LANDMARK_MAP in vision/detector.py)
  const GROUPS = { ear: [2, 40], eye: [8, 10, 12, 14], muzzle: [30, 32, 35, 37], head: [0, 25] };
  const COLORS = { ear: "#06b6d4", eye: "#10b981", muzzle: "#f59e0b", head: "#8b5cf6" };

  function drawPoints(lm, s) {
    lm.forEach((p, i) => {
      const group = Object.keys(GROUPS).find((g) => GROUPS[g].includes(i));
      ctx.beginPath();
      ctx.arc(p.x, p.y, (group ? 5 : 3) * s, 0, 2 * Math.PI);
      ctx.fillStyle = group ? COLORS[group] : "rgba(255,255,255,0.5)";
      ctx.fill();
      if (group) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 * s; ctx.stroke(); }
    });
  }

  function drawLines(lm, s) {
    const line = (idx, color, dash, close = false) => {
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2 * s; ctx.setLineDash(dash.map((d) => d * s));
      idx.forEach((i, k) => (k ? ctx.lineTo(lm[i].x, lm[i].y) : ctx.moveTo(lm[i].x, lm[i].y)));
      if (close) ctx.closePath();
      ctx.stroke();
    };
    ctx.save();
    line([2, 40], COLORS.ear, [6, 4]);
    line([0, 25], COLORS.head, [4, 4]);
    line([8, 10, 12, 14], COLORS.eye, [], true);
    line([30, 37, 35, 32], COLORS.muzzle, [5, 3], true);
    ctx.restore();
  }

  function drawThermal() {
    if (!state.thermalImg) return;
    tCanvas.width = state.thermalImg.width; tCanvas.height = state.thermalImg.height;
    tctx.drawImage(state.thermalImg, 0, 0);
    const t = state.result?.thermal;
    if (!t) return;
    const r = Math.max(8, tCanvas.width / 40);
    tctx.strokeStyle = "#5ff2ff"; tctx.lineWidth = Math.max(2, tCanvas.width / 300);
    t.hotspots.forEach((p) => { tctx.beginPath(); tctx.arc(p.x, p.y, r, 0, 2 * Math.PI); tctx.stroke(); });
  }

  ["toggleLandmarks", "toggleLines"].forEach((id) => $(id).addEventListener("change", drawRgb));

  // ---------------------------------------------------------------------------
  // Analysis
  // ---------------------------------------------------------------------------
  function showError(msg) {
    $("errorBox").textContent = msg;
    $("errorBox").classList.toggle("hidden", !msg);
  }

  $("analyzeBtn").addEventListener("click", async () => {
    showError("");
    $("analyzeBtn").disabled = true; $("btnSpinner").classList.remove("hidden");
    $("btnText").textContent = "Analysing…";
    try {
      const form = new FormData();
      form.append("image", state.rgbFile);
      form.append("thermal_image", state.thermalFile);
      form.append("cat_name", $("catNameInput").value.trim() || "Patient");
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(`${data.error || res.statusText}${data.step ? ` (step: ${data.step})` : ""}`);
      state.result = data; state.chat = [];
      render(data);
    } catch (err) {
      showError(err.message);
    } finally {
      $("btnSpinner").classList.add("hidden");
      updateButton();
    }
  });

  function render(d) {
    drawRgb(); drawThermal();
    const lvl = d.verdict.level, color = LEVEL_COLOR[lvl];

    $("verdictBanner").className = `verdict-banner status-${lvl}`;
    $("verdictBadge").className = `verdict-badge badge-${lvl}`;
    $("verdictBadge").textContent = d.verdict.label;
    $("verdictTitle").textContent = d.ontology.primary_status;
    $("verdictSubtitle").textContent = d.ontology.explanation;
    const mock = (d.landmarks_source || "").startsWith("mock:");
    $("mockWarning").hidden = !mock;
    if (mock) $("mockWarning").textContent = `Warning: SYNTHETIC landmarks from the mock server (profile "${d.landmarks_source.slice(5)}"). ` +
      "The FGS score does not come from the photo: the real Tech4Animals API is needed.";

    const total = d.scores.total;
    $("gaugeScoreNum").textContent = total;
    $("gaugeProgress").style.strokeDashoffset = CIRCUMFERENCE - (total / 6) * CIRCUMFERENCE;
    $("gaugeProgress").style.stroke = color;
    $("scorePercentLabel").textContent = `Pain index: ${Math.round((total / 6) * 100)}%`;

    const t = d.thermal;
    $("tempValue").textContent = t.temperature.toFixed(1);
    $("thermalStatus").textContent = t.thermal_status;
    $("thermalDetail").textContent = `Surface ${t.surface_max_c} °C + offset ${t.body_offset_c} °C · scale ${t.scale.min}–${t.scale.max} °C`;
    const pct = Math.min(100, Math.max(0, ((t.temperature - 36) / 6) * 100));
    $("tempMarker").style.left = `${pct}%`; $("tempMarker").classList.remove("hidden");

    [["ear", "Ratio"], ["eye", "EAR"], ["muzzle", "Ratio"]].forEach(([k, label]) => {
      $(`${k}ScoreBadge`).textContent = d.scores[k];
      $(`${k}ScoreBadge`).className = `au-badge-score score-${d.scores[k]}`;
      $(`${k}Metric`).textContent = `${label}: ${d.ratios[`${k}_ratio`]}`;
      $(`${k}Desc`).textContent = d.descriptions[k];
    });

    $("swrlRuleCode").textContent = d.ontology.fired_rules.map((r) => r.swrl).join("\n\n") || "No rule fired";
    $("reasonerTime").textContent = `Reasoner: ${d.ontology.reasoner} · inferred statuses: ${d.ontology.inferred_status.join(", ")} · ${new Date().toLocaleTimeString()}`;

    $("reportText").innerHTML = markdown(d.report.text);
    $("reportEngine").textContent = d.report.engine === "summary"
      ? `Automatic summary${d.report.error ? ` — ${d.report.error}` : " (Azure OpenAI not configured)"}`
      : `Written by ${d.report.engine}`;

    $("chatLog").innerHTML = "";
    $("chatInput").disabled = $("chatBtn").disabled = false;
  }

  function escapeHtml(s) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function markdown(s) { return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>"); }

  // ---------------------------------------------------------------------------
  // Chat about the report
  // ---------------------------------------------------------------------------
  function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.innerHTML = markdown(text);
    $("chatLog").appendChild(div);
    $("chatLog").scrollTop = $("chatLog").scrollHeight;
  }

  $("chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = $("chatInput").value.trim();
    if (!question || !state.result) return;
    $("chatInput").value = "";
    addMessage("user", question);
    $("chatBtn").disabled = true;
    try {
      const res = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context: state.result, history: state.chat }),
      });
      const data = await res.json();
      const answer = res.ok ? data.answer : `Error: ${data.error}`;
      addMessage("assistant", answer);
      state.chat.push({ role: "user", content: question }, { role: "assistant", content: answer });
    } catch (err) {
      addMessage("assistant", `Error: ${err.message}`);
    } finally {
      $("chatBtn").disabled = false;
    }
  });
});
