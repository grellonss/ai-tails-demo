/**
 * AI-Tails: Feline Grimace Scale (FGS) Clinical Analyzer
 * Client JavaScript Application
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("uploadDropzone");
  const fileInput = document.getElementById("fileInput");
  const canvas = document.getElementById("catCanvas");
  const ctx = canvas.getContext("2d");
  const emptyState = document.getElementById("canvasEmptyState");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");
  const catNameInput = document.getElementById("catNameInput");
  const toggleLandmarks = document.getElementById("toggleLandmarks");
  const toggleLines = document.getElementById("toggleLines");
  const profilePreset = document.getElementById("profilePreset");
  const apiUrlInput = document.getElementById("apiUrlInput");

  // Output Elements
  const verdictBanner = document.getElementById("verdictBanner");
  const verdictBadge = document.getElementById("verdictBadge");
  const verdictTitle = document.getElementById("verdictTitle");
  const verdictSubtitle = document.getElementById("verdictSubtitle");
  const verdictAdvice = document.getElementById("verdictAdvice");
  const gaugeScoreNum = document.getElementById("gaugeScoreNum");
  const gaugeProgress = document.getElementById("gaugeProgress");
  const scorePercentLabel = document.getElementById("scorePercentLabel");

  const earScoreBadge = document.getElementById("earScoreBadge");
  const earMetric = document.getElementById("earMetric");
  const earDesc = document.getElementById("earDesc");

  const eyeScoreBadge = document.getElementById("eyeScoreBadge");
  const eyeMetric = document.getElementById("eyeMetric");
  const eyeDesc = document.getElementById("eyeDesc");

  const muzzleScoreBadge = document.getElementById("muzzleScoreBadge");
  const muzzleMetric = document.getElementById("muzzleMetric");
  const muzzleDesc = document.getElementById("muzzleDesc");

  const swrlRuleCode = document.getElementById("swrlRuleCode");
  const reasonerTime = document.getElementById("reasonerTime");

  // App State
  let currentImage = null;
  let currentFile = null;
  let currentLandmarks = null;
  const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * 40; // r = 40 in SVG

  // =========================================================================
  // Drag and Drop & File Upload
  // =========================================================================

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  // Support paste from clipboard (Ctrl + V)
  document.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        handleSelectedFile(file);
        break;
      }
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Seleziona un file immagine valido (JPEG, PNG, WEBP).");
      return;
    }

    currentFile = file;
    currentLandmarks = null;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        currentImage = img;
        renderCanvas();
        emptyState.classList.add("hidden");
        analyzeBtn.disabled = false;
        
        // Auto-extract or preview
        runAnalysis();
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  // =========================================================================
  // Canvas Rendering (Image + FGS 48 Landmark Mesh)
  // =========================================================================

  function renderCanvas() {
    if (!currentImage) return;

    // Set canvas dimensions matching source image
    canvas.width = currentImage.width;
    canvas.height = currentImage.height;

    // Draw photo
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

    // If landmarks exist and toggle is active, draw mesh overlay
    if (currentLandmarks && currentLandmarks.length >= 48) {
      const showMesh = toggleLandmarks.checked;
      const showLines = toggleLines.checked;

      if (showLines) {
        drawGeometricLines(currentLandmarks);
      }

      if (showMesh) {
        drawLandmarkPoints(currentLandmarks);
      }
    }
  }

  function drawLandmarkPoints(lm) {
    // Specific FGS Action Unit indices
    const earPoints = [2, 40];
    const eyePoints = [8, 10, 12, 14];
    const muzzlePoints = [30, 32, 35, 37];
    const headPoints = [0, 25];

    lm.forEach((pt, idx) => {
      ctx.beginPath();
      let color = "rgba(255, 255, 255, 0.4)";
      let radius = Math.max(3, canvas.width * 0.007);

      if (earPoints.includes(idx)) {
        color = "#06b6d4"; // Cyan
        radius *= 1.4;
      } else if (eyePoints.includes(idx)) {
        color = "#10b981"; // Emerald
        radius *= 1.4;
      } else if (muzzlePoints.includes(idx)) {
        color = "#f59e0b"; // Amber
        radius *= 1.4;
      } else if (headPoints.includes(idx)) {
        color = "#8b5cf6"; // Violet
        radius *= 1.3;
      }

      ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // Outer ring for highlighted Action Units
      if (earPoints.includes(idx) || eyePoints.includes(idx) || muzzlePoints.includes(idx)) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }

  function drawGeometricLines(lm) {
    ctx.save();
    ctx.lineWidth = Math.max(2, canvas.width * 0.0035);

    // 1. Orecchie (Cyan line connecting ear tips)
    if (lm[2] && lm[40]) {
      ctx.strokeStyle = "rgba(6, 182, 212, 0.85)";
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(lm[2].x, lm[2].y);
      ctx.lineTo(lm[40].x, lm[40].y);
      ctx.stroke();
    }

    // 2. Ampiezza Testa (Head span)
    if (lm[0] && lm[25]) {
      ctx.strokeStyle = "rgba(139, 92, 246, 0.6)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(lm[0].x, lm[0].y);
      ctx.lineTo(lm[25].x, lm[25].y);
      ctx.stroke();
    }

    // 3. Occhi (Eye EAR box)
    if (lm[8] && lm[10] && lm[12] && lm[14]) {
      ctx.strokeStyle = "rgba(16, 185, 129, 0.9)";
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(lm[8].x, lm[8].y);
      ctx.lineTo(lm[10].x, lm[10].y);
      ctx.lineTo(lm[12].x, lm[12].y);
      ctx.lineTo(lm[14].x, lm[14].y);
      ctx.closePath();
      ctx.stroke();
    }

    // 4. Muso (Muzzle ellipse)
    if (lm[30] && lm[32] && lm[35] && lm[37]) {
      ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(lm[30].x, lm[30].y);
      ctx.lineTo(lm[37].x, lm[37].y);
      ctx.lineTo(lm[35].x, lm[35].y);
      ctx.lineTo(lm[32].x, lm[32].y);
      ctx.closePath();
      ctx.stroke();
    }

    ctx.restore();
  }

  // Toggles event listeners
  toggleLandmarks.addEventListener("change", renderCanvas);
  toggleLines.addEventListener("change", renderCanvas);

  // =========================================================================
  // API Call & Pipeline Execution
  // =========================================================================

  analyzeBtn.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    if (!currentImage) return;

    // Loading state
    analyzeBtn.disabled = true;
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Inferenza Ontologica in corso...";

    try {
      const catName = catNameInput.value.trim() || "Paziente_Felino";
      const preset = profilePreset.value;
      const customApiUrl = apiUrlInput.value.trim();

      const formData = new FormData();
      if (currentFile) {
        formData.append("image", currentFile);
      }
      formData.append("cat_name", catName);
      formData.append("profile_preset", preset);
      formData.append("image_width", currentImage.width);
      formData.append("image_height", currentImage.height);
      if (customApiUrl) {
        formData.append("api_url", customApiUrl);
      }

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Errore server: ${response.statusText}`);
      }

      const result = await response.json();
      displayResults(result);

    } catch (err) {
      console.error("Errore analisi:", err);
      alert(`Si è verificato un errore durante l'analisi: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      btnSpinner.classList.add("hidden");
      btnText.textContent = "Analizza Stato di Salute";
    }
  }

  // =========================================================================
  // Render Diagnosis & Results
  // =========================================================================

  function displayResults(data) {
    // 1. Update landmarks & re-render canvas
    if (data.landmarks) {
      currentLandmarks = data.landmarks;
      renderCanvas();
    }

    // 2. Verdict Banner (Answer to: È malato oppure no?)
    const status = data.health_status.toLowerCase(); // 'good', 'medium', 'poor'
    verdictBanner.className = `verdict-banner status-${status}`;

    if (status === "good") {
      verdictBadge.className = "verdict-badge badge-good";
      verdictBadge.textContent = "🟢 NON MALATO (SANO)";
      verdictTitle.textContent = "Gatto in Condizioni Ottimali";
      verdictSubtitle.textContent = "Nessun indicatore clinico o morfologico di dolore o patologia facciale rilevato.";
      verdictAdvice.textContent = "Raccomandazione: Il felino presenta tratti distesi e normali. Nessun intervento veterinario richiesto.";
      verdictAdvice.style.color = "var(--status-good)";
    } else if (status === "medium") {
      verdictBadge.className = "verdict-badge badge-medium";
      verdictBadge.textContent = "🟡 STATO DI ATTENZIONE";
      verdictTitle.textContent = "Disagio Lieve o Tensione";
      verdictSubtitle.textContent = "Rilevata alterazione moderata delle unità d'azione facciali. Possibile fastidio o dolore iniziale.";
      verdictAdvice.textContent = "Raccomandazione: Monitorare il comportamento alimentare e l'interazione nelle successive 2-4 ore.";
      verdictAdvice.style.color = "var(--status-medium)";
    } else {
      verdictBadge.className = "verdict-badge badge-poor";
      verdictBadge.textContent = "🔴 MALATO (DOLORE ACUTO)";
      verdictTitle.textContent = "Dolore Acuto Evidente";
      verdictSubtitle.textContent = "Segni marcati di sofferenza facciale FGS. Orecchie ruotate, fessura orbitale serrata o muso teso.";
      verdictAdvice.textContent = "Raccomandazione: Richiesta visita clinica veterinaria immediata e somministrazione di adeguata terapia analgesica.";
      verdictAdvice.style.color = "var(--status-poor)";
    }

    // 3. Gauge Progress (0 to 6)
    const totalScore = data.scores.total;
    gaugeScoreNum.textContent = totalScore;
    const percent = Math.round((totalScore / 6) * 100);
    scorePercentLabel.textContent = `Indice di sofferenza: ${percent}%`;

    const offset = CIRCLE_CIRCUMFERENCE - (totalScore / 6) * CIRCLE_CIRCUMFERENCE;
    gaugeProgress.style.strokeDashoffset = offset;
    gaugeProgress.style.stroke = status === "good" ? "var(--status-good)" : (status === "medium" ? "var(--status-medium)" : "var(--status-poor)");

    // 4. Action Units Cards
    updateAUCard(earScoreBadge, earMetric, earDesc, data.scores.ear, `Ratio: ${data.ratios.ear_ratio}`, data.descriptions.ear);
    updateAUCard(eyeScoreBadge, eyeMetric, eyeDesc, data.scores.eye, `EAR: ${data.ratios.eye_ratio}`, data.descriptions.eye);
    updateAUCard(muzzleScoreBadge, muzzleMetric, muzzleDesc, data.scores.muzzle, `Ratio: ${data.ratios.muzzle_ratio}`, data.descriptions.muzzle);

    // 5. Semantic Reasoner & SWRL Trace
    swrlRuleCode.textContent = data.swrl_rule;
    reasonerTime.textContent = `Timestamp: ${new Date().toLocaleTimeString()} • Reasoner: Pellet / OWL DL • Subject: ${data.cat_name}`;
  }

  function updateAUCard(badgeElem, metricElem, descElem, score, metricText, descText) {
    badgeElem.textContent = score;
    badgeElem.className = `au-badge-score score-${score}`;
    metricElem.textContent = metricText;
    descElem.textContent = descText;
  }
});
