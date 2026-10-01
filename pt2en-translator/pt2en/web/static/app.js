// PT→EN Course PDF Translator — front-end (no build step required).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const STAGES = [
  ["uploading", "Uploading"],
  ["analyzing", "Analyzing"],
  ["extracting", "Extracting"],
  ["ocr", "OCR"],
  ["translating", "Translating"],
  ["rebuilding", "Rebuilding layout"],
  ["validating", "Validating"],
  ["finalizing", "Finalizing"],
];
const STYLE_HINTS = {
  academic: "Natural, formal English for university course material.",
  technical: "Precise, concise English with standard field terminology.",
  literal: "Faithful to the source wording and structure; minimal rephrasing.",
};

const state = {
  config: null,
  file: null,
  jobId: null,
  job: null,
  report: null,
  pages: null,
  token: localStorage.getItem("pt2en-token") || "",
  zoom: 1,
  mode: "split",
  severity: "all",
  events: null,
  poll: null,
};

// ------------------------------------------------------------------ utils
function withToken(url) {
  if (!state.token) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(state.token);
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (state.token) headers["Authorization"] = "Bearer " + state.token;
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    const token = prompt("This server requires an access token:");
    if (token) {
      state.token = token.trim();
      localStorage.setItem("pt2en-token", state.token);
      return api(path, opts);
    }
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res;
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(0) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}

function fmtDuration(s) {
  if (s == null || !isFinite(s)) return "";
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${s % 60 ? (s % 60) + "s" : ""}`.trim();
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function show(view) {
  for (const v of ["upload", "progress", "result", "error"]) $(`#view-${v}`).hidden = v !== view;
  window.scrollTo({ top: 0 });
}

function setHash(jobId) {
  history.replaceState(null, "", jobId ? `#job=${jobId}` : location.pathname);
}

// ----------------------------------------------------------------- config
async function loadConfig() {
  try {
    const cfg = await api("/api/config");
    state.config = cfg;
    $("#limitMb").textContent = cfg.limits.max_upload_mb;
    $("#limitPages").textContent = cfg.limits.max_pages;
    $("#versionText").textContent = `pt2en-translator ${cfg.version}`;
    const sel = $("#provider");
    sel.innerHTML = "";
    for (const p of cfg.providers) {
      const opt = new Option(p.label + (p.available ? "" : " — not configured"), p.name);
      opt.disabled = !p.available;
      sel.add(opt);
    }
    sel.value = cfg.default_provider;
    const d = cfg.defaults;
    $("#variant").value = d.english_variant;
    const styleInput = $(`input[name=style][value=${d.style}]`);
    if (styleInput) styleInput.checked = true;
    $("#preserveTerms").checked = d.preserve_terminology;
    $("#translateImages").checked = d.translate_images;
    $("#ocrEnabled").checked = d.ocr_enabled && cfg.ocr.available;
    $("#ocrEnabled").disabled = !cfg.ocr.available;
    if (cfg.ocr.reason) {
      const hint = $("#ocrHint");
      hint.textContent = cfg.ocr.reason;
      hint.classList.toggle("warn", !cfg.ocr.available || cfg.ocr.reason.includes("missing"));
    }
    $("#localizeNumbers").checked = d.localize_numbers;
    $("#qaMode").value = d.qa_review;
    updateEngine();
    updateStyleHint();
  } catch (err) {
    const el = $("#engine");
    el.className = "engine bad";
    $("#engineText").textContent = "Server unavailable";
    console.error(err);
  }
}

function updateEngine() {
  const cfg = state.config;
  if (!cfg) return;
  const name = $("#provider").value;
  const p = cfg.providers.find((x) => x.name === name);
  const el = $("#engine");
  let label = p ? p.label.split(" — ")[0] : name;
  if (name === "anthropic") label = `Claude · ${cfg.anthropic_model}`;
  el.className = "engine " + (name === "demo" ? "demo" : p && p.available ? "ok" : "bad");
  $("#engineText").textContent = label + (cfg.ocr.available ? " · OCR ready" : " · OCR unavailable");
  el.title = cfg.ocr.available
    ? `OCR: Tesseract (${cfg.ocr.languages})${cfg.ocr.path ? " at " + cfg.ocr.path : ""}`
    : `OCR unavailable: ${cfg.ocr.reason}`;
  $("#providerHint").textContent =
    name === "demo"
      ? "Demo mode uses a small offline dictionary to exercise the pipeline. Configure ANTHROPIC_API_KEY for real translations."
      : "Translations are produced by " + label + ".";
}

function updateStyleHint() {
  const v = $("input[name=style]:checked").value;
  $("#styleHint").textContent = STYLE_HINTS[v] || "";
}

// ------------------------------------------------------------- file input
function setFile(file) {
  $("#formError").hidden = true;
  if (!file) {
    state.file = null;
    $("#fileInfo").hidden = true;
    $("#startBtn").disabled = true;
    return;
  }
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  if (!isPdf) return formError("Please choose a PDF file.");
  const max = (state.config?.limits.max_upload_mb || 100) * 1048576;
  if (file.size > max) return formError(`The file is larger than ${fmtBytes(max)}.`);
  state.file = file;
  $("#fileName").textContent = file.name;
  $("#fileSize").textContent = fmtBytes(file.size);
  $("#fileInfo").hidden = false;
  $("#startBtn").disabled = false;
}

function formError(msg) {
  const el = $("#formError");
  el.textContent = msg;
  el.hidden = false;
}

function bindUpload() {
  const dz = $("#dropzone");
  const input = $("#fileInput");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => setFile(input.files[0]));
  for (const ev of ["dragenter", "dragover"]) {
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); });
  }
  dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  // Allow dropping anywhere on the upload view.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!$("#view-upload").hidden && e.dataTransfer?.files?.length) setFile(e.dataTransfer.files[0]);
  });
  $("#clearFile").addEventListener("click", () => { input.value = ""; setFile(null); });

  $("#sampleBtn").addEventListener("click", async () => {
    try {
      const res = await api("/api/sample");
      const blob = await res.blob();
      setFile(new File([blob], "curso_exemplo_pt.pdf", { type: "application/pdf" }));
    } catch (err) {
      formError("Could not load the sample document: " + err.message);
    }
  });

  $$("input[name=style]").forEach((r) => r.addEventListener("change", updateStyleHint));
  $("#provider").addEventListener("change", updateEngine);

  const gl = $("#glossary");
  const updateCount = () => {
    const n = gl.value.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).length;
    const badge = $("#glossaryCount");
    badge.textContent = n;
    badge.hidden = n === 0;
  };
  gl.addEventListener("input", updateCount);
  $("#importGlossary").addEventListener("click", () => $("#glossaryFile").click());
  $("#glossaryFile").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const text = await f.text();
    gl.value = (gl.value.trim() ? gl.value.trim() + "\n" : "") + text.trim();
    updateCount();
    $("#glossaryBox").open = true;
  });

  $("#settingsForm").addEventListener("submit", (e) => { e.preventDefault(); startJob(); });
}

function collectOptions() {
  return {
    provider: $("#provider").value,
    style: $("input[name=style]:checked").value,
    english_variant: $("#variant").value,
    preserve_terminology: $("#preserveTerms").checked,
    translate_images: $("#translateImages").checked,
    ocr_enabled: $("#ocrEnabled").checked,
    localize_numbers: $("#localizeNumbers").checked,
    qa_review: $("#qaMode").value,
    glossary_text: $("#glossary").value,
  };
}

// -------------------------------------------------------------- start job
function startJob() {
  if (!state.file) return;
  const fd = new FormData();
  fd.append("file", state.file, state.file.name);
  fd.append("options", JSON.stringify(collectOptions()));
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  if (state.token) xhr.setRequestHeader("Authorization", "Bearer " + state.token);
  $("#uploadBox").hidden = false;
  $("#startBtn").disabled = true;
  renderStepper("uploading");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $("#uploadBar").style.width = pct + "%";
    $("#uploadPct").textContent = pct + "%";
    $("#uploadText").textContent = `Uploading ${fmtBytes(e.loaded)} of ${fmtBytes(e.total)}`;
  };
  xhr.onload = () => {
    $("#uploadBox").hidden = true;
    $("#startBtn").disabled = false;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
    if (xhr.status === 201) {
      followJob(body);
    } else {
      formError(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `Upload failed (${xhr.status}).`);
    }
  };
  xhr.onerror = () => {
    $("#uploadBox").hidden = true;
    $("#startBtn").disabled = false;
    formError("Network error while uploading the file.");
  };
  xhr.send(fd);
}

// --------------------------------------------------------------- progress
function renderStepper(active) {
  const ol = $("#stepper");
  const idx = STAGES.findIndex(([k]) => k === active);
  ol.innerHTML = STAGES.map(([key, label], i) => {
    const cls = active === "done" || i < idx ? "done" : i === idx ? "active" : "";
    const mark = cls === "done" ? "✓" : i + 1;
    return `<li class="${cls}" data-stage="${key}"><span class="node">${mark}</span>${label}</li>`;
  }).join("");
}

function followJob(job) {
  stopFollowing();
  state.jobId = job.id;
  state.job = job;
  setHash(job.id);
  $("#prog-title").textContent = job.filename;
  $("#progMeta").textContent = `${job.page_count} page${job.page_count === 1 ? "" : "s"}`;
  $("#activity").innerHTML = "";
  show("progress");
  updateProgress(job);
  if (["completed", "failed", "cancelled"].includes(job.status)) return onTerminal(job);

  if ("EventSource" in window) {
    const es = new EventSource(withToken(`/api/jobs/${job.id}/events`));
    state.events = es;
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (!msg.job) return;
      state.job = msg.job;
      updateProgress(msg.job);
      if (["completed", "failed", "cancelled"].includes(msg.job.status)) {
        stopFollowing();
        onTerminal(msg.job);
      }
    };
    es.onerror = () => { es.close(); state.events = null; startPolling(); };
  } else {
    startPolling();
  }
}

function startPolling() {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${state.jobId}`);
      state.job = job;
      updateProgress(job);
      if (["completed", "failed", "cancelled"].includes(job.status)) { stopFollowing(); onTerminal(job); }
    } catch (err) { console.warn(err); }
  }, 1500);
}

function stopFollowing() {
  if (state.events) { state.events.close(); state.events = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
}

function updateProgress(job) {
  const p = job.progress || {};
  const stage = job.status === "queued" ? "analyzing" : p.stage || "analyzing";
  renderStepper(stage === "done" ? "done" : stage);
  const pct = Math.round((p.overall || 0) * 100);
  $("#overallBar").style.width = pct + "%";
  $("#overallPct").textContent = pct + "%";
  if (job.status === "queued") {
    $("#stageMsg").textContent = "Waiting for a free worker…";
    $("#etaText").textContent = "Queued";
  } else {
    $("#stageMsg").textContent = p.message || p.stage_label || "Working…";
    const elapsed = p.elapsed ? `Elapsed ${fmtDuration(p.elapsed)}` : "";
    const eta = p.eta_seconds ? ` · about ${fmtDuration(p.eta_seconds)} remaining` : p.overall > 0.03 ? "" : " · estimating…";
    $("#etaText").textContent = elapsed + eta;
  }
  const log = job.log || [];
  $("#activity").innerHTML = log.slice(-8).reverse().map((m) => `<li>${escapeHtml(m)}</li>`).join("");
}

async function onTerminal(job) {
  if (job.status === "completed") return loadResult(job);
  if (job.status === "cancelled") {
    show("upload");
    setHash(null);
    formError("Translation cancelled.");
    return;
  }
  $("#errorMessage").textContent = job.error || "Unknown error.";
  $("#errorDetail").textContent = job.error_detail || "";
  show("error");
}

// ----------------------------------------------------------------- result
async function loadResult(job) {
  state.jobId = job.id;
  try {
    const [report, pages] = await Promise.all([
      api(`/api/jobs/${job.id}/report`),
      api(`/api/jobs/${job.id}/pages`),
    ]);
    state.report = report;
    state.pages = pages;
  } catch (err) {
    $("#errorMessage").textContent = "The result could not be loaded: " + err.message;
    show("error");
    return;
  }
  renderSummary(job, state.report);
  renderChecks(state.report.checks);
  renderIssues();
  renderGlossary(state.report.glossary);
  show("result");
  buildViewer();
}

function renderSummary(job, report) {
  const s = report.summary;
  $("#result-title").textContent = job.filename.replace(/\.pdf$/i, "") + " — English";
  const statusText = { passed: "Passed all checks", passed_with_warnings: "Completed with warnings", needs_review: "Needs review" }[s.status] || s.status;
  $("#statusBadge").textContent = statusText;
  $("#resultMeta").textContent = `${report.document.pages} pages · ${report.translation.provider} · ${report.translation.style} style · ${report.translation.english_variant}`;
  const ring = $("#scoreRing");
  ring.className = "score " + (s.score >= 85 ? "" : s.score >= 60 ? "warn" : "bad");
  $("#scoreValue").textContent = s.score;
  requestAnimationFrame(() => { $("#scoreArc").style.strokeDashoffset = String(106.8 * (1 - s.score / 100)); });
  const counts = [];
  if (s.errors) counts.push(`<span class="pill error">${s.errors} error${s.errors > 1 ? "s" : ""}</span>`);
  if (s.warnings) counts.push(`<span class="pill warning">${s.warnings} warning${s.warnings > 1 ? "s" : ""}</span>`);
  if (s.info) counts.push(`<span class="pill info">${s.info} note${s.info > 1 ? "s" : ""}</span>`);
  if (!counts.length) counts.push('<span class="pill ok">No issues found</span>');
  $("#counts").innerHTML = counts.join("");
  $("#dlTranslated").href = withToken(`/api/jobs/${job.id}/download/translated`);
  $("#dlReview").href = withToken(`/api/jobs/${job.id}/download/review`);
  $("#dlReport").href = withToken(`/api/jobs/${job.id}/download/report`);

  const st = report.stats;
  const items = [
    [st.translated_blocks, "text blocks translated"],
    [st.headings, "headings"],
    [st.table_cells, "table cells"],
    [st.display_formulas + st.inline_formulas, "formulas preserved"],
    [st.images_translated + "/" + st.images, "images with translated text"],
    [st.image_labels_translated, "labels translated in images"],
    [report.document.scanned_pages, "scanned pages (OCR)"],
    [report.glossary.length, "glossary terms enforced"],
  ];
  $("#stats").innerHTML = items.map(([v, l]) => `<div class="stat"><b>${escapeHtml(v)}</b><span>${escapeHtml(l)}</span></div>`).join("");
}

function renderChecks(checks) {
  const icon = { pass: "✓", warn: "!", fail: "✕", skipped: "–" };
  $("#checks").innerHTML = checks.map((c) => `
    <li class="${c.status}"><span class="ico">${icon[c.status] || "?"}</span>
      <span>${escapeHtml(c.label)}<small>${escapeHtml(c.details)}</small></span></li>`).join("");
}

function renderIssues() {
  const issues = state.report.issues.filter((i) => state.severity === "all" || i.severity === state.severity);
  $("#issueCount").textContent = state.report.issues.length;
  const ul = $("#issues");
  if (!issues.length) {
    ul.innerHTML = `<li class="empty">${state.report.issues.length ? "No items with this severity." : "Nothing to review — all checks passed."}</li>`;
    return;
  }
  ul.innerHTML = issues.map((i) => {
    const idx = state.report.issues.indexOf(i);
    const page = i.page_number ? `Page ${i.page_number}` : "Document";
    const src = i.source_text ? `<q>PT: ${escapeHtml(i.source_text.slice(0, 160))}</q>` : "";
    const tgt = i.translated_text ? `<q>EN: ${escapeHtml(i.translated_text.slice(0, 160))}</q>` : "";
    const sug = i.suggestion ? `<q>→ ${escapeHtml(i.suggestion.slice(0, 200))}</q>` : "";
    const fixed = i.auto_fixed ? ' · <span class="fixed">auto-corrected</span>' : "";
    return `<li class="issue ${i.severity}" data-idx="${idx}" tabindex="0">
      <div class="meta"><span>${escapeHtml(i.category.replace(/_/g, " "))}${fixed}</span><span>${page}</span></div>
      ${escapeHtml(i.message)}${src}${tgt}${sug}</li>`;
  }).join("");
  $$(".issue", ul).forEach((li) => {
    const go = () => focusIssue(Number(li.dataset.idx));
    li.addEventListener("click", go);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
}

function renderGlossary(entries) {
  $("#glossUsed").textContent = entries.length;
  const tbody = $("#glossTable tbody");
  tbody.innerHTML = entries.length
    ? entries.map((e) => `<tr><td>${escapeHtml(e.pt)}</td><td>${escapeHtml(e.keep ? e.pt + " (kept)" : e.en)}</td><td>${escapeHtml(e.source)}</td></tr>`).join("")
    : '<tr><td colspan="3" class="empty">No document-specific terms.</td></tr>';
}

// ----------------------------------------------------------------- viewer
let observer = null;
let syncing = false;

function buildViewer() {
  const total = (state.pages.translated || state.pages.original || []).length;
  $("#pageTotal").textContent = total;
  $("#pageInput").max = total;
  if (observer) observer.disconnect();
  observer = new IntersectionObserver(onVisible, { root: null, rootMargin: "600px 0px" });
  for (const [doc, paneSel] of [["original", "#paneOriginal"], ["translated", "#paneTranslated"]]) {
    const container = $(`${paneSel} .pages`);
    container.innerHTML = "";
    (state.pages[doc] || []).forEach((size, i) => {
      const div = document.createElement("div");
      div.className = "page loading";
      div.dataset.doc = doc;
      div.dataset.page = String(i + 1);
      div.style.aspectRatio = `${size.width} / ${size.height}`;
      div.innerHTML = `<span class="num">${i + 1}</span>`;
      if (doc === "translated") addHighlights(div, i, size);
      container.appendChild(div);
      observer.observe(div);
    });
  }
  applyZoom();
  setMode(state.mode);
}

function addHighlights(div, pageIndex, size) {
  state.report.issues.forEach((issue, idx) => {
    if (issue.page !== pageIndex || !issue.bbox) return;
    const [x0, y0, x1, y1] = issue.bbox;
    const hl = document.createElement("div");
    hl.className = `hl ${issue.severity}`;
    hl.dataset.idx = String(idx);
    hl.style.left = (100 * x0 / size.width) + "%";
    hl.style.top = (100 * y0 / size.height) + "%";
    hl.style.width = Math.max(0.6, 100 * (x1 - x0) / size.width) + "%";
    hl.style.height = Math.max(0.6, 100 * (y1 - y0) / size.height) + "%";
    hl.title = `${issue.severity.toUpperCase()}: ${issue.message}`;
    hl.addEventListener("click", () => {
      state.severity = "all";
      $$("#issueFilters .chip").forEach((c) => c.classList.toggle("active", c.dataset.sev === "all"));
      renderIssues();
      const li = $(`.issue[data-idx="${idx}"]`);
      if (li) { li.scrollIntoView({ block: "center", behavior: "smooth" }); li.focus(); }
    });
    div.appendChild(hl);
  });
}

function renderScale(div) {
  const width = div.clientWidth || 600;
  const size = state.pages[div.dataset.doc][Number(div.dataset.page) - 1];
  const scale = Math.min(3, Math.max(1.25, (width / size.width) * (window.devicePixelRatio || 1) * 1.15));
  return Math.round(scale * 4) / 4;
}

function onVisible(entries) {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const div = entry.target;
    const scale = renderScale(div);
    if (div.dataset.scale === String(scale)) continue;
    div.dataset.scale = String(scale);
    let img = div.querySelector("img");
    if (!img) {
      img = document.createElement("img");
      img.alt = `${div.dataset.doc === "original" ? "Original" : "Translated"} page ${div.dataset.page}`;
      img.decoding = "async";
      img.onload = () => div.classList.remove("loading");
      div.prepend(img);
    }
    img.src = withToken(`/api/jobs/${state.jobId}/pages/${div.dataset.page}.png?doc=${div.dataset.doc}&scale=${scale}`);
  }
}

function setMode(mode) {
  state.mode = mode;
  const panes = $("#panes");
  panes.classList.remove("split", "original", "translated");
  panes.classList.add(mode);
  $$(".tab").forEach((t) => {
    const on = t.dataset.mode === mode;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
  });
}

function applyZoom() {
  $("#zoomText").textContent = Math.round(state.zoom * 100) + "%";
  $$(".pages").forEach((p) => {
    p.style.width = (state.zoom * 100) + "%";
    p.style.minWidth = state.zoom > 1 ? (state.zoom * 100) + "%" : "";
  });
  // Re-request sharper renderings for visible pages.
  if (observer) $$(".page").forEach((d) => { observer.unobserve(d); observer.observe(d); });
}

function currentPage(pane) {
  const pages = $$(".page", pane);
  const top = pane.scrollTop + 60;
  let current = 1;
  for (const p of pages) {
    if (p.offsetTop <= top) current = Number(p.dataset.page);
    else break;
  }
  return current;
}

function scrollToPage(n) {
  const total = Number($("#pageTotal").textContent) || 1;
  n = Math.min(Math.max(1, n), total);
  for (const pane of $$(".pane")) {
    const page = $(`.page[data-page="${n}"]`, pane);
    if (page) pane.scrollTo({ top: page.offsetTop - 44, behavior: "smooth" });
  }
  $("#pageInput").value = n;
}

function focusIssue(idx) {
  const issue = state.report.issues[idx];
  if (issue.page == null) return;
  if (state.mode === "original") setMode("split");
  scrollToPage(issue.page + 1);
  $$(".hl.flash").forEach((h) => h.classList.remove("flash"));
  const hl = $(`.hl[data-idx="${idx}"]`);
  if (hl) {
    setTimeout(() => {
      const pane = $("#paneTranslated");
      pane.scrollTo({ top: hl.parentElement.offsetTop + hl.offsetTop - pane.clientHeight / 3, behavior: "smooth" });
      void hl.offsetWidth;
      hl.classList.add("flash");
    }, 350);
  }
}

function bindViewer() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
  $("#prevPage").addEventListener("click", () => scrollToPage(Number($("#pageInput").value) - 1));
  $("#nextPage").addEventListener("click", () => scrollToPage(Number($("#pageInput").value) + 1));
  $("#pageInput").addEventListener("change", (e) => scrollToPage(Number(e.target.value)));
  $("#zoomIn").addEventListener("click", () => { state.zoom = Math.min(2.5, state.zoom + 0.25); applyZoom(); });
  $("#zoomOut").addEventListener("click", () => { state.zoom = Math.max(0.5, state.zoom - 0.25); applyZoom(); });
  $("#showIssues").addEventListener("change", (e) => $("#panes").classList.toggle("no-hl", !e.target.checked));
  $$("#issueFilters .chip").forEach((chip) => chip.addEventListener("click", () => {
    state.severity = chip.dataset.sev;
    $$("#issueFilters .chip").forEach((c) => c.classList.toggle("active", c === chip));
    renderIssues();
  }));
  // Synchronised scrolling between the original and the translation.
  const [a, b] = [$("#paneOriginal"), $("#paneTranslated")];
  const sync = (src, dst) => () => {
    if (syncing) return;
    syncing = true;
    const ratio = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
    dst.scrollTop = ratio * (dst.scrollHeight - dst.clientHeight);
    dst.scrollLeft = src.scrollLeft;
    $("#pageInput").value = currentPage(src);
    requestAnimationFrame(() => { syncing = false; });
  };
  a.addEventListener("scroll", sync(a, b), { passive: true });
  b.addEventListener("scroll", sync(b, a), { passive: true });
}

// ---------------------------------------------------------------- actions
function resetToUpload() {
  stopFollowing();
  setHash(null);
  state.jobId = null;
  state.report = null;
  $("#fileInput").value = "";
  setFile(null);
  show("upload");
}

function bindActions() {
  $("#cancelBtn").addEventListener("click", async () => {
    if (!state.jobId) return;
    if (!confirm("Cancel this translation?")) return;
    try { await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }); } catch (err) { console.warn(err); }
  });
  $("#newDocBtn").addEventListener("click", resetToUpload);
  $("#retryBtn").addEventListener("click", resetToUpload);
}

async function resumeFromHash() {
  const m = location.hash.match(/job=([a-f0-9]{32})/);
  if (!m) return;
  try {
    const job = await api(`/api/jobs/${m[1]}`);
    followJob(job);
  } catch (_) {
    setHash(null);
  }
}

// ------------------------------------------------------------------- boot
bindUpload();
bindViewer();
bindActions();
renderStepper("uploading");
loadConfig().then(resumeFromHash);
window.addEventListener("resize", () => { if (!$("#view-result").hidden) applyZoom(); });
