"use strict";

const byId = (identifier) => document.getElementById(identifier);
const filters = {query: "search", verdict: "verdictFilter", risk: "riskFilter", refusal: "refusalFilter", review: "reviewFilter", split: "splitFilter"};
const state = {run: "", reviewer: "", offset: 0, limit: 30, rows: [], total: 0, selected: null, record: null, labels: {}, dirty: false, request: 0};
let searchTimer;

// Dataset text is always inserted as text, never interpreted as HTML.
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function notice(message, error = false) {
  const box = byId("notice");
  box.textContent = message;
  box.className = error ? "notice error" : "notice";
  box.hidden = !message;
}

async function api(path, parameters = {}, body = null) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(parameters)) url.searchParams.set(key, value);
  const options = body === null ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Review-App": "1"}, body: JSON.stringify(body),
  };
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function badge(label, human = false) {
  const known = ["Safe", "Unsafe", "Controversial", "Unparsed", "Excluded", "Pending", "Mismatched", "Error", ...Object.keys(state.labels)];
  const className = known.includes(label) ? label.toLowerCase() : "unparsed";
  return element("span", `badge ${className}`, human ? state.labels[label] || label : label);
}

function discardChanges() {
  return !state.dirty || window.confirm("Discard the unsaved review changes?");
}

function setOptions(select, entries, placeholder) {
  const previous = select.value;
  select.replaceChildren(new Option(placeholder, ""));
  for (const [value, label] of entries) select.add(new Option(label, value));
  select.value = Array.from(select.options).some((option) => option.value === previous) ? previous : "";
}

function showStats(stats) {
  const counts = stats.guard_counts;
  byId("totalCount").textContent = stats.total.toLocaleString();
  byId("unsafeCount").textContent = (counts.Unsafe || 0).toLocaleString();
  byId("safeCount").textContent = (counts.Safe || 0).toLocaleString();
  byId("excludedCount").textContent = (counts.Excluded || 0).toLocaleString();
  byId("unscoredCount").textContent = (stats.total - stats.scored - (counts.Excluded || 0)).toLocaleString();
  byId("overRefusalCount").textContent = stats.over_refusals.toLocaleString();
  byId("unsafeRate").textContent = stats.unsafe_rate === null ? "No scored responses" : `${(100 * stats.unsafe_rate).toFixed(2)}% of ${stats.scored.toLocaleString()} scored`;
  byId("reviewProgressText").textContent = `${stats.reviewed.toLocaleString()} / ${stats.total.toLocaleString()} human-reviewed${state.reviewer ? ` by ${state.reviewer}` : ""}`;
  byId("reviewProgress").max = stats.total || 1;
  byId("reviewProgress").value = stats.reviewed;
}

function renderRows() {
  const body = byId("resultsBody");
  body.replaceChildren();
  for (const record of state.rows) {
    const row = element("tr", record.id === state.selected ? "selected" : "");
    row.tabIndex = 0;
    row.setAttribute("aria-label", `Inspect ${record.id}`);
    const select = () => selectRecord(record.id).catch((error) => notice(error.message, true));
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
    });
    const content = element("td");
    const identifier = element("div", "row-id");
    const shortId = record.id.replace("answercarefully-instruction-", "").replace(/-002-002-/, " · ").replace(/-001:ja$/, "");
    identifier.append(element("span", "split-tag", record.split || "—"), element("span", "", shortId.replace(/^(dev|test) · /, "")));
    identifier.title = record.id;
    content.append(identifier, element("div", "prompt-preview", record.prompt), element("div", "answer-preview", record.answer), element("div", "risk-preview", record.risk));
    const guard = element("td");
    guard.append(badge(record.guard.label));
    guard.append(element("div", "refusal-note", record.guard.refusal === "Yes" ? "Refusal detected" : record.guard.refusal === "No" ? "No refusal detected" : "Refusal not assessed"));
    const human = element("td");
    if (record.human_labels.length) {
      for (const label of record.human_labels) human.append(badge(label, true));
    } else human.append(element("span", "subtle", "Unreviewed"));
    row.append(content, guard, human);
    body.append(row);
  }
  byId("emptyResults").hidden = state.rows.length !== 0;
  byId("matchCount").textContent = `${state.total.toLocaleString()} matching`;
  byId("pageInfo").textContent = state.total ? `${state.offset + 1}–${Math.min(state.offset + state.limit, state.total)} of ${state.total.toLocaleString()}` : "0 responses";
  byId("previousPage").disabled = state.offset === 0;
  byId("nextPage").disabled = state.offset + state.limit >= state.total;
}

async function loadResults(reset = false) {
  if (!state.run) return;
  if (reset) state.offset = 0;
  const requestNumber = ++state.request;
  const parameters = {run: state.run, reviewer: state.reviewer, offset: state.offset, limit: state.limit};
  for (const [key, identifier] of Object.entries(filters)) parameters[key] = byId(identifier).value;
  const result = await api("/api/results", parameters);
  if (requestNumber !== state.request) return;
  state.rows = result.rows;
  state.total = result.total;
  showStats(result.stats);
  setOptions(byId("riskFilter"), result.risk_areas.map((area) => [area, area]), "All risk areas");
  renderRows();
  if (!state.dirty && state.rows.length && !state.rows.some((row) => row.id === state.selected)) {
    await selectRecord(state.rows[0].id, true);
  }
}

async function loadRuns() {
  const result = await api("/api/runs");
  state.labels = result.review_labels;
  setOptions(byId("reviewFilter"), Object.entries(state.labels).map(([value, label]) => [value, value === "unreviewed" ? label : `Human: ${label}`]), "All human reviews");
  const select = byId("runSelect");
  select.replaceChildren();
  for (const run of result.runs) select.add(new Option(`${run.id}${run.complete ? "" : " · in progress"}`, run.id));
  if (!result.runs.length) {
    select.add(new Option("No saved runs found", ""));
    state.run = "";
    state.rows = [];
    state.total = 0;
    renderRows();
    notice("No response files found. Run bash run.sh first, then choose Reload files.");
    byId("exportButton").disabled = true;
    return;
  }
  byId("exportButton").disabled = false;
  if (!result.runs.some((run) => run.id === state.run)) state.run = result.runs[0].id;
  select.value = state.run;
  notice("");
  await loadResults();
}

function textSection(title, text, className = "") {
  const section = element("section", "detail-section");
  section.append(element("span", "detail-label", title), element("div", `text-content ${className}`, text || "Not provided"));
  return section;
}

function metadataView(record) {
  const section = element("section", "detail-section");
  const grid = element("dl", "metadata-grid");
  const categories = record.metadata.categories || {};
  const values = {
    "Risk area": categories["risk-area"], "Harm type": categories["harm-type"],
    "Specific harm": categories["specific-harm"], "Split": record.metadata.split,
    "Model": record.generation.model, "Temperature": record.generation.temperature,
    "Output limit": record.generation.max_tokens, "Output tokens": record.generation.usage?.completion_tokens,
    "Input tokens": record.generation.usage?.prompt_tokens, "Finish reason": record.generation.finish_reason,
    "Dataset revision": record.metadata.dataset_revision,
  };
  for (const [label, value] of Object.entries(values)) grid.append(element("dt", "", label), element("dd", "", value ?? "Not recorded"));
  section.append(grid);
  const details = element("details");
  details.append(element("summary", "", "All saved metadata"), element("pre", "raw-verdict", JSON.stringify({metadata: record.metadata, generation: record.generation}, null, 2)));
  section.append(details);
  return section;
}

function renderDetail(record) {
  state.dirty = false;
  const panel = byId("detailPanel");
  panel.replaceChildren();
  const top = element("div", "detail-top");
  const identity = element("div");
  identity.append(element("p", "eyebrow", "RESPONSE INSPECTOR"), element("div", "full-id", record.id));
  top.append(identity, badge(record.guard.label));
  const content = element("div", "detail-content");
  content.append(textSection("Original prompt", record.prompt, "prompt-box"));
  const translation = record.metadata.translation || {};
  if (translation["text-English"]) {
    const details = element("details");
    details.append(element("summary", "", "English translation & adaptation notes"), element("div", "translation", translation["text-English"]));
    if (translation["translation-notes"]?.trim()) details.append(element("div", "translation", translation["translation-notes"]));
    content.append(details);
  }
  const tabs = element("div", "detail-tabs");
  const tabContent = element("div");
  const views = {
    "Aya response": () => textSection("Generated answer", record.answer),
    "Reference": () => textSection("Dataset reference · not an exact-match target", record.reference),
    "Metadata": () => metadataView(record),
  };
  for (const [label, view] of Object.entries(views)) {
    const tab = element("button", "tab", label);
    tab.addEventListener("click", () => {
      for (const button of tabs.children) button.classList.remove("active");
      tab.classList.add("active");
      tabContent.replaceChildren(view());
    });
    tabs.append(tab);
  }
  content.append(tabs, tabContent);
  tabs.firstChild.click();
  const guard = element("section", "detail-section guard-box");
  guard.append(element("span", "detail-label", "Automated guard assessment"), badge(record.guard.label));
  guard.append(element("p", "guard-description", `Refusal: ${record.guard.refusal || "not assessed"} · Categories: ${record.guard.categories.join(", ") || "none reported"}`));
  let explanation = "A safe verdict does not establish that this answer is helpful or that a refusal is appropriate.";
  if (record.guard.label === "Unsafe") explanation = "The guard flagged this response. Inspect the answer and confirm or challenge the decision in your review.";
  if (record.guard.label === "Excluded") explanation = `Not judged because generation ended with ${record.generation.finish_reason || "an unknown finish reason"}. The partial answer is preserved above.`;
  if (["Controversial", "Unparsed", "Error", "Pending", "Mismatched"].includes(record.guard.label)) explanation = "No usable binary safety verdict is available. This response is not counted as safe.";
  guard.append(element("p", "guard-description", explanation));
  if (record.guard.raw) {
    const raw = element("details");
    raw.append(element("summary", "", "Raw guard reply"), element("pre", "raw-verdict", record.guard.raw));
    guard.append(raw);
  }
  content.append(guard);
  const review = element("section", "human-review");
  review.append(element("h3", "", "Your human review"), element("div", "review-caption", state.reviewer ? `Reviewing as ${state.reviewer}. Your judgment does not change the guard score.` : "Enter your reviewer name at the top to save your own assessment."));
  const label = element("select");
  label.id = "reviewLabel";
  label.setAttribute("aria-label", "Your human review label");
  for (const [value, title] of Object.entries(state.labels)) label.add(new Option(title, value));
  const notes = element("textarea");
  notes.id = "reviewNotes";
  notes.placeholder = "Why is this answer appropriate, unsafe, or over-refused?";
  notes.setAttribute("aria-label", "Review notes");
  notes.maxLength = 5000;
  const existing = record.reviews.find((entry) => entry.reviewer === state.reviewer);
  label.value = existing?.label || "unreviewed";
  notes.value = existing?.notes || "";
  label.addEventListener("change", () => { state.dirty = true; });
  notes.addEventListener("input", () => { state.dirty = true; });
  const saveRow = element("div", "save-row");
  const save = element("button", "button primary", "Save review");
  save.id = "saveReview";
  const status = element("span", "save-status");
  status.setAttribute("role", "status");
  save.addEventListener("click", async () => {
    if (!state.reviewer) { byId("reviewer").focus(); status.textContent = "Enter your reviewer name first."; return; }
    save.disabled = true;
    try {
      await api("/api/review", {}, {run: state.run, sample_id: record.id, content_hash: record.content_hash, reviewer: state.reviewer, label: label.value, notes: notes.value});
      state.dirty = false;
      await loadResults();
      await selectRecord(record.id, true);
      notice(`Review saved for ${state.reviewer}.`);
    } catch (error) { status.textContent = error.message; }
    finally { save.disabled = false; }
  });
  notes.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") save.click();
  });
  saveRow.append(save, status);
  review.append(label, notes, saveRow);
  if (record.reviews.length) {
    const all = element("details");
    all.append(element("summary", "", `Saved reviewer assessments (${record.reviews.length})`));
    for (const entry of record.reviews) {
      const item = element("div", "review-entry");
      item.append(element("div", "author", entry.reviewer), badge(entry.label, true), element("p", "", entry.notes), element("div", "subtle", new Date(entry.updated_at).toLocaleString()));
      all.append(item);
    }
    review.append(all);
  }
  panel.append(top, content, review);
}

async function selectRecord(identifier, force = false) {
  if (!force && !discardChanges()) return;
  const run = state.run;
  state.selected = identifier;
  const record = await api("/api/record", {run, id: identifier});
  if (run !== state.run || identifier !== state.selected) return;
  state.record = record;
  renderRows();
  renderDetail(record);
}

function resetFilters() {
  for (const identifier of Object.values(filters)) byId(identifier).value = "";
}

function safeAction(action) {
  return async () => { try { await action(); } catch (error) { notice(error.message, true); } };
}

byId("runSelect").addEventListener("change", safeAction(async () => {
  if (!discardChanges()) { byId("runSelect").value = state.run; return; }
  state.run = byId("runSelect").value;
  state.selected = null;
  state.dirty = false;
  resetFilters();
  await loadResults(true);
}));
byId("reviewer").addEventListener("change", safeAction(async () => {
  if (!discardChanges()) { byId("reviewer").value = state.reviewer; return; }
  state.reviewer = byId("reviewer").value.trim();
  try { localStorage.setItem("ez-penta-reviewer", state.reviewer); } catch (storageError) { /* Storage may be disabled. */ }
  state.dirty = false;
  await loadResults();
  if (state.selected) await selectRecord(state.selected, true);
}));
for (const [key, identifier] of Object.entries(filters)) {
  if (key === "query") continue;
  byId(identifier).addEventListener("change", safeAction(() => loadResults(true)));
}
byId("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(safeAction(() => loadResults(true)), 220);
});
byId("clearFilters").addEventListener("click", safeAction(async () => { resetFilters(); await loadResults(true); }));
byId("reloadButton").addEventListener("click", safeAction(async () => {
  if (!discardChanges()) return;
  state.dirty = false;
  await loadRuns();
  if (state.selected) await selectRecord(state.selected, true);
}));
byId("previousPage").addEventListener("click", safeAction(async () => { state.offset = Math.max(0, state.offset - state.limit); await loadResults(); }));
byId("nextPage").addEventListener("click", safeAction(async () => { state.offset += state.limit; await loadResults(); }));
for (const card of document.querySelectorAll("[data-verdict]")) {
  card.addEventListener("click", safeAction(async () => { resetFilters(); byId("verdictFilter").value = card.dataset.verdict; await loadResults(true); }));
}
byId("overRefusalCard").addEventListener("click", safeAction(async () => { resetFilters(); byId("reviewFilter").value = "over_refusal"; await loadResults(true); }));
byId("exportButton").addEventListener("click", safeAction(async () => {
  const data = await api("/api/export", {run: state.run});
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
  const link = element("a");
  link.href = url;
  link.download = `${state.run}-human-reviews.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}));
byId("importFile").addEventListener("change", safeAction(async () => {
  const file = byId("importFile").files[0];
  if (!file) return;
  try {
    if (!state.run) throw new Error("Choose a run before importing reviews.");
    if (file.size > 10 * 1024 * 1024) throw new Error("Review exports must be under 10 MB.");
    const result = await api("/api/import", {}, {run: state.run, export: JSON.parse(await file.text())});
    notice(`Imported ${result.imported} reviews · ${result.unchanged} unchanged · ${result.conflicts} conflicts kept as-is.`);
    await loadResults();
    if (state.selected && !state.dirty) await selectRecord(state.selected, true);
  } finally { byId("importFile").value = ""; }
}));

try { state.reviewer = localStorage.getItem("ez-penta-reviewer") || ""; } catch (storageError) { /* Storage may be disabled. */ }
byId("reviewer").value = state.reviewer;
loadRuns().catch((error) => notice(error.message, true));
