import {
  EditorState, EditorSelection, StateField, StateEffect,
  EditorView, Decoration, keymap, placeholder, drawSelection, highlightActiveLine, lineNumbers,
  history, defaultKeymap, historyKeymap, indentWithTab,
  syntaxHighlighting, HighlightStyle, markdown, tags,
} from "./vendor/codemirror.js";
import { planInsertion } from "./insert.js";

// ---------- helpers ----------

const $ = (sel) => document.querySelector(sel);
const enc = encodeURIComponent;

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  return el;
}

function debounce(fn, ms) {
  let timer;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(timer);
  return wrapped;
}

async function api(method, url, body, options = {}) {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const error = new Error(data.detail || res.statusText);
    error.status = res.status;
    error.data = data;
    throw error;
  }
  return data;
}

const store = {
  get(key) { try { return localStorage.getItem(`isis.${key}`); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(`isis.${key}`, value); } catch {} },
};

let flashTimer;
function flash(message) {
  $("#flash").textContent = message;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => ($("#flash").textContent = ""), 2500);
}

function fail(error) {
  console.error(error);
  flash(`Error: ${error.message}`);
}

// ---------- editors ----------

const markdownStyle = HighlightStyle.define([
  { tag: tags.heading1, fontSize: "1.3em", fontWeight: "700" },
  { tag: tags.heading2, fontSize: "1.15em", fontWeight: "700" },
  { tag: [tags.heading3, tags.heading4, tags.heading5, tags.heading6], fontWeight: "700" },
  { tag: tags.emphasis, fontStyle: "italic" },
  { tag: tags.strong, fontWeight: "700" },
  { tag: tags.link, color: "var(--accent)" },
  { tag: tags.url, color: "var(--muted)" },
  { tag: tags.monospace, fontFamily: "var(--mono)", fontSize: "0.9em" },
  { tag: tags.quote, color: "var(--muted)" },
  { tag: [tags.processingInstruction, tags.meta, tags.contentSeparator], color: "var(--faint)" },
]);

const theme = EditorView.theme({
  "&": { height: "100%", fontSize: "15.5px", backgroundColor: "var(--bg)", color: "var(--fg)" },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { fontFamily: "var(--serif)", lineHeight: "1.62" },
  ".cm-content": { padding: "18px 0 40vh", maxWidth: "44em", caretColor: "var(--fg)" },
  ".cm-line": { padding: "0 22px" },
  ".cm-gutters": { backgroundColor: "var(--bg)", color: "var(--faint)", border: "none", fontFamily: "var(--sans)", fontSize: "11px" },
  ".cm-cursor": { borderLeftColor: "var(--fg)", borderLeftWidth: "2px" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "var(--selection) !important" },
  ".cm-activeLine": { backgroundColor: "var(--active-line)" },
  ".cm-placeholder": { color: "var(--faint)", fontStyle: "italic" },
  ".cm-hit-line": { backgroundColor: "var(--hit-line)" },
  ".cm-hit-match": { backgroundColor: "var(--mark)", borderRadius: "2px" },
  ".cm-just-added": { backgroundColor: "var(--added)", transition: "background-color 1s" },
});

// A replaceable set of decorations, used for the search hit in the reader and for
// briefly marking what was just added to the task.
function decorationSlot() {
  const set = StateEffect.define();
  const field = StateField.define({
    create: () => Decoration.none,
    update(decos, tr) {
      decos = decos.map(tr.changes);
      for (const e of tr.effects) if (e.is(set)) decos = e.value;
      return decos;
    },
    provide: (f) => EditorView.decorations.from(f),
  });
  return { set, field };
}

const hitSlot = decorationSlot();
const addedSlot = decorationSlot();

function readerState(doc) {
  return EditorState.create({
    doc,
    extensions: [
      EditorState.readOnly.of(true),
      lineNumbers(),
      drawSelection(),
      EditorView.lineWrapping,
      markdown(),
      syntaxHighlighting(markdownStyle),
      theme,
      hitSlot.field,
      placeholder("Open a file from Files, or click a search result."),
      keymap.of([{ key: "Mod-Enter", run: () => (addReaderSelection(), true) }]),
      EditorView.updateListener.of((u) => {
        if (u.selectionSet) $("#reader-add").hidden = u.state.selection.main.empty;
      }),
    ],
  });
}

function taskExtensions() {
  return [
    history(),
    drawSelection(),
    highlightActiveLine(),
    EditorView.lineWrapping,
    markdown(),
    syntaxHighlighting(markdownStyle),
    theme,
    addedSlot.field,
    placeholder(
      "Collect passages here: select text in the reader and press Ctrl/⌘+Enter, or use + on a result. " +
        "They go in at this cursor. Write your own notes around them: the Related panel reads this whole document."
    ),
    keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
    EditorView.updateListener.of(onTaskUpdate),
  ];
}

const reader = new EditorView({ parent: $("#reader-editor"), state: readerState("") });
const taskView = new EditorView({
  parent: $("#task-editor"),
  state: EditorState.create({
    doc: "",
    extensions: [theme, EditorState.readOnly.of(true), placeholder("Create a task with + to start collecting.")],
  }),
});

// ---------- state ----------

const state = {
  task: null, // active task name
  tasks: [], // [{name, mtime}]
  docs: new Map(), // task name -> {state, mtime, dirty, saving, again}
  readerPath: null,
  files: [],
  panel: "search",
  relatedStale: true,
  relatedSeq: 0,
  searchSeq: 0,
  lastRelatedKeys: null,
  indexUpdated: null,
};

// ---------- tasks ----------

function renderTaskTabs() {
  const list = $("#task-tabs");
  list.replaceChildren();
  if (!state.tasks.length) {
    list.append(h("li", { class: "tasks-empty" }, "No tasks yet. Press + to create one."));
    return;
  }
  for (const { name } of state.tasks) {
    const active = name === state.task;
    list.append(
      h("li", { class: `task-tab${active ? " active" : ""}`, role: "presentation" },
        h("button", { class: "name", role: "tab", "aria-selected": String(active), title: name, onclick: () => switchTask(name).catch(fail) }, name),
        h("span", { class: "tab-actions" },
          h("button", { title: "Rename task", "aria-label": `Rename ${name}`, onclick: () => renameTask(name).catch(fail) }, "✎"),
          h("button", { title: "Delete task (moved to the tasks folder's .trash)", "aria-label": `Delete ${name}`, onclick: () => deleteTask(name).catch(fail) }, "×"),
        ),
      ),
    );
  }
}

async function loadTasks() {
  state.tasks = await api("GET", "/api/tasks");
  renderTaskTabs();
}

async function switchTask(name) {
  if (state.task === name) return;
  if (state.task && state.docs.has(state.task)) {
    state.docs.get(state.task).state = taskView.state;
    await saveTask(state.task);
  }
  let doc = state.docs.get(name);
  if (!doc) {
    const task = await api("GET", `/api/tasks/${enc(name)}`);
    doc = { state: EditorState.create({ doc: task.content, extensions: taskExtensions() }), mtime: task.mtime, dirty: false };
    state.docs.set(name, doc);
  }
  state.task = name;
  state.lastRelatedKeys = null;
  store.set("task", name);
  taskView.setState(doc.state);
  $("#task-name").textContent = name;
  $("#conflict").hidden = true;
  setSaveState(doc.dirty ? "Unsaved" : "Saved");
  renderTaskTabs();
  requestRelated(0);
}

function showNoTask() {
  state.task = null;
  $("#task-name").textContent = "No task";
  setSaveState("");
  taskView.setState(
    EditorState.create({ doc: "", extensions: [theme, EditorState.readOnly.of(true), placeholder("Create a task with + to start collecting.")] }),
  );
  renderTaskTabs();
  renderRelatedEmpty("Create a task to see related passages.");
}

async function newTask() {
  const name = prompt("Name of the new task:");
  if (!name || !name.trim()) return null;
  const task = await api("POST", "/api/tasks", { name });
  await loadTasks();
  await switchTask(task.name);
  return task.name;
}

async function renameTask(name) {
  const newName = prompt("Rename task:", name);
  if (!newName || newName.trim() === name) return;
  await saveTask(name);
  const task = await api("POST", `/api/tasks/${enc(name)}/rename`, { new_name: newName });
  const doc = state.docs.get(name);
  state.docs.delete(name);
  if (doc) state.docs.set(task.name, { ...doc, mtime: task.mtime });
  if (state.task === name) {
    state.task = task.name;
    store.set("task", task.name);
    $("#task-name").textContent = task.name;
  }
  await loadTasks();
}

async function deleteTask(name) {
  if (!confirm(`Delete the task “${name}”? The file is moved to the tasks folder's .trash.`)) return;
  await api("DELETE", `/api/tasks/${enc(name)}`);
  state.docs.delete(name);
  await loadTasks();
  if (state.task === name) {
    state.task = null;
    if (state.tasks.length) await switchTask(state.tasks[0].name);
    else showNoTask();
  }
}

// ---------- saving ----------

function setSaveState(text) {
  $("#save-state").textContent = text;
}

const scheduleSave = debounce(() => saveTask(state.task).catch(fail), 800);

async function saveTask(name, { keepalive = false } = {}) {
  const doc = name && state.docs.get(name);
  if (!doc) return;
  if (doc.saving) {
    doc.again = true;
    return doc.saving;
  }
  if (!doc.dirty) return;
  const content = (name === state.task ? taskView.state : doc.state).doc.toString();
  doc.dirty = false;
  if (name === state.task) setSaveState("Saving…");
  doc.saving = (async () => {
    try {
      const result = await api("PUT", `/api/tasks/${enc(name)}`, { content, base_mtime: doc.mtime }, { keepalive });
      doc.mtime = result.mtime;
      if (name === state.task) setSaveState(doc.dirty ? "Unsaved" : "Saved");
    } catch (error) {
      doc.dirty = true;
      if (error.status === 409) {
        doc.conflictMtime = error.data.mtime;
        if (name === state.task) {
          $("#conflict").hidden = false;
          setSaveState("Not saved");
        }
      } else {
        setSaveState("Save failed");
        throw error;
      }
    }
  })();
  try {
    await doc.saving;
  } finally {
    doc.saving = null;
  }
  if (doc.again && !doc.conflictMtime) {
    doc.again = false;
    return saveTask(name);
  }
}

async function resolveConflict(keepMine) {
  const name = state.task;
  const doc = state.docs.get(name);
  if (keepMine) {
    doc.mtime = doc.conflictMtime;
    doc.conflictMtime = null;
    doc.dirty = true;
    await saveTask(name);
  } else {
    const task = await api("GET", `/api/tasks/${enc(name)}`);
    doc.state = EditorState.create({ doc: task.content, extensions: taskExtensions() });
    doc.mtime = task.mtime;
    doc.dirty = false;
    doc.conflictMtime = null;
    taskView.setState(doc.state);
    setSaveState("Saved");
  }
  $("#conflict").hidden = true;
  requestRelated(0);
}

// ---------- task editor events ----------

// The whole document is the related query, so only edits refresh it: moving the cursor or
// selecting text does not.
function onTaskUpdate(update) {
  if (!state.task || update.state.readOnly || !update.docChanged) return;
  const doc = state.docs.get(state.task);
  if (doc) doc.dirty = true;
  setSaveState("Unsaved");
  scheduleSave();
  requestRelated(1500);
}

// ---------- adding to the task ----------

async function addToTask(text, origin = null) {
  if (!text.trim()) return;
  if (!state.task) {
    const created = await newTask();
    if (!created) return;
  }
  const doc = taskView.state.doc.toString();
  const plan = planInsertion(doc, taskView.state.selection.main.head, text);
  const start = plan.from + plan.insert.length - plan.insert.trimStart().length;
  taskView.dispatch({
    changes: { from: plan.from, insert: plan.insert },
    selection: EditorSelection.cursor(plan.cursorAfter),
    effects: [
      EditorView.scrollIntoView(plan.cursorAfter, { y: "center" }),
      addedSlot.set.of(Decoration.set([Decoration.mark({ class: "cm-just-added" }).range(start, plan.cursorAfter)])),
    ],
    userEvent: "input.paste",
  });
  setTimeout(() => taskView.dispatch({ effects: addedSlot.set.of(Decoration.none) }), 1600);
  flash(`Added to “${state.task}”`);
  if (origin?.lane) {
    api("POST", `/api/tasks/${enc(state.task)}/feedback`, { action: "add", lane: origin.lane, path: origin.path }).catch(fail);
  }
}

function addReaderSelection() {
  const sel = reader.state.selection.main;
  if (sel.empty) return;
  addToTask(reader.state.sliceDoc(sel.from, sel.to)).catch(fail);
}

// ---------- reader ----------

async function openFile(path, location = null) {
  if (path !== state.readerPath) {
    const { content } = await api("GET", `/api/file?path=${enc(path)}`);
    reader.setState(readerState(content));
    state.readerPath = path;
    $("#reader-path").textContent = path;
    $("#reader-path").title = path;
    $("#reader-add").hidden = true;
    markCurrentFile();
  }
  if (!location) {
    reader.dispatch({ effects: hitSlot.set.of(Decoration.none) });
    return;
  }
  const doc = reader.state.doc;
  const line = doc.line(Math.min(location.line + 1, doc.lines));
  const from = line.from + Math.min(location.ch ?? 0, line.length);
  const to = Math.min(from + (location.len ?? 0), line.to);
  const decos = [];
  if (location.start !== undefined) {
    for (let n = location.start + 1; n <= Math.min(location.end + 1, doc.lines); n++) {
      decos.push(Decoration.line({ class: "cm-hit-line" }).range(doc.line(n).from));
    }
  }
  if (to > from) decos.push(Decoration.mark({ class: "cm-hit-match" }).range(from, to));
  reader.dispatch({
    selection: EditorSelection.cursor(from),
    effects: [hitSlot.set.of(Decoration.set(decos, true)), EditorView.scrollIntoView(from, { y: "center" })],
  });
}

// ---------- files panel ----------

const MAX_FILES_SHOWN = 2000;

function renderFiles() {
  const filter = $("#file-filter").value.trim().toLowerCase();
  const matches = state.files.filter((f) => !filter || f.path.toLowerCase().includes(filter));
  const list = $("#file-list");
  list.replaceChildren();
  let folder = null;
  for (const { path } of matches.slice(0, MAX_FILES_SHOWN)) {
    const slash = path.lastIndexOf("/");
    const dir = slash === -1 ? "" : path.slice(0, slash);
    if (dir !== folder) {
      folder = dir;
      if (dir) list.append(h("li", { class: "folder" }, `${dir}/`));
    }
    list.append(
      h("li", { class: `file${path === state.readerPath ? " current" : ""}`, "data-path": path, title: path, onclick: () => openFile(path).catch(fail) },
        path.slice(slash + 1).replace(/\.(md|markdown)$/i, "")),
    );
  }
  if (matches.length > MAX_FILES_SHOWN) {
    list.append(h("li", { class: "folder" }, `…and ${matches.length - MAX_FILES_SHOWN} more: narrow the filter.`));
  }
  if (!matches.length) list.append(h("li", { class: "folder" }, state.files.length ? "No matching files." : "No markdown files indexed yet."));
}

function markCurrentFile() {
  for (const el of document.querySelectorAll("#file-list .file")) el.classList.toggle("current", el.dataset.path === state.readerPath);
}

async function loadFiles() {
  state.files = await api("GET", "/api/files");
  renderFiles();
}

// ---------- results ----------

// Snippet lines flow as prose (hard-wrapped text would otherwise wrap twice in a narrow
// panel); blank lines start a new paragraph. Context lines around the passage are dimmed.
function renderSnippet(hit) {
  const body = h("div", { class: "hit-body" });
  let para = null;
  for (const line of hit.snippet) {
    if (!line.text.trim()) {
      para = null;
      continue;
    }
    if (!para) body.append((para = h("p")));
    const span = h("span", { class: line.in_passage ? "line" : "line ctx" });
    let pos = 0;
    for (const [start, end] of line.marks) {
      if (start > pos) span.append(line.text.slice(pos, start));
      span.append(h("mark", {}, line.text.slice(start, end)));
      pos = end;
    }
    span.append(line.text.slice(pos));
    para.append(span, " ");
  }
  return body;
}

const WHY_LABELS = {
  semantic_z: (v) => `meaning z ${v}`,
  lexical_rank: (v) => `keyword rank ${v}`,
  obviousness: (v) => `keyword overlap ${Math.round(v * 100)}%`,
  themes: (v) => v.join(" ↔ "),
  rarity: (v) => `themes rarely meet (${Math.round(v * 100)}%)`,
  collected_in_theme: (v) => `${v} collected in theme`,
  matches_task_line: (v) => `closest to task line ${v + 1}`,
  cosine: (v) => `cosine ${v.toFixed(2)}`,
};

// For search hits found with "meaning" ticked: where each signal ranked the passage.
function describeSignals(why) {
  if (!("meaning_rank" in why)) return "";
  return [
    why.keyword_rank ? `keywords #${why.keyword_rank}` : "not a top keyword match",
    why.meaning_rank ? `meaning #${why.meaning_rank}` : "not a top meaning match",
  ].join(" · ");
}

function describeWhy(why) {
  return Object.entries(why)
    .filter(([k, v]) => v !== null && WHY_LABELS[k])
    .map(([k, v]) => WHY_LABELS[k](v))
    .join(" · ");
}

function renderHit(hit, { lane = null, isNew = false } = {}) {
  const file = hit.path.slice(hit.path.lastIndexOf("/") + 1).replace(/\.(md|markdown)$/i, "");
  const open = (event) => {
    if (event.target.closest("button")) return;
    openFile(hit.path, {
      line: hit.focus_line,
      ch: hit.match ? hit.match[1] : 0,
      len: hit.match ? hit.match[2] : 0,
      start: hit.start_line,
      end: hit.end_line,
    }).catch(fail);
  };
  const actions = h("span", { class: "hit-actions" },
    h("button", { class: "add", title: "Add this passage to the task, at its cursor", onclick: () => addToTask(hit.text, lane && { lane, path: hit.path }).catch(fail) }, "+"),
    lane && h("button", { title: "Not useful: hide it and steer away from it", onclick: (e) => dismiss(hit, lane, e.target.closest(".hit")).catch(fail) }, "×"),
  );
  const why = lane ? describeWhy(hit.why) : describeSignals(hit.why);
  return h("article", { class: `hit${isNew ? " is-new" : ""}`, onclick: open, title: lane ? JSON.stringify(hit.why, null, 1) : null },
    h("div", { class: "hit-head" }, h("span", { class: "hit-title" }, hit.heading || file), actions),
    h("div", { class: "hit-loc" }, `${hit.path}:${hit.focus_line + 1}`),
    renderSnippet(hit),
    why && h("div", { class: "hit-why" }, why),
  );
}

// ---------- search panel ----------

const runSearch = debounce(async () => {
  const q = $("#search-input").value;
  const mode = $("#search-hybrid").checked ? "hybrid" : "lexical";
  const seq = ++state.searchSeq;
  if (!q.trim()) {
    $("#search-status").textContent = "";
    $("#search-results").replaceChildren();
    return;
  }
  try {
    const result = await api("GET", `/api/search?q=${enc(q)}&mode=${mode}&limit=50`);
    if (seq !== state.searchSeq) return;
    const n = result.hits.length;
    const how = mode === "hybrid" ? " by keywords and meaning" : "";
    $("#search-status").textContent = n ? `${n}${n === 50 ? "+" : ""} passages${how} · ${result.ms} ms` : "No matches.";
    $("#search-results").replaceChildren(...result.hits.map((hit) => renderHit(hit)));
  } catch (error) {
    fail(error);
  }
}, 200);

// ---------- related panel ----------

let relatedTimer;

// Refresh the related panel after `delay` ms, or when it is next shown.
function requestRelated(delay) {
  state.relatedStale = true;
  clearTimeout(relatedTimer);
  if (state.panel === "related") relatedTimer = setTimeout(() => refreshRelated().catch(fail), delay);
}

function renderRelatedEmpty(message) {
  $("#related-status").textContent = message;
  $("#related-terms").replaceChildren();
  $("#related-lanes").replaceChildren();
}

async function refreshRelated() {
  if (!state.task) {
    renderRelatedEmpty("Create a task to see related passages.");
    return;
  }
  const name = state.task;
  const seq = ++state.relatedSeq;
  state.relatedStale = false;
  $("#related-status").textContent = "Updating…";
  const result = await api("POST", `/api/tasks/${enc(name)}/related`, { text: taskView.state.doc.toString() });
  if (seq !== state.relatedSeq || name !== state.task) return;

  const previous = state.lastRelatedKeys;
  const keys = new Set();
  const lanes = result.lanes.map((lane) =>
    h("section", { class: "lane" },
      h("h3", {}, lane.title, h("span", { class: "count" }, String(lane.items.length))),
      h("p", { class: "lane-desc" }, lane.description),
      lane.note && h("p", { class: "lane-note" }, lane.note),
      lane.items.map((hit) => {
        const key = `${hit.path}:${hit.start_line}`;
        keys.add(key);
        return renderHit(hit, { lane: lane.name, isNew: previous !== null && !previous.has(key) });
      }),
    ),
  );
  state.lastRelatedKeys = keys;
  renderChips(result.terms);
  $("#related-lanes").replaceChildren(...lanes);
  const s = result.stats;
  const themes = result.themes.map((t) => t.label).join(" · ");
  $("#related-status").textContent = s.gated === undefined
    ? (result.lanes[0]?.note || "")
    : `${s.gated} clearly related of ${s.passages.toLocaleString()} passages${themes ? ` · themes: ${themes}` : ""}`;
}

function renderChips(terms) {
  $("#related-terms").replaceChildren(
    ...terms.map((t) =>
      h("span", { class: `chip${t.pinned ? " pinned" : ""}${t.banned ? " banned" : ""}`, style: `--w: ${t.weight.toFixed(2)}` },
        h("button", {
          class: "chip-label",
          title: t.banned ? "Banned: click to use it again" : t.pinned ? "Pinned: click to unpin" : "Click to pin: always search for this term",
          onclick: () => termAction(t.term, t.banned ? "unban" : t.pinned ? "unpin" : "pin").catch(fail),
        }, t.label),
        !t.banned && h("button", { class: "chip-x", title: "Ban: never search for this term in this task", onclick: () => termAction(t.term, "ban").catch(fail) }, "×"),
      ),
    ),
  );
}

async function termAction(term, action) {
  await api("POST", `/api/tasks/${enc(state.task)}/terms`, { term, action });
  await refreshRelated();
}

async function dismiss(hit, lane, element) {
  element?.remove();
  await api("POST", `/api/tasks/${enc(state.task)}/feedback`, { action: "dismiss", lane, path: hit.path, text: hit.text });
  requestRelated(400);
}

// ---------- panels ----------

function showPanel(name) {
  state.panel = name;
  store.set("panel", name);
  for (const button of document.querySelectorAll(".side-tabs button")) {
    button.setAttribute("aria-selected", String(button.dataset.panel === name));
  }
  for (const panel of ["files", "search", "related"]) $(`#panel-${panel}`).hidden = panel !== name;
  if (name === "related" && state.relatedStale) refreshRelated().catch(fail);
  if (name === "search") $("#search-input").focus();
}

// ---------- index status ----------

async function pollStatus() {
  let ready = true;
  try {
    const s = await api("GET", "/api/status");
    ready = s.state === "ready";
    const model = s.model ? s.model.split("/").pop() : "keywords only";
    let text = `${s.files.toLocaleString()} files · ${s.passages.toLocaleString()} passages · ${model}`;
    if (s.state === "embedding") text = `Embedding ${s.embedded.toLocaleString()} / ${s.to_embed.toLocaleString()} passages… (search works meanwhile)`;
    else if (s.state !== "ready") text = "Indexing…";
    $("#index-status").textContent = text;
    if (state.indexUpdated !== null && s.updated !== state.indexUpdated) {
      loadFiles().catch(fail);
      requestRelated(500);
    }
    state.indexUpdated = s.updated;
  } catch {
    $("#index-status").textContent = "Server not reachable";
    ready = false;
  }
  setTimeout(pollStatus, ready ? 10000 : 2000);
}

// ---------- wiring ----------

function bind() {
  $("#new-task").addEventListener("click", () => newTask().catch(fail));
  $("#reader-add").addEventListener("click", addReaderSelection);
  $("#conflict-load").addEventListener("click", () => resolveConflict(false).catch(fail));
  $("#conflict-keep").addEventListener("click", () => resolveConflict(true).catch(fail));
  $("#file-filter").addEventListener("input", debounce(renderFiles, 120));
  $("#search-input").addEventListener("input", runSearch);
  $("#search-hybrid").addEventListener("change", runSearch);
  for (const button of document.querySelectorAll(".side-tabs button")) {
    button.addEventListener("click", () => showPanel(button.dataset.panel));
  }
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      showPanel("search");
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") saveTask(state.task, { keepalive: true }).catch(() => {});
  });
  window.addEventListener("beforeunload", (e) => {
    if ([...state.docs.values()].some((d) => d.dirty)) e.preventDefault();
  });
}

async function init() {
  bind();
  showPanel(store.get("panel") || "search");
  await Promise.all([loadTasks(), loadFiles()]);
  const last = store.get("task");
  const name = state.tasks.find((t) => t.name === last)?.name ?? state.tasks[0]?.name;
  if (name) await switchTask(name);
  else showNoTask();
  pollStatus();
}

init().catch(fail);

// Exposed for debugging and end-to-end tests.
window.isis = { state, reader, taskView, openFile, addToTask, refreshRelated };
