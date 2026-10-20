import {
  EditorState, EditorSelection, StateField, StateEffect,
  EditorView, Decoration, WidgetType, keymap, placeholder, drawSelection, highlightActiveLine, lineNumbers,
  history, defaultKeymap, historyKeymap, indentWithTab,
  syntaxHighlighting, HighlightStyle, markdown, tags,
} from "./vendor/codemirror.js";
import { planInsertion } from "./insert.js";
import { paragraphAt, softBreaks } from "./reflow.js";

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
  ".cm-hit-range": { backgroundColor: "var(--hit-line)" },
  ".cm-soft-break": { whiteSpace: "normal" }, // a joined line break: no stray space at the start of a wrapped line
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

// Hard-wrapped paragraphs flow to the width of the pane: their line breaks show as spaces (see
// reflow.js). In the task editor the paragraph being edited keeps its real lines while the
// editor has focus, so pressing Enter visibly starts a new line.
class SpaceWidget extends WidgetType {
  eq() { return true; }
  toDOM() {
    const span = document.createElement("span");
    span.className = "cm-soft-break";
    span.textContent = " ";
    return span;
  }
}
const softBreak = Decoration.replace({ widget: new SpaceWidget() });
const setFocused = StateEffect.define();

function reflow({ editable }) {
  const build = (state, focused) => {
    const lines = state.doc.toJSON();
    const [lo, hi] = editable && focused ? paragraphAt(lines, state.doc.lineAt(state.selection.main.head).number - 1) : [-1, -1];
    const decos = [];
    for (const i of softBreaks(lines)) {
      if (i >= lo && i < hi) continue;
      const end = state.doc.line(i + 1).to;
      decos.push(softBreak.range(end, end + 1));
    }
    return Decoration.set(decos);
  };
  // Line-break replacements must come from a state field, not a view plugin.
  const field = StateField.define({
    create: (state) => ({ focused: false, decos: build(state, false) }),
    update(value, tr) {
      let focused = value.focused;
      for (const e of tr.effects) if (e.is(setFocused)) focused = e.value;
      if (!tr.docChanged && focused === value.focused && !(editable && focused && tr.selection)) return value;
      return { focused, decos: build(tr.state, focused) };
    },
    provide: (f) => EditorView.decorations.from(f, (value) => value.decos),
  });
  return editable ? [field, EditorView.focusChangeEffect.of((_, focusing) => setFocused.of(focusing))] : field;
}

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
      reflow({ editable: false }),
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
    reflow({ editable: true }),
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
  paramSpecs: { search: [], related: [] }, // from /api/params
  params: loadParams(), // this browser's overrides: {search: {name: value}, related: {...}}
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
      const body = { content, base_mtime: doc.mtime };
      // Browsers refuse keepalive requests over 64 KiB; a larger task saves as a normal request.
      const small = new TextEncoder().encode(JSON.stringify(body)).length < 60000;
      const result = await api("PUT", `/api/tasks/${enc(name)}`, body, { keepalive: keepalive && small });
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
    // The passage itself, exactly: it may start or end inside a paragraph, which flows as one line.
    const start = doc.line(Math.min(location.start + 1, doc.lines)).from;
    const end = doc.line(Math.min(location.end + 1, doc.lines)).to;
    if (end > start) decos.push(Decoration.mark({ class: "cm-hit-range" }).range(start, end));
  }
  const words = location.marks ?? (to > from ? [[location.line, location.ch, location.len + location.ch]] : []);
  for (const [n, s, e] of words) {
    if (n >= doc.lines) continue;
    const l = doc.line(n + 1);
    const a = l.from + Math.min(s, l.length);
    const b = l.from + Math.min(e, l.length);
    if (b > a) decos.push(Decoration.mark({ class: "cm-hit-match" }).range(a, b));
  }
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
      marks: hit.marks,
    }).catch(fail);
  };
  const actions = h("span", { class: "hit-actions" },
    h("button", { class: "add", title: "Add this passage to the task, at its cursor", onclick: () => addToTask(hit.text, lane && { lane, path: hit.path }).catch(fail) }, "+"),
    lane && h("button", { title: "Not useful: hide it", onclick: (e) => dismiss(hit, lane, e.target.closest(".hit")).catch(fail) }, "×"),
    lane && h("button", { class: "avoid", title: "Avoid: hide it and every passage like it", onclick: (e) => dismiss(hit, lane, e.target.closest(".hit"), true).catch(fail) }, "⊘"),
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
    const tuned = Object.keys(state.params.search).length ? `&params=${enc(JSON.stringify(state.params.search))}` : "";
    const result = await api("GET", `/api/search?q=${enc(q)}&mode=${mode}&limit=50${tuned}`);
    if (seq !== state.searchSeq) return;
    const n = result.hits.length;
    const how = mode === "hybrid" ? " by keywords and meaning" : "";
    $("#search-status").textContent = n ? `${n}${n === 50 ? "+" : ""} passages${how} · ${result.ms} ms` : "No matches.";
    $("#search-results").replaceChildren(...result.hits.map((hit) => renderHit(hit)));
  } catch (error) {
    if (seq !== state.searchSeq) return;
    if (error.status !== 400) fail(error);
    $("#search-status").textContent = error.status === 400 ? `⚠ ${error.message}` : "";
    $("#search-results").replaceChildren();
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
  $("#related-feedback").hidden = true;
  $("#related-steer").hidden = true;
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
  const result = await api("POST", `/api/tasks/${enc(name)}/related`, {
    text: taskView.state.doc.toString(),
    params: state.params.related,
  });
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
  renderSteer(result.prefs.steer);
  renderFeedback(result.prefs, result.terms, result.lanes);
  $("#related-lanes").replaceChildren(...lanes);
  const s = result.stats;
  const themes = result.themes.map((t) => t.label).join(" · ");
  $("#related-status").textContent = s.gated === undefined
    ? (result.lanes[0]?.note || "")
    : `${s.gated} clearly related of ${s.passages.toLocaleString()} passages${themes ? ` · themes: ${themes}` : ""}`;
}

// The keywords taken from the task. Boosting writes the word into the steer line; ignoring takes
// a keyword out of the query without hiding the passages that contain it.
function renderChips(terms) {
  $("#related-terms").replaceChildren(
    ...terms.map((t) =>
      h("span", { class: `chip${t.boosted ? " boosted" : ""}${t.ignored ? " ignored" : ""}`, style: `--w: ${t.weight.toFixed(2)}` },
        h("button", {
          class: "chip-label",
          title: t.ignored
            ? "Ignored: click to search for it again"
            : t.boosted
              ? "Boosted by the steer line: click to stop boosting"
              : "Click to boost: always search for it, at full weight (adds it to the steer line)",
          onclick: () => (t.ignored ? termAction(t.term, "unignore") : t.boosted ? termAction(t.term, "unboost") : termAction(t.label, "boost")).catch(fail),
        }, t.label),
        t.ignored
          ? h("button", { class: "chip-x", title: `Exclude: never show passages containing “${t.label}” (adds -${t.label} to the steer line)`, onclick: () => termAction(t.label, "exclude").catch(fail) }, "⊘")
          : h("button", { class: "chip-x", title: "Ignore: stop searching for this word (passages containing it can still be suggested)", onclick: () => termAction(t.term, "ignore").catch(fail) }, "×"),
      ),
    ),
  );
}

async function termAction(term, action) {
  await api("POST", `/api/tasks/${enc(state.task)}/terms`, { term, action });
  await refreshRelated();
}

// The steer line: clauses as in a search, kept with the task. Enter saves; Escape undoes the edit.
function renderSteer(steer) {
  $("#related-steer").hidden = false;
  const input = $("#steer-input");
  input.dataset.saved = steer;
  if (document.activeElement !== input) input.value = steer;
}

async function saveSteer(event) {
  event.preventDefault();
  const input = $("#steer-input");
  if (!state.task || input.value.trim() === (input.dataset.saved ?? "")) return;
  try {
    const prefs = await api("PUT", `/api/tasks/${enc(state.task)}/steer`, { text: input.value });
    input.value = input.dataset.saved = prefs.steer;
    $("#steer-error").hidden = true;
    await refreshRelated();
  } catch (error) {
    if (error.status !== 400) throw error;
    $("#steer-error").textContent = `⚠ ${error.message}`;
    $("#steer-error").hidden = false;
  }
}

async function dismiss(hit, lane, element, avoid = false) {
  element?.remove();
  await api("POST", `/api/tasks/${enc(state.task)}/feedback`, {
    action: avoid ? "avoid" : "dismiss", lane, path: hit.path, text: hit.text, start_line: hit.start_line, end_line: hit.end_line,
  });
  requestRelated(400);
}

// ---------- feedback drawer ----------

// Everything the panel remembers about the task, each with its undo. It stays with the task
// when the text changes, so it should never be out of sight.
function renderFeedback(prefs, terms, lanes) {
  const drawer = $("#related-feedback");
  drawer.hidden = false;
  const labels = new Map(terms.map((t) => [t.term, t.label]));
  const ignored = prefs.ignored;
  const avoided = prefs.dismissed.filter((d) => d.avoid).length;
  const key = JSON.stringify([state.task, prefs, ignored.map((t) => labels.get(t))]);
  if (drawer.dataset.key === key) return; // keep the list's scroll position while typing
  drawer.dataset.key = key;

  const laneTitles = new Map(lanes.map((l) => [l.name, l.title]));
  const counted = Object.entries(prefs.feedback).filter(([, [adds, dismisses]]) => adds || dismisses);
  const added = counted.reduce((sum, [, [adds]]) => sum + adds, 0);
  const summary = [
    prefs.dismissed.length - avoided && `${prefs.dismissed.length - avoided} dismissed`,
    avoided && `${avoided} avoided`,
    prefs.steer && "steered",
    added && `${added} added`,
    ignored.length && `${ignored.length} ignored`,
  ].filter(Boolean);
  const empty = !prefs.dismissed.length && !prefs.steer && !ignored.length && !counted.length;

  const dismissed = [...prefs.dismissed].reverse().map((d) => {
    const known = d.start_line !== undefined && d.start_line !== null;
    const location = known ? { line: d.start_line, start: d.start_line, end: d.end_line ?? d.start_line } : null;
    const text = d.text.replace(/\s+/g, " ").trim();
    const where = known ? `${d.path}:${d.start_line + 1}` : d.path;
    return h("li", { class: "feedback-item" },
      h("div", { class: "feedback-head" },
        h("button", { class: "feedback-open", title: `Open ${where} in the reader`, onclick: () => openFile(d.path, location).catch(fail) }, where),
        h("button", { title: "Let it be suggested again, and take back its dismissal", onclick: (e) => restoreDismissed(d, e.target.closest("li")).catch(fail) }, "Restore"),
      ),
      h("div", { class: "feedback-text" },
        d.avoid && h("span", { class: "feedback-tag", title: "Avoided: passages like this one are hidden too" }, "⊘ avoided"),
        text.length > 160 ? `${text.slice(0, 160).trimEnd()}…` : text),
    );
  });
  const ignoredChip = (term) =>
    h("span", { class: "chip ignored" },
      h("span", { class: "chip-label" }, labels.get(term) ?? term),
      h("button", { class: "chip-x", title: "Search for it again", onclick: () => termAction(term, "unignore").catch(fail) }, "×"),
    );
  const laneCounts = counted.map(([lane, [adds, dismisses]]) =>
    `${laneTitles.get(lane) ?? lane}: ${[adds && `${adds} added`, dismisses && `${dismisses} dismissed`].filter(Boolean).join(", ")}`);

  drawer.replaceChildren(
    h("summary", {}, "Feedback", h("span", { class: "tune-count" }, summary.length ? ` · ${summary.join(" · ")}` : "")),
    h("div", { class: "tune-body" },
      empty && h("p", { class: "feedback-note" },
        "Nothing yet. The passages you add (+), dismiss (×) or avoid (⊘) here, the steer line, and the keywords you ignore are remembered for this task, whatever its text."),
      dismissed.length > 0 && [
        h("div", { class: "tune-group" }, "Dismissed and avoided passages"),
        h("p", { class: "feedback-note" }, "Never suggested again. Passages like an avoided one are hidden too (Parameters › Avoid: similarity), and Parameters › Dismissal push can also rank passages like dismissed ones lower."),
        h("ul", { class: "feedback-list" }, dismissed),
      ],
      prefs.steer && [
        h("div", { class: "tune-group" }, "Steer line"),
        h("code", { class: "feedback-steer" }, prefs.steer),
        h("p", { class: "feedback-note" }, "Edit it in the box above the suggestions."),
      ],
      ignored.length > 0 && [
        h("div", { class: "tune-group" }, "Ignored keywords"),
        h("p", { class: "feedback-note" }, "Not searched for; passages containing them can still be suggested."),
        h("div", { class: "chips" }, ignored.map(ignoredChip)),
      ],
      counted.length > 0 && [
        h("div", { class: "tune-group" }, "Lane feedback"),
        h("p", { class: "feedback-note" }, `${laneCounts.join(" · ")}. Lanes you add from get more of the results.`),
      ],
      !empty && h("div", { class: "tune-foot" },
        h("button", { onclick: () => forgetFeedback().catch(fail) }, "Forget all"),
        h("p", { class: "feedback-note" }, "Clears all of the above for this task. A copy goes to the tasks folder's .trash."),
      ),
    ),
  );
}

async function restoreDismissed(dismissal, element) {
  element?.remove();
  await api("POST", `/api/tasks/${enc(state.task)}/feedback`, { action: "restore", id: dismissal.id });
  await refreshRelated();
}

async function forgetFeedback() {
  const name = state.task;
  if (!confirm(`Forget all feedback for “${name}”? Dismissed and avoided passages can come back, the steer line and ignored keywords are cleared, and the lanes start even.`)) return;
  await api("DELETE", `/api/tasks/${enc(name)}/prefs`);
  flash(`Forgot the feedback for “${name}”`);
  await refreshRelated();
}

// ---------- tunable parameters ----------

// Each panel has a drawer of sliders built from /api/params. Overrides live in this browser and
// travel with every request, so the server's defaults never change.

function loadParams() {
  try {
    const saved = JSON.parse(store.get("params") || "{}");
    return { search: saved.search || {}, related: saved.related || {} };
  } catch {
    return { search: {}, related: {} };
  }
}

function saveParams() {
  store.set("params", JSON.stringify(state.params));
}

function formatValue(spec, value) {
  if (spec.type !== "float") return String(value);
  const digits = String(spec.step).split(".")[1]?.length ?? 0;
  return Number(value).toFixed(digits);
}

const pythonValue = (v) => (typeof v === "string" ? JSON.stringify(v) : String(v));

// The same settings as Python, e.g. RelatedConfig(z_strong=2.0), for the notebook.
function pythonSnippet(kind) {
  const overrides = state.params[kind];
  const specs = state.paramSpecs[kind];
  const args = [];
  const done = new Set();
  for (const spec of specs) {
    const [field, key] = spec.name.split(".");
    if (!key) {
      if (spec.name in overrides) args.push(`${field}=${pythonValue(overrides[spec.name])}`);
      continue;
    }
    if (done.has(field)) continue;
    done.add(field);
    const members = specs.filter((s) => s.name.startsWith(`${field}.`));
    if (!members.some((s) => s.name in overrides)) continue;
    const entries = members.map((s) => `"${s.name.slice(field.length + 1)}": ${pythonValue(overrides[s.name] ?? s.default)}`);
    args.push(`${field}={${entries.join(", ")}}`);
  }
  return `${kind === "search" ? "SearchConfig" : "RelatedConfig"}(${args.join(", ")})`;
}

function paramRow(kind, spec) {
  const value = state.params[kind][spec.name] ?? spec.default;
  const output = h("output", {}, formatValue(spec, value));
  const input = spec.type === "choice"
    ? h("select", {}, spec.choices.map((c) => h("option", { value: c, selected: c === value }, c)))
    : h("input", { type: "range", min: spec.min, max: spec.max, step: spec.step, value });
  input.addEventListener("input", () => {
    const v = spec.type === "choice" ? input.value : Number(input.value);
    output.textContent = formatValue(spec, v);
    if (v === spec.default) delete state.params[kind][spec.name];
    else state.params[kind][spec.name] = v;
    saveParams();
    updateTuner(kind);
    if (kind === "search") runSearch();
    else requestRelated(250);
  });
  return h("label", { class: "tune-row", "data-name": spec.name, title: `${spec.help}\nDefault: ${formatValue(spec, spec.default)}` },
    h("span", { class: "tune-label" }, spec.label), input, spec.type === "choice" ? null : output);
}

function buildTuner(kind) {
  const rows = [];
  let group = null;
  for (const spec of state.paramSpecs[kind]) {
    if (spec.group !== group) rows.push(h("div", { class: "tune-group" }, (group = spec.group)));
    rows.push(paramRow(kind, spec));
  }
  $(`#${kind}-tune`).replaceChildren(
    h("summary", {}, "Parameters", h("span", { class: "tune-count" })),
    h("div", { class: "tune-body" },
      rows,
      h("div", { class: "tune-foot" },
        h("button", { onclick: () => resetParams(kind) }, "Reset to defaults"),
        h("code", { class: "tune-python", title: "The same settings in Python, for notebooks/explore.ipynb" }),
      ),
    ),
  );
  updateTuner(kind);
}

function updateTuner(kind) {
  const el = $(`#${kind}-tune`);
  const changed = Object.keys(state.params[kind]).length;
  el.querySelector(".tune-count").textContent = changed ? ` · ${changed} changed` : "";
  for (const row of el.querySelectorAll(".tune-row")) row.classList.toggle("changed", row.dataset.name in state.params[kind]);
  el.querySelector(".tune-python").textContent = pythonSnippet(kind);
}

function resetParams(kind) {
  state.params[kind] = {};
  saveParams();
  buildTuner(kind);
  if (kind === "search") runSearch();
  else requestRelated(0);
}

async function loadParamSpecs() {
  state.paramSpecs = await api("GET", "/api/params");
  for (const kind of ["search", "related"]) {
    const known = new Set(state.paramSpecs[kind].map((s) => s.name));
    for (const name of Object.keys(state.params[kind])) if (!known.has(name)) delete state.params[kind][name];
    buildTuner(kind);
  }
  saveParams();
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
    else if (s.state === "error") text = `Indexing failed: ${s.error} (see the server log; it retries when files change)`;
    else if (s.state !== "ready") text = "Indexing…";
    if (s.state === "error") ready = true; // nothing to wait for: poll slowly
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

// ---------- resizable panes ----------

// The borders right of the task list, the side panel and the reader can be dragged (or moved with
// the arrow keys when focused). Widths are kept in this browser; double-click a border to reset it.
const PANE_LIMITS = { tasks: [100, 420], side: [220, 900] };

function applyLayout(layout) {
  const app = $("#app");
  for (const name of ["tasks", "side"]) {
    if (layout[name]) app.style.setProperty(`--w-${name}`, `${layout[name]}px`);
    else app.style.removeProperty(`--w-${name}`);
  }
  if (layout.reader) {
    app.style.setProperty("--fr-reader", `${layout.reader}fr`);
    app.style.setProperty("--fr-task", `${1 - layout.reader}fr`);
  } else {
    app.style.removeProperty("--fr-reader");
    app.style.removeProperty("--fr-task");
  }
}

function bindResizers() {
  let layout = {};
  try {
    layout = JSON.parse(store.get("layout") || "{}");
  } catch {}
  applyLayout(layout);
  const save = () => store.set("layout", JSON.stringify(layout));
  for (const handle of document.querySelectorAll(".resizer")) {
    const name = handle.dataset.resize;
    const pane = handle.parentElement;
    const moveBorderTo = (x) => {
      const left = pane.getBoundingClientRect().left;
      if (name === "reader") {
        const total = pane.offsetWidth + $("#task").offsetWidth;
        layout.reader = Math.min(0.85, Math.max(0.15, (x - left) / total));
      } else {
        const [min, max] = PANE_LIMITS[name];
        layout[name] = Math.round(Math.min(max, Math.max(min, x - left)));
      }
      applyLayout(layout);
    };
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
    });
    handle.addEventListener("pointermove", (e) => {
      if (handle.hasPointerCapture(e.pointerId)) moveBorderTo(e.clientX);
    });
    handle.addEventListener("lostpointercapture", () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
      save();
    });
    handle.addEventListener("dblclick", () => {
      delete layout[name];
      applyLayout(layout);
      save();
    });
    handle.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      moveBorderTo(pane.getBoundingClientRect().right + (e.key === "ArrowLeft" ? -16 : 16));
      save();
    });
  }
}

// Drawers stay open or closed across reloads.
function rememberOpen(drawer, key) {
  drawer.open = store.get(key) === "open";
  drawer.addEventListener("toggle", () => store.set(key, drawer.open ? "open" : "closed"));
}

function bind() {
  for (const kind of ["search", "related"]) rememberOpen($(`#${kind}-tune`), `tune-${kind}`);
  rememberOpen($("#related-feedback"), "feedback");
  bindResizers();
  $("#new-task").addEventListener("click", () => newTask().catch(fail));
  $("#reader-add").addEventListener("click", addReaderSelection);
  $("#conflict-load").addEventListener("click", () => resolveConflict(false).catch(fail));
  $("#conflict-keep").addEventListener("click", () => resolveConflict(true).catch(fail));
  $("#file-filter").addEventListener("input", debounce(renderFiles, 120));
  $("#search-input").addEventListener("input", runSearch);
  $("#search-hybrid").addEventListener("change", runSearch);
  $("#related-steer").addEventListener("submit", (e) => saveSteer(e).catch(fail));
  $("#steer-input").addEventListener("input", () => ($("#steer-error").hidden = true));
  $("#steer-input").addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    e.preventDefault(); // a search field would clear itself
    e.target.value = e.target.dataset.saved ?? "";
    $("#steer-error").hidden = true;
  });
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
    // A save still on its way counts as unsaved: closing the page could cancel it.
    if ([...state.docs.values()].some((d) => d.dirty || d.saving)) e.preventDefault();
  });
}

async function init() {
  bind();
  await loadParamSpecs();
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
