import {
  EditorState, EditorSelection, StateField, StateEffect,
  EditorView, Decoration, WidgetType, keymap, placeholder, drawSelection, highlightActiveLine, lineNumbers,
  history, defaultKeymap, historyKeymap, indentWithTab,
  syntaxHighlighting, HighlightStyle, markdown, tags,
} from "./vendor/codemirror.js";
import { ancestors, fileTree, visibleRows } from "./filetree.js";
import { planInsertion } from "./insert.js";
import { envelopeAt, paragraphAt, softBreaks, structure } from "./reflow.js";

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
  ".cm-env": { backgroundColor: "var(--panel)", boxShadow: "inset 3px 0 0 var(--border)" },
  ".cm-env-meta": { fontFamily: "var(--sans)", fontSize: "0.8em", lineHeight: "1.55", color: "var(--muted)" },
  ".cm-env-key": { fontWeight: "600" },
  ".cm-env-edge": { fontSize: "6px", lineHeight: "16px", userSelect: "none" },
  ".cm-env-edge, .cm-env-edge span, .cm-rule, .cm-rule span": { color: "transparent !important" },
  ".cm-rule": {
    backgroundImage: "linear-gradient(var(--border), var(--border))",
    backgroundSize: "100% 1px",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
  },
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

// So the parser agrees with reflow.js: an enveloped block isn't a heading under its closing ---.
const envelopedBlocks = {
  defineNodes: [{ name: "EnvelopeData", block: true, style: tags.meta }],
  parseBlock: [{
    name: "Envelope",
    before: "HorizontalRule",
    parse(cx, line) {
      if (!line.text.startsWith("---") || cx.parentType().name !== "Document") return false;
      const start = cx.lineStart;
      const before = start === 0 ? "" : cx.input.read(Math.max(0, start - 2), start);
      const lines = cx.input.read(start, Math.min(cx.input.length, start + 40000)).split("\n");
      // envelopeAt looks at the line above, and treats line 0 as the top of the file.
      const envelope = start === 0 ? envelopeAt(lines, 0) : before.endsWith("\n\n") ? envelopeAt(["", ...lines], 1) : null;
      if (!envelope) return false;
      const offset = start === 0 ? 0 : 1; // the "" above
      const at = (i) => start + lines.slice(0, i).join("\n").length + (i ? 1 : 0);
      const close = envelope.close - offset;
      const end = at(close) + lines[close].length;
      if (envelope.meta) {
        cx.addElement(cx.elt("EnvelopeData", start, end));
      } else {
        cx.addElement(cx.elt("HorizontalRule", start, start + lines[0].length));
        const from = at(1);
        const to = at(close) - 1;
        cx.addElement(cx.elt("Paragraph", from, to, cx.parser.parseInline(cx.input.read(from, to), from)));
        cx.addElement(cx.elt("HorizontalRule", at(close), end));
      }
      for (let i = 0; i <= close; i++) cx.nextLine();
      return true;
    },
  }],
};

// CodeMirror lays out only what is on screen and estimates the height of the rest (here 12% too
// low). Dragging the scrollbar lands on estimated text, real heights arrive, the position is
// corrected, and the thumb falls behind the mouse by a third. So lay out the whole document once,
// when it opens and when its width changes: heights it has measured, it keeps. Nothing on screen
// moves. This is the mode CodeMirror prints in, viewState.printing, not a public API: if it goes
// away in an update, the scrollbar is merely back to lagging. Very long documents are left alone,
// since this holds up the page (about 0.4 s per 80 KB on a slow CPU).
const MAX_MEASURED_CHARS = 500_000;
const measuredWidths = new WeakMap();

function measureWholeDocument(view) {
  measuredWidths.set(view, view.contentDOM.clientWidth);
  if (view.state.doc.length > MAX_MEASURED_CHARS || !view.dom.offsetHeight) return;
  view.viewState.printing = true;
  view.measure();
  view.viewState.printing = false;
  view.requestMeasure();
}

const remeasureSoon = debounce((view) => measureWholeDocument(view), 200);
const remeasureOnResize = EditorView.updateListener.of((update) => {
  if (update.geometryChanged && measuredWidths.get(update.view) !== update.view.contentDOM.clientWidth) remeasureSoon(update.view);
});

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
    const editing = (i) => i >= lo && i <= hi;
    const decos = [];
    for (const i of softBreaks(lines)) {
      if (i >= lo && i < hi) continue;
      const end = state.doc.line(i + 1).to;
      decos.push(softBreak.range(end, end + 1));
    }
    const { envelopes, rules } = structure(lines);
    for (const { open, close, meta } of envelopes) {
      for (let i = open; i <= close; i++) {
        const line = state.doc.line(i + 1);
        const edge = (i === open || i === close) && !editing(i);
        decos.push(Decoration.line({ class: `cm-env${meta ? " cm-env-meta" : ""}${edge ? " cm-env-edge" : ""}` }).range(line.from));
        const key = meta && i !== open && i !== close && /^\s*[\w.-]+:/.exec(line.text);
        if (key) decos.push(Decoration.mark({ class: "cm-env-key" }).range(line.from, line.from + key[0].length));
      }
    }
    for (const i of rules) {
      if (!editing(i)) decos.push(Decoration.line({ class: "cm-rule" }).range(state.doc.line(i + 1).from));
    }
    return Decoration.set(decos, true);
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
      markdown({ extensions: envelopedBlocks }),
      syntaxHighlighting(markdownStyle),
      theme,
      reflow({ editable: false }),
      hitSlot.field,
      placeholder("Open a file from Files, or click a search result."),
      keymap.of([{ key: "Mod-Enter", run: () => (addReaderSelection(), true) }]),
      remeasureOnResize,
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
    markdown({ extensions: envelopedBlocks }),
    syntaxHighlighting(markdownStyle),
    theme,
    reflow({ editable: true }),
    addedSlot.field,
    placeholder(
      "Collect passages here: select text in the reader and press Ctrl/⌘+Enter, or use + on a result. " +
        "They go in at this cursor. Write your own notes around them: the Related panel reads this whole document."
    ),
    keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
    remeasureOnResize,
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
  openFolders: new Set(), // folders open in the Files panel; all are closed at first
  skipped: { search: loadSkipped("search"), related: loadSkipped("related") }, // corpus folders left out, per panel
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
  measureWholeDocument(taskView);
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
    measureWholeDocument(taskView);
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
    measureWholeDocument(reader);
    state.readerPath = path;
    $("#reader-path").textContent = path;
    $("#reader-path").title = path;
    $("#reader-add").hidden = true;
    revealCurrentFile();
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
  const matches = state.files.map((f) => f.path).filter((path) => !filter || path.toLowerCase().includes(filter));
  // While filtering, every folder that holds a match is open.
  const { rows, more } = visibleRows(fileTree(matches), (folder) => Boolean(filter) || state.openFolders.has(folder), MAX_FILES_SHOWN);
  const indent = (depth, base) => `padding-left: ${base + depth * 14}px`;
  const list = $("#file-list");
  list.replaceChildren();
  for (const row of rows) {
    if (row.kind === "folder") {
      list.append(
        h("li", { class: "dir-row" },
          h("button", { class: "dir", "data-dir": row.path, title: row.path, "aria-expanded": String(row.open), style: indent(row.depth, 4), disabled: Boolean(filter), onclick: () => toggleFolder(row.path) },
            h("span", { class: "dir-arrow" }, row.open ? "▾" : "▸"),
            h("span", { class: "dir-name" }, row.name),
            h("span", { class: "dir-count" }, row.count.toLocaleString()),
          ),
        ),
      );
    } else {
      list.append(
        h("li", { class: `file${row.path === state.readerPath ? " current" : ""}`, "data-path": row.path, title: row.path, style: indent(row.depth, 20), onclick: () => openFile(row.path).catch(fail) },
          row.name.replace(/\.(md|markdown)$/i, "")),
      );
    }
  }
  if (more) list.append(h("li", { class: "folder" }, `…and ${more.toLocaleString()} more: narrow the filter.`));
  if (!matches.length) list.append(h("li", { class: "folder" }, state.files.length ? "No matching files." : "No markdown files indexed yet."));
}

function toggleFolder(folder) {
  if (!state.openFolders.delete(folder)) state.openFolders.add(folder);
  renderFiles();
  $(`#file-list [data-dir="${CSS.escape(folder)}"]`)?.focus();
}

// The file in the reader stays findable: opening one, from anywhere, opens the folders above it.
function revealCurrentFile() {
  const closed = ancestors(state.readerPath).filter((folder) => !state.openFolders.has(folder));
  for (const folder of closed) state.openFolders.add(folder);
  if (closed.length) renderFiles();
  else markCurrentFile();
  $("#file-list .file.current")?.scrollIntoView({ block: "nearest" });
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

// For search hits found with Meaning on: where each signal ranked the passage.
function describeSignals(why) {
  if (!("meaning_rank" in why)) return "";
  return [
    why.keyword_rank ? `keywords #${why.keyword_rank}` : "not a top keyword match",
    why.embedded === false ? "not embedded" : why.meaning_rank ? `meaning #${why.meaning_rank}` : "not a top meaning match",
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
    lane && h("button", {
      class: "like",
      "aria-pressed": String(Boolean(hit.why.liked)),
      title: hit.why.liked ? "More like this is on: click to stop" : "More like this: pull the suggestions toward this passage, without adding it",
      onclick: () => likeHit(hit, lane).catch(fail),
    }, "↑"),
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
  const mode = "hybrid"; // Meaning off is meaning_weight 0, which the server ranks by keywords alone
  const seq = ++state.searchSeq;
  if (!q.trim()) {
    $("#search-status").textContent = "";
    $("#search-results").replaceChildren();
    return;
  }
  try {
    const tuned = Object.keys(state.params.search).length ? `&params=${enc(JSON.stringify(state.params.search))}` : "";
    const skip = state.skipped.search.size ? `&skip=${enc(JSON.stringify([...state.skipped.search]))}` : "";
    const result = await api("GET", `/api/search?q=${enc(q)}&mode=${mode}&limit=50${tuned}${skip}`);
    if (seq !== state.searchSeq) return;
    const n = result.hits.length;
    const how = result.hits.some((hit) => "meaning_rank" in hit.why) ? " by keywords and meaning" : "";
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

// One refresh at a time: on a long task they can take longer than a pause in typing, and the server
// can't stop one it has started. A request during a refresh is served once that one ends, and any
// number of them become a single refresh of the latest text.
let relatedRunning = null;
let relatedAgain = false;

function refreshRelated() {
  if (relatedRunning) {
    relatedAgain = true;
    return relatedRunning;
  }
  relatedRunning = (async () => {
    try {
      do {
        relatedAgain = false;
        await fetchRelated();
      } while (relatedAgain);
    } finally {
      relatedRunning = null;
    }
  })();
  return relatedRunning;
}

async function fetchRelated() {
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
    skip_folders: [...state.skipped.related],
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

async function likeHit(hit, lane) {
  await api("POST", `/api/tasks/${enc(state.task)}/feedback`, hit.why.liked
    ? { action: "unlike", id: hit.why.liked }
    : { action: "like", lane, path: hit.path, text: hit.text, start_line: hit.start_line, end_line: hit.end_line });
  await refreshRelated();
}

async function unlikePassage(liked, element) {
  element?.remove();
  await api("POST", `/api/tasks/${enc(state.task)}/feedback`, { action: "unlike", id: liked.id });
  await refreshRelated();
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
  const added = Math.max(counted.reduce((sum, [, [adds]]) => sum + adds, 0) - prefs.liked.length, 0); // a ↑ also counts for its lane
  const summary = [
    prefs.liked.length && `${prefs.liked.length} more like this`,
    prefs.dismissed.length - avoided && `${prefs.dismissed.length - avoided} dismissed`,
    avoided && `${avoided} avoided`,
    prefs.steer && "steered",
    added && `${added} added`,
    ignored.length && `${ignored.length} ignored`,
  ].filter(Boolean);
  const empty = !prefs.liked.length && !prefs.dismissed.length && !prefs.steer && !ignored.length && !counted.length;

  const passageItem = (d, undo, tag) => {
    const known = d.start_line !== undefined && d.start_line !== null;
    const location = known ? { line: d.start_line, start: d.start_line, end: d.end_line ?? d.start_line } : null;
    const text = d.text.replace(/\s+/g, " ").trim();
    const where = known ? `${d.path}:${d.start_line + 1}` : d.path;
    return h("li", { class: "feedback-item" },
      h("div", { class: "feedback-head" },
        h("button", { class: "feedback-open", title: `Open ${where} in the reader`, onclick: () => openFile(d.path, location).catch(fail) }, where),
        undo,
      ),
      h("div", { class: "feedback-text" }, tag, text.length > 160 ? `${text.slice(0, 160).trimEnd()}…` : text),
    );
  };
  const dismissed = [...prefs.dismissed].reverse().map((d) => passageItem(d,
    h("button", { title: "Let it be suggested again, and take back its dismissal", onclick: (e) => restoreDismissed(d, e.target.closest("li")).catch(fail) }, "Restore"),
    d.avoid && h("span", { class: "feedback-tag", title: "Avoided: passages like this one are hidden too" }, "⊘ avoided")));
  const liked = [...prefs.liked].reverse().map((d) => passageItem(d,
    h("button", { title: "Stop pulling the suggestions toward this passage", onclick: (e) => unlikePassage(d, e.target.closest("li")).catch(fail) }, "Stop")));
  const ignoredChip = (term) =>
    h("span", { class: "chip ignored" },
      h("span", { class: "chip-label" }, labels.get(term) ?? term),
      h("button", { class: "chip-x", title: "Search for it again", onclick: () => termAction(term, "unignore").catch(fail) }, "×"),
    );
  const laneCounts = counted.map(([lane, [adds, dismisses]]) =>
    `${laneTitles.get(lane) ?? lane}: ${[adds && `${adds} ${prefs.liked.length ? "added or ↑" : "added"}`, dismisses && `${dismisses} dismissed`].filter(Boolean).join(", ")}`);

  drawer.replaceChildren(
    h("summary", {}, "Feedback", h("span", { class: "tune-count" }, summary.length ? ` · ${summary.join(" · ")}` : "")),
    h("div", { class: "tune-body" },
      empty && h("p", { class: "feedback-note" },
        "Nothing yet. The passages you add (+), want more like (↑), dismiss (×) or avoid (⊘) here, the steer line, and the keywords you ignore are remembered for this task, whatever its text."),
      liked.length > 0 && [
        h("div", { class: "tune-group" }, "More like this"),
        h("p", { class: "feedback-note" }, "These pull the suggestions toward them, in meaning and keywords, without being part of the task. Parameters › More like this sets how strongly."),
        h("ul", { class: "feedback-list" }, liked),
      ],
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
  const toDefault = h("button", {
    class: "tune-default",
    type: "button",
    title: `Back to the default, ${formatValue(spec, spec.default)}`,
    onclick: () => setTuned(kind, spec.name, spec.default),
  }, "default");
  input.id = `tune-${kind}-${spec.name}`;
  return h("div", { class: "tune-row", "data-name": spec.name, title: `${spec.help}\nDefault: ${formatValue(spec, spec.default)}` },
    h("label", { class: "tune-label", for: input.id }, spec.label), input, spec.type === "choice" ? null : output, toDefault);
}

// Move a slider as if by hand.
function setTuned(kind, name, value) {
  const input = $(`#${kind}-tune [data-name="${name}"] input, #${kind}-tune [data-name="${name}"] select`);
  input.value = value;
  input.dispatchEvent(new Event("input"));
}

// The Meaning button is Meaning weight at 0 or not; turning it back on restores the last value.
function meaningWeight() {
  const spec = state.paramSpecs.search.find((s) => s.name === "meaning_weight");
  return state.params.search.meaning_weight ?? spec?.default ?? 1;
}

function toggleMeaning() {
  const now = meaningWeight();
  if (now > 0) state.meaningWeightBefore = now;
  const spec = state.paramSpecs.search.find((s) => s.name === "meaning_weight");
  setTuned("search", "meaning_weight", now > 0 ? 0 : state.meaningWeightBefore ?? spec.default);
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
  if (kind === "search") $("#search-meaning").setAttribute("aria-pressed", String(meaningWeight() > 0));
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

// When several folders are served, each panel has its own boxes to leave some of them out. They
// are remembered in this browser, like the parameters.
function loadSkipped(panel) {
  try {
    return new Set(JSON.parse(store.get(`skip-${panel}`) || "[]"));
  } catch {
    return new Set();
  }
}

function renderFolderPicks(folders) {
  for (const panel of ["search", "related"]) {
    const skipped = state.skipped[panel];
    for (const name of [...skipped]) if (!folders.includes(name)) skipped.delete(name);
    const box = $(`#${panel}-folders`);
    box.hidden = folders.length < 2;
    box.replaceChildren(...folders.map((name) =>
      h("label", { class: "toggle", title: `Include ${name} in ${panel === "search" ? "Search" : "Related"}` },
        h("input", {
          type: "checkbox",
          checked: !skipped.has(name),
          onchange: (e) => {
            if (e.target.checked) skipped.delete(name);
            else skipped.add(name);
            store.set(`skip-${panel}`, JSON.stringify([...skipped]));
            if (panel === "search") runSearch();
            else requestRelated(0);
          },
        }),
        name,
      )));
  }
}

function formatBytes(bytes) {
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  return bytes < 1e9 ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / 1e9).toFixed(2)} GB`;
}

async function pollStatus() {
  let ready = true;
  try {
    const s = await api("GET", "/api/status");
    ready = s.state === "ready";
    const model = s.model ? s.model.split("/").pop() : "keywords only";
    const keywordsOnly = s.model && s.unembedded ? ` (${s.unembedded.toLocaleString()} keywords only)` : "";
    let text = `${s.passages.toLocaleString()} passages${keywordsOnly} · ${model}`;
    const documents = `${s.files.toLocaleString()} document${s.files === 1 ? "" : "s"}`;
    $("#index-size").textContent = s.index_bytes ? `${documents} · index ${formatBytes(s.index_bytes)}` : documents;
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
  $("#search-meaning").addEventListener("click", toggleMeaning);
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
  const status = await api("GET", "/api/status");
  renderFolderPicks(status.folders);
  if (!status.model) {
    const button = $("#search-meaning");
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    button.title = "Running without a model: keywords only.";
  }
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
