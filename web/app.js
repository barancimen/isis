import {
  EditorState, EditorSelection, StateField, StateEffect, Prec,
  EditorView, Decoration, WidgetType, keymap, placeholder, drawSelection, highlightActiveLine, lineNumbers,
  history, defaultKeymap, historyKeymap, indentWithTab,
  syntaxHighlighting, HighlightStyle, markdown, tags,
} from "./vendor/codemirror.js";
import { ancestors, fileTree, visibleRows } from "./filetree.js";
import { planInsertion } from "./insert.js";
import { enterInList, envelopeAt, listStructure, paragraphAt, softBreaks, structure } from "./reflow.js";

// ---------- helpers ----------

const $ = (sel) => document.querySelector(sel);
const enc = encodeURIComponent;

const absent = (v) => v === null || v === undefined || v === false;

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (!absent(v)) el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...children.flat().filter((c) => !absent(c)));
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
  // A saved JSON value; `fallback` when there is none or it can't be read.
  json(key, fallback) { try { return JSON.parse(this.get(key)) ?? fallback; } catch { return fallback; } },
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
  { tag: tags.monospace, fontSize: "0.9em" },
  { tag: tags.quote, color: "var(--muted)" },
  { tag: [tags.processingInstruction, tags.meta, tags.contentSeparator], color: "var(--faint)" },
]);

const theme = EditorView.theme({
  "&": { height: "100%", fontSize: "16px", backgroundColor: "var(--bg)", color: "var(--fg)" },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { fontFamily: "var(--font)", lineHeight: "1.65", overflowY: "scroll" },
  ".cm-content": { padding: "18px 0 40vh", maxWidth: "44em", caretColor: "var(--fg)" },
  ".cm-line": { padding: "0 22px" },
  ".cm-gutters": { backgroundColor: "var(--bg)", color: "var(--faint)", border: "none", fontSize: "12px" },
  ".cm-cursor": { borderLeftColor: "var(--fg)", borderLeftWidth: "2px" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "var(--selection) !important" },
  ".cm-activeLine": { backgroundColor: "var(--active-line)" },
  ".cm-placeholder": { color: "var(--faint)", fontStyle: "italic" },
  ".cm-hit-range": { backgroundColor: "var(--hit-line)" },
  ".cm-env": { backgroundColor: "var(--panel)", boxShadow: "inset 3px 0 0 var(--border)" },
  ".cm-env-meta": { fontSize: "0.8em", lineHeight: "1.55", color: "var(--muted)" },
  ".cm-env-key": { fontWeight: "600" },
  ".cm-env-edge": { fontSize: "6px", lineHeight: "16px", userSelect: "none" },
  ".cm-env-edge, .cm-env-edge span, .cm-rule, .cm-rule span": { color: "transparent !important" },
  ".cm-rule": {
    backgroundImage: "linear-gradient(var(--border), var(--border))",
    backgroundSize: "100% 1px",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
  },
  // Each level is well indented, and wrapped lines align under the text, not under the bullet.
  ".cm-line.cm-li": { paddingLeft: "calc(22px + (var(--li-depth) + 1) * 1.7em)", textIndent: "-1.4em" },
  ".cm-line.cm-li-raw": { textIndent: "0" },
  ".cm-li-marker, .cm-li-marker *": { color: "var(--muted) !important" },
  ".cm-li-marker": { display: "inline-block", minWidth: "1.4em", textIndent: "0" },
  ".cm-li-bullet": { width: "1em", minWidth: "1em", marginRight: "0.4em", textAlign: "center" }, // a bullet and its - share a centre
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
// away in an update, the scrollbar is merely back to lagging. It holds up the page (about 0.4 s
// per 80 KB on a slow CPU), so a very long document waits until its scrollbar is pressed.
const MEASURE_AT_ONCE_CHARS = 500_000;
const measuredWidths = new WeakMap();
const measureOnDrag = new WeakSet();
const remeasureTimers = new WeakMap();

function layOutWholeDocument(view) {
  measureOnDrag.delete(view);
  view.viewState.printing = true;
  view.measure();
  view.viewState.printing = false;
  view.requestMeasure();
}

function measureWholeDocument(view) {
  measuredWidths.set(view, view.contentDOM.clientWidth);
  if (!view.dom.offsetHeight) return;
  if (view.state.doc.length > MEASURE_AT_ONCE_CHARS) measureOnDrag.add(view);
  else layOutWholeDocument(view);
}

const remeasureOnResize = EditorView.updateListener.of((update) => {
  const view = update.view;
  if (!update.geometryChanged || measuredWidths.get(view) === view.contentDOM.clientWidth) return;
  clearTimeout(remeasureTimers.get(view)); // one timer per editor: a window resize changes both
  remeasureTimers.set(view, setTimeout(() => measureWholeDocument(view), 200));
});

function measureWhenScrollbarPressed(view) {
  view.scrollDOM.addEventListener("pointerdown", (event) => {
    const onScrollbar = event.target === view.scrollDOM && event.offsetX >= view.scrollDOM.clientWidth;
    if (onScrollbar && measureOnDrag.has(view)) layOutWholeDocument(view);
  });
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

const BULLETS = ["•", "◦", "▪"];
const hidden = Decoration.replace({});
const markerClass = (bullet) => (bullet ? "cm-li-marker cm-li-bullet" : "cm-li-marker");

// What a list item's marker shows as: a bullet that changes with the level, or its number.
class ListMarker extends WidgetType {
  constructor(text, bullet) {
    super();
    this.text = text;
    this.bullet = bullet;
  }
  eq(other) { return other.text === this.text && other.bullet === this.bullet; }
  toDOM() {
    const span = document.createElement("span");
    span.className = markerClass(this.bullet);
    span.textContent = this.text;
    return span;
  }
}

// Enter in a list: the next item, straight below (see reflow.js). A plain Enter would copy a blank
// line from a list that has them, and CodeMirror's has no option against that.
function continueList(view) {
  const { state } = view;
  const { main } = state.selection;
  if (!main.empty) return false;
  const line = state.doc.lineAt(main.head);
  const lines = state.doc.toJSON();
  const plan = enterInList(lines, line.number - 1, main.head - line.from);
  if (!plan) return false;
  const insert = `\n${plan.prefix}`;
  view.dispatch({
    changes: [
      { from: main.head, insert },
      ...plan.renumber.map((r) => {
        const at = state.doc.line(r.line + 1).from;
        return { from: at + r.from, to: at + r.to, insert: r.text };
      }),
    ],
    selection: EditorSelection.cursor(main.head + insert.length),
    scrollIntoView: true,
    userEvent: "input",
  });
  return true;
}

const setFocused = StateEffect.define();

function reflow({ editable }) {
  const build = (state, focused) => {
    const lines = state.doc.toJSON();
    const cursorLine = editable && focused ? state.doc.lineAt(state.selection.main.head).number - 1 : -1;
    const [lo, hi] = cursorLine >= 0 ? paragraphAt(lines, cursorLine) : [-1, -1];
    const editing = (i) => i >= lo && i <= hi;
    const decos = [];
    const lists = listStructure(lines);
    const inList = new Set([...lists.items, ...lists.continuations].map((entry) => entry.line));
    for (const i of softBreaks(lines)) {
      if (i >= lo && i < hi && !inList.has(i)) continue; // Enter there makes a new item, so a list stays as it is
      const end = state.doc.line(i + 1).to;
      // The break, and the indent of a list item's next line.
      decos.push(softBreak.range(end, end + 1 + /^\s*/.exec(lines[i + 1])[0].length));
    }
    for (const { line: i, depth, ordered, marker, markerFrom, markerTo, textFrom } of lists.items) {
      const line = state.doc.line(i + 1);
      decos.push(Decoration.line({ class: "cm-li", attributes: { style: `--li-depth: ${depth}` } }).range(line.from));
      if (i === cursorLine) {
        // The real marker, in the same slot as the bullet it stands for: nothing else moves.
        if (markerFrom) decos.push(hidden.range(line.from, line.from + markerFrom));
        decos.push(Decoration.mark({ class: markerClass(!ordered) }).range(line.from + markerFrom, line.from + markerTo));
        decos.push(hidden.range(line.from + markerTo, line.from + textFrom));
      } else {
        const shown = ordered ? marker : BULLETS[depth % BULLETS.length];
        decos.push(Decoration.replace({ widget: new ListMarker(shown, !ordered) }).range(line.from, line.from + textFrom));
      }
    }
    for (const { line: i, depth, indent, joined } of lists.continuations) {
      if (joined) continue; // it flows on from the line above
      const line = state.doc.line(i + 1);
      decos.push(Decoration.line({ class: "cm-li cm-li-raw", attributes: { style: `--li-depth: ${depth}` } }).range(line.from));
      if (indent) decos.push(hidden.range(line.from, line.from + indent));
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
      keymap.of([{
        key: "Mod-Enter",
        run: () => {
          addReaderSelection();
          return true;
        },
      }]),
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
    Prec.highest(keymap.of([{ key: "Enter", run: continueList }])), // ahead of the markdown mode's Enter
    keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
    remeasureOnResize,
    EditorView.updateListener.of(onTaskUpdate),
  ];
}

const taskState = (doc) => EditorState.create({ doc, extensions: taskExtensions() });
const noTaskState = () => EditorState.create({
  doc: "",
  extensions: [theme, EditorState.readOnly.of(true), placeholder("Create a task with + to start collecting.")],
});

const reader = new EditorView({ parent: $("#reader-editor"), state: readerState("") });
const taskView = new EditorView({ parent: $("#task-editor"), state: noTaskState() });

measureWhenScrollbarPressed(reader);
measureWhenScrollbarPressed(taskView);

// ---------- state ----------

const state = {
  task: null, // active task name
  tasks: [], // [{name, mtime}]
  docs: new Map(), // task name -> {state, mtime, dirty, saving, again, conflictMtime}
  readerPath: null,
  openedHit: null, // the result last opened in the reader, which stays marked
  files: [],
  openFolders: new Set(), // folders open in the Files panel; all are closed at first
  skipped: { search: loadSkipped("search"), related: loadSkipped("related") }, // corpus folders left out, per panel
  panel: "search",
  relatedStale: false, // something changed since the suggestions on show were made
  relatedFor: null, // the task they are for
  relatedTerms: [],
  relatedLanes: [],
  likedIds: new Map(), // "path:line" -> id, for the ↑ buttons
  searchSeq: 0,
  lastRelatedKeys: null,
  indexUpdated: null,
  paramSpecs: { search: [], related: [] }, // from /api/params
  params: loadParams(), // this browser's overrides: {search: {name: value}, related: {...}}
  meaningWeightBefore: null, // what the Meaning button restores
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
    doc = { state: taskState(task.content), mtime: task.mtime, dirty: false };
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
  state.relatedStale = false;
  updateRefreshButton();
  if (state.panel === "related") refreshRelated().catch(fail); // what is on show is another task's
}

function showNoTask() {
  state.task = null;
  state.relatedFor = null;
  state.relatedStale = false;
  updateRefreshButton();
  $("#task-name").textContent = "No task";
  setSaveState("");
  taskView.setState(noTaskState());
  renderTaskTabs();
  renderRelatedEmpty(NO_TASK_RELATED);
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
  if (state.relatedFor === name) state.relatedFor = task.name;
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
        if (name === state.task) setSaveState("Save failed");
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
    doc.state = taskState(task.content);
    doc.mtime = task.mtime;
    doc.dirty = false;
    doc.conflictMtime = null;
    taskView.setState(doc.state);
    measureWholeDocument(taskView);
    setSaveState("Saved");
  }
  $("#conflict").hidden = true;
  requestRelated();
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
  requestRelated();
}

// ---------- adding to the task ----------

let addedTimer;

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
  clearTimeout(addedTimer);
  addedTimer = setTimeout(() => taskView.dispatch({ effects: addedSlot.set.of(Decoration.none) }), 1600);
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
  const decos = [];
  if (location.start !== undefined) {
    // The passage itself, exactly: it may start or end inside a paragraph, which flows as one line.
    const start = doc.line(Math.min(location.start + 1, doc.lines)).from;
    const end = doc.line(Math.min(location.end + 1, doc.lines)).to;
    if (end > start) decos.push(Decoration.mark({ class: "cm-hit-range" }).range(start, end));
  }
  for (const [n, s, e] of location.marks ?? []) {
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

const withoutExtension = (name) => name.replace(/\.(md|markdown)$/i, "");

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
        h("li", {},
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
          withoutExtension(row.name)),
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

const passageKey = (passage) => `${passage.path}:${passage.start_line}`;
const LIKE_TITLE = "More like this: pull the suggestions toward this passage, without adding it";

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
    if (!para) {
      para = h("p");
      body.append(para);
    }
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
  const file = withoutExtension(hit.path.split("/").pop());
  const key = passageKey(hit);
  const open = (event) => {
    if (event.target.closest("button")) return;
    state.openedHit = key;
    for (const card of document.querySelectorAll(".hit")) card.classList.toggle("opened", card.dataset.key === key);
    openFile(hit.path, {
      line: hit.focus_line,
      ch: hit.match?.[1] ?? 0,
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
      title: LIKE_TITLE,
      onclick: () => likeHit(hit, lane).catch(fail),
    }, "↑"),
    lane && h("button", { title: "Not useful: hide it", onclick: (e) => dismiss(hit, lane, e.target.closest(".hit")).catch(fail) }, "×"),
    lane && h("button", { class: "avoid", title: "Avoid: hide it and every passage like it", onclick: (e) => dismiss(hit, lane, e.target.closest(".hit"), true).catch(fail) }, "⊘"),
  );
  const why = lane ? describeWhy(hit.why) : describeSignals(hit.why);
  return h("article", { class: `hit${isNew ? " is-new" : ""}${key === state.openedHit ? " opened" : ""}`, "data-key": key, onclick: open, title: lane ? JSON.stringify(hit.why, null, 1) : null },
    h("div", { class: "hit-head" }, h("span", { class: "hit-title" }, hit.heading || file), actions),
    h("div", { class: "hit-loc" }, `${hit.path}:${hit.focus_line + 1}`),
    renderSnippet(hit),
    why && h("div", { class: "hit-why" }, why),
  );
}

// ---------- search panel ----------

const SEARCH_LIMIT = 50;

const runSearch = debounce(async () => {
  const q = $("#search-input").value;
  const seq = ++state.searchSeq;
  if (!q.trim()) {
    $("#search-status").textContent = "";
    $("#search-results").replaceChildren();
    return;
  }
  try {
    const tuned = Object.keys(state.params.search).length ? `&params=${enc(JSON.stringify(state.params.search))}` : "";
    const skip = state.skipped.search.size ? `&skip=${enc(JSON.stringify([...state.skipped.search]))}` : "";
    // Always hybrid: Meaning off is meaning_weight 0, which the server ranks by keywords alone.
    const result = await api("GET", `/api/search?q=${enc(q)}&mode=hybrid&limit=${SEARCH_LIMIT}${tuned}${skip}`);
    if (seq !== state.searchSeq) return;
    const n = result.hits.length;
    const how = result.hits.some((hit) => "meaning_rank" in hit.why) ? " by keywords and meaning" : "";
    $("#search-status").textContent = n ? `${n}${n === SEARCH_LIMIT ? "+" : ""} passages${how} · ${result.ms} ms` : "No matches.";
    $("#search-results").replaceChildren(...result.hits.map((hit) => renderHit(hit)));
  } catch (error) {
    if (seq !== state.searchSeq) return;
    if (error.status !== 400) fail(error);
    $("#search-status").textContent = error.status === 400 ? `⚠ ${error.message}` : "";
    $("#search-results").replaceChildren();
  }
}, 200);

// ---------- related panel ----------

const NO_TASK_RELATED = "Create a task to see related passages.";

// One refresh at a time: on a long task they can take longer than a pause in typing, and the server
// can't stop one it has started. A request during a refresh is served once that one ends, and any
// number of them become a single refresh of the latest text.
let relatedRunning = null;
let relatedAgain = false;

// The suggestions change only when asked to: something may be read, and is not rewritten under the
// reader. A change only marks them out of date and arms the Refresh button. They load by themselves
// only when what would be on show is nothing, or another task's.
function requestRelated() {
  state.relatedStale = true;
  updateRefreshButton();
}

function updateRefreshButton() {
  $("#related-refresh").disabled = !state.relatedStale || Boolean(relatedRunning);
}

function renderRelatedEmpty(message) {
  $("#related-status").textContent = message;
  $("#related-terms").replaceChildren();
  $("#related-lanes").replaceChildren();
  $("#related-feedback").hidden = true;
  $("#related-steer").hidden = true;
}

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
    } catch (error) {
      state.relatedStale = true;
      $("#related-status").textContent = `⚠ ${error.message}`;
      throw error;
    } finally {
      relatedRunning = null;
      updateRefreshButton();
    }
  })();
  updateRefreshButton();
  return relatedRunning;
}

async function fetchRelated() {
  if (!state.task) {
    renderRelatedEmpty(NO_TASK_RELATED);
    return;
  }
  const name = state.task;
  state.relatedStale = false;
  updateRefreshButton();
  $("#related-status").textContent = "Updating…";
  const result = await api("POST", `/api/tasks/${enc(name)}/related`, {
    text: taskView.state.doc.toString(),
    params: state.params.related,
    skip_folders: [...state.skipped.related],
  });
  if (name !== state.task) return;

  state.relatedFor = name;
  state.relatedTerms = result.terms;
  state.relatedLanes = result.lanes;
  rememberLikes(result.prefs);
  const previous = state.lastRelatedKeys;
  const keys = new Set();
  const lanes = result.lanes.map((lane) =>
    h("section", { class: "lane" },
      h("h3", {}, lane.title, h("span", { class: "count" }, String(lane.items.length))),
      h("p", { class: "lane-desc" }, lane.description),
      lane.note && h("p", { class: "lane-note" }, lane.note),
      lane.items.map((hit) => {
        const key = passageKey(hit);
        keys.add(key);
        return renderHit(hit, { lane: lane.name, isNew: previous !== null && !previous.has(key) });
      }),
    ),
  );
  state.lastRelatedKeys = keys;
  renderChips(state.relatedTerms);
  $("#related-lanes").replaceChildren(...lanes);
  showPrefs(result.prefs);
  const s = result.stats;
  const themes = result.themes.map((t) => t.label).join(" · ");
  $("#related-status").textContent = s.gated === undefined
    ? (result.lanes[0]?.note || "")
    : `${s.gated} clearly related of ${s.passages.toLocaleString()} passages${themes ? ` · themes: ${themes}` : ""}`;
}

// The keywords taken from the task. Boosting writes the word into the steer line; ignoring takes
// a keyword out of the query without hiding the passages that contain it.
function renderChips(terms) {
  const chip = (t) => {
    const [title, term, action] = t.ignored
      ? ["Ignored: click to search for it again", t.term, "unignore"]
      : t.boosted
        ? ["Boosted by the steer line: click to stop boosting", t.term, "unboost"]
        : ["Click to boost: always search for it, at full weight (adds it to the steer line)", t.label, "boost"];
    return h("span", { class: `chip${t.boosted ? " boosted" : ""}${t.ignored ? " ignored" : ""}`, style: `--w: ${t.weight.toFixed(2)}` },
      h("button", { class: "chip-label", title, onclick: () => termAction(t, term, action).catch(fail) }, t.label),
      t.ignored
        ? h("button", { class: "chip-x", title: `Exclude: never show passages containing “${t.label}” (adds -${t.label} to the steer line)`, onclick: () => termAction(t, t.label, "exclude").catch(fail) }, "⊘")
        : h("button", { class: "chip-x", title: "Ignore: stop searching for this word (passages containing it can still be suggested)", onclick: () => termAction(t, t.term, "ignore").catch(fail) }, "×"),
    );
  };
  $("#related-terms").replaceChildren(...terms.map(chip));
}

async function termAction(chip, term, action) {
  if (!(await changePrefs("POST", "/terms", { term, action }))) return;
  if (!chip) return; // from the drawer: the word may not be among the task's keywords any more
  chip.boosted = action === "boost";
  chip.ignored = action === "ignore";
  if (action === "exclude") state.relatedTerms = state.relatedTerms.filter((t) => t !== chip);
  renderChips(state.relatedTerms);
}

// Change what the task remembers and show the result, unless the user has moved to another task
// meanwhile: its panel must not get this task's steer line and feedback. Null in that case.
async function changePrefs(method, path, body) {
  const name = state.task;
  const prefs = await api(method, `/api/tasks/${enc(name)}${path}`, body);
  if (name !== state.task) return null;
  showPrefs(prefs);
  requestRelated();
  return prefs;
}

// What the panel remembers changed: show it now, whether or not the suggestions are refreshed.
function showPrefs(prefs) {
  renderSteer(prefs.steer);
  renderFeedback(prefs, state.relatedTerms, state.relatedLanes);
  rememberLikes(prefs);
  for (const card of document.querySelectorAll("#related-lanes .hit")) {
    const on = state.likedIds.has(card.dataset.key);
    const button = card.querySelector("button.like");
    button.setAttribute("aria-pressed", String(on));
    button.title = on ? "More like this is on: click to stop" : LIKE_TITLE;
  }
}

function rememberLikes(prefs) {
  state.likedIds = new Map(prefs.liked.map((d) => [passageKey(d), d.id]));
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
    const prefs = await changePrefs("PUT", "/steer", { text: input.value });
    if (!prefs) return;
    input.value = prefs.steer;
    $("#steer-error").hidden = true;
  } catch (error) {
    if (error.status !== 400) throw error;
    $("#steer-error").textContent = `⚠ ${error.message}`;
    $("#steer-error").hidden = false;
  }
}

const passageFields = (hit, lane) => ({ lane, path: hit.path, text: hit.text, start_line: hit.start_line, end_line: hit.end_line });

async function dismiss(hit, lane, element, avoid = false) {
  element?.remove();
  await changePrefs("POST", "/feedback", { action: avoid ? "avoid" : "dismiss", ...passageFields(hit, lane) });
}

async function likeHit(hit, lane) {
  const id = state.likedIds.get(passageKey(hit));
  await changePrefs("POST", "/feedback", id ? { action: "unlike", id } : { action: "like", ...passageFields(hit, lane) });
}

// From the drawer: "restore" a dismissed passage, or "unlike" a liked one.
async function undoFeedback(action, id, element) {
  element?.remove();
  await changePrefs("POST", "/feedback", { action, id });
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
    h("button", { title: "Let it be suggested again, and take back its dismissal", onclick: (e) => undoFeedback("restore", d.id, e.target.closest("li")).catch(fail) }, "Restore"),
    d.avoid && h("span", { class: "feedback-tag", title: "Avoided: passages like this one are hidden too" }, "⊘ avoided")));
  const liked = [...prefs.liked].reverse().map((d) => passageItem(d,
    h("button", { title: "Stop pulling the suggestions toward this passage", onclick: (e) => undoFeedback("unlike", d.id, e.target.closest("li")).catch(fail) }, "Stop")));
  const ignoredChip = (term) =>
    h("span", { class: "chip ignored" },
      h("span", { class: "chip-label" }, labels.get(term) ?? term),
      h("button", { class: "chip-x", title: "Search for it again", onclick: () => termAction(state.relatedTerms.find((t) => t.term === term), term, "unignore").catch(fail) }, "×"),
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

async function forgetFeedback() {
  const name = state.task;
  if (!confirm(`Forget all feedback for “${name}”? Dismissed and avoided passages can come back, the steer line and ignored keywords are cleared, and the lanes start even.`)) return;
  const prefs = await changePrefs("DELETE", "/prefs");
  flash(`Forgot the feedback for “${name}”`);
  if (!prefs) return;
  state.relatedTerms = state.relatedTerms.filter((t) => !t.ignored);
  for (const t of state.relatedTerms) t.boosted = false;
  renderChips(state.relatedTerms);
}

// ---------- tunable parameters ----------

// Each panel has a drawer of sliders built from /api/params. Overrides live in this browser and
// travel with every request, so the server's defaults never change.

function loadParams() {
  const saved = store.json("params", {});
  return { search: saved.search || {}, related: saved.related || {} };
}

function saveParams() {
  store.set("params", JSON.stringify(state.params));
}

// A setting changed: a search runs again at once, and Related offers its Refresh.
function settingChanged(kind) {
  if (kind === "search") runSearch();
  else requestRelated();
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
    settingChanged(kind);
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
const meaningSpec = () => state.paramSpecs.search.find((s) => s.name === "meaning_weight");

function meaningWeight() {
  return state.params.search.meaning_weight ?? meaningSpec()?.default ?? 1;
}

function toggleMeaning() {
  const now = meaningWeight();
  if (now > 0) state.meaningWeightBefore = now;
  setTuned("search", "meaning_weight", now > 0 ? 0 : state.meaningWeightBefore ?? meaningSpec().default);
}

function buildTuner(kind) {
  const rows = [];
  let group = null;
  for (const spec of state.paramSpecs[kind]) {
    if (spec.group !== group) {
      group = spec.group;
      rows.push(h("div", { class: "tune-group" }, group));
    }
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
  settingChanged(kind);
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
  if (name === "related" && state.task && state.relatedFor !== state.task) refreshRelated().catch(fail);
  if (name === "search") $("#search-input").focus();
}

// ---------- corpus folders ----------

// When several folders are served, each panel has its own boxes to leave some of them out. They
// are remembered in this browser, like the parameters.
function loadSkipped(panel) {
  const saved = store.json(`skip-${panel}`, []);
  return new Set(Array.isArray(saved) ? saved : []);
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
            settingChanged(panel);
          },
        }),
        name,
      )));
  }
}

// ---------- index status ----------

function formatBytes(bytes) {
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  return bytes < 1e9 ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / 1e9).toFixed(2)} GB`;
}

function statusText(s) {
  if (s.state === "embedding") return `Embedding ${s.embedded.toLocaleString()} / ${s.to_embed.toLocaleString()} passages… (search works meanwhile)`;
  if (s.state === "error") return `Indexing failed: ${s.error} (see the server log; it retries when files change)`;
  if (s.state !== "ready") return "Indexing…";
  const model = s.model ? s.model.split("/").pop() : "keywords only";
  const keywordsOnly = s.model && s.unembedded ? ` (${s.unembedded.toLocaleString()} keywords only)` : "";
  return `${s.passages.toLocaleString()} passages${keywordsOnly} · ${model}`;
}

async function pollStatus() {
  let settled = false; // nothing to wait for: poll slowly
  try {
    const s = await api("GET", "/api/status");
    settled = s.state === "ready" || s.state === "error";
    const documents = `${s.files.toLocaleString()} document${s.files === 1 ? "" : "s"}`;
    $("#index-size").textContent = s.index_bytes ? `${documents} · index ${formatBytes(s.index_bytes)}` : documents;
    $("#index-status").textContent = statusText(s);
    if (state.indexUpdated !== null && s.updated !== state.indexUpdated) {
      loadFiles().catch(fail);
      if (state.relatedFor) requestRelated();
    }
    state.indexUpdated = s.updated;
  } catch {
    $("#index-status").textContent = "Server not reachable";
  }
  setTimeout(pollStatus, settled ? 10000 : 2000);
}

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
  const layout = store.json("layout", {});
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

// ---------- wiring ----------

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
  $("#related-refresh").addEventListener("click", () => refreshRelated().catch(fail));
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
