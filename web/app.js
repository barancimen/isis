import {
  EditorState, EditorSelection, StateField, StateEffect, Prec,
  EditorView, Decoration, WidgetType, keymap, placeholder, drawSelection, highlightActiveLine, lineNumbers, layer, RectangleMarker,
  history, defaultKeymap, historyKeymap, indentWithTab,
  syntaxHighlighting, HighlightStyle, markdown, tags,
  search, SearchQuery, setSearchQuery, findNext, findPrevious, openSearchPanel, closeSearchPanel,
} from "./vendor/codemirror.js";
import { ancestors, fileTree, visibleRows } from "./filetree.js";
import { cleanHistory, forget, remember } from "./history.js";
import { planInsertion } from "./insert.js";
import { enterInList, envelopeAt, linesNotProse, listStructure, paragraphAt, softBreaks, structure } from "./reflow.js";

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

// Small line icons, drawn the same on every system. They take the text colour.
const ICONS = {
  plus: '<path d="M8 3v10M3 8h10"/>',
  more: '<path d="M8 13.5V2.75M3.25 7.5 8 2.75l4.75 4.75"/>',
  less: '<path d="M8 2.5v10.75M3.25 8.5 8 13.25l4.75-4.75"/>',
  hide: '<path d="M4 4l8 8M12 4l-8 8"/>',
  doc: '<path d="M3.75 1.75h5.5l3 3v9.5h-8.5z"/><path d="M6.25 7.75l3.5 3.5M9.75 7.75l-3.5 3.5"/>',
  menu: '<circle cx="3" cy="8" r="1.5"/><circle cx="8" cy="8" r="1.5"/><circle cx="13" cy="8" r="1.5"/>',
  check: '<path d="M3 8.5l3.25 3.25L13 5"/>',
};

function icon(name) {
  const span = document.createElement("span");
  span.className = `icon icon-${name}`;
  span.setAttribute("aria-hidden", "true");
  span.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14">${ICONS[name]}</svg>`;
  return span;
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

// "13ms" or "12s 552ms".
function formatElapsed(ms) {
  const whole = Math.floor(ms);
  return whole < 1000 ? `${whole}ms` : `${Math.floor(whole / 1000)}s ${whole % 1000}ms`;
}

// Counts the time since now, live, through `show`, until the returned stop() is called. stop()
// gives the time taken and may be called more than once.
function stopwatch(show) {
  const started = performance.now();
  let frame = requestAnimationFrame(function tick() {
    show(formatElapsed(performance.now() - started));
    frame = requestAnimationFrame(tick);
  });
  show(formatElapsed(0));
  let took = null;
  return () => {
    cancelAnimationFrame(frame);
    took ??= formatElapsed(performance.now() - started);
    return took;
  };
}

// While a task's rename is on its way to the server, its requests wait for it and then use the
// name it ends up with. The rename's own requests pass `duringRename`.
let renaming = null; // a promise of { from, to }: the name shown meanwhile and the final one

async function api(method, url, body, { duringRename = false, ...options } = {}) {
  if (renaming && !duringRename && url.startsWith("api/tasks/")) {
    const { from, to } = await renaming;
    const prefix = `api/tasks/${enc(from)}`;
    if (url === prefix || url.startsWith(`${prefix}/`)) url = `api/tasks/${enc(to)}${url.slice(prefix.length)}`;
  }
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
  remove(key) { try { localStorage.removeItem(`isis.${key}`); } catch {} },
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
  flash(error.message.charAt(0).toUpperCase() + error.message.slice(1));
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
  ".cm-content": { padding: "18px 0 50vh", maxWidth: "44em", caretColor: "var(--fg)" },
  ".cm-line": { padding: "0 22px", textAlign: "justify" },
  ".cm-line.cm-not-prose, .cm-line.cm-env-meta": { textAlign: "left" },
  ".cm-gutters": { backgroundColor: "var(--bg)", color: "var(--faint)", border: "none", fontSize: "12px" },
  ".cm-cursor": { borderLeftColor: "var(--fg)", borderLeftWidth: "2px" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "var(--selection) !important" },
  ".cm-activeLine": { backgroundColor: "var(--active-line)" },
  ".cm-placeholder": { color: "var(--faint)", fontStyle: "italic" },
  ".cm-hit-layer": { zIndex: -1 },
  ".cm-hit-fill": { backgroundColor: "var(--hit-line)" },
  ".cm-env": { backgroundColor: "var(--panel)", boxShadow: "inset 3px 0 0 var(--border)" },
  // The selection is drawn under the lines, so this background would hide it. Here the browser's own
  // highlight shows instead, which the editor otherwise turns off.
  ".cm-line.cm-env ::selection, .cm-line.cm-env::selection": { backgroundColor: "var(--selection) !important" },
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
  // Each level is well indented, and wrapped lines align under the text, not under the bullet. The
  // indent is a border and the bullet hangs by its own margin: CodeMirror takes the left edge of a
  // highlight or selection from the padding and text-indent of the first line it draws, so a list
  // line there must have the same ones as any other line.
  ".cm-line.cm-li": { borderLeft: "calc((var(--li-depth) + 1) * 1.7em) solid transparent" },
  ".cm-li-marker, .cm-li-marker *": { color: "var(--muted) !important" },
  ".cm-li-marker": { display: "inline-block", minWidth: "1.4em", marginLeft: "-1.4em" },
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

// The color behind a highlighted passage. A color set on the text itself covers only the height of
// the font, whatever that is on this computer, so lines would have gaps or overlap. Rectangles for
// each line would still be rounded apart at some zoom levels. So each paragraph of the passage is
// one element, cut to the shape of its lines: it has no inner edges to show.
class HitBlock {
  // The editor can give several rectangles for one row, such as either side of a joined line break.
  // A row must be one span, or the outline crosses itself and leaves holes.
  constructor(rects) {
    this.rects = [];
    const sorted = rects.map(({ left, top, width, height }) => ({ left, top, right: left + width, bottom: top + height })).sort((a, b) => a.top - b.top);
    for (const rect of sorted) {
      const row = this.rects[this.rects.length - 1];
      if (row && rect.top < row.bottom - 1) {
        row.left = Math.min(row.left, rect.left);
        row.right = Math.max(row.right, rect.right);
        row.bottom = Math.max(row.bottom, rect.bottom);
      } else {
        this.rects.push(rect);
      }
    }
  }
  eq(other) {
    return JSON.stringify(this.rects) === JSON.stringify(other.rects);
  }
  draw() {
    const element = document.createElement("div");
    element.className = "cm-hit-fill";
    this.place(element);
    return element;
  }
  update(element) {
    this.place(element);
    return true;
  }
  place(element) {
    const left = Math.min(...this.rects.map((r) => r.left)), right = Math.max(...this.rects.map((r) => r.right));
    const top = this.rects[0].top, bottom = Math.max(...this.rects.map((r) => r.bottom));
    const point = (x, y) => `${x - left}px ${y - top}px`;
    const rightEdge = this.rects.flatMap((r) => [point(r.right, r.top), point(r.right, r.bottom)]);
    const leftEdge = [...this.rects].reverse().flatMap((r) => [point(r.left, r.bottom), point(r.left, r.top)]);
    Object.assign(element.style, { left: `${left}px`, top: `${top}px`, width: `${right - left}px`, height: `${bottom - top}px`, clipPath: `polygon(${[...rightEdge, ...leftEdge].join(",")})` });
  }
}

const hitFill = layer({
  above: false,
  class: "cm-hit-layer",
  update: (u) => u.docChanged || u.viewportChanged || u.geometryChanged || u.startState.field(hitSlot.field) !== u.state.field(hitSlot.field),
  markers(view) {
    const { doc } = view.state;
    const found = view.state.field(hitSlot.field).iter();
    while (found.value && found.value.spec.class !== "cm-hit-range") found.next();
    if (!found.value) return [];
    const markers = [];
    const draw = (from, to) => {
      const rects = RectangleMarker.forRange(view, "cm-hit-fill", EditorSelection.range(from, to));
      if (rects.length) markers.push(new HitBlock(rects));
    };
    let start = null, end = null;
    for (let n = doc.lineAt(found.from).number; n <= doc.lineAt(found.to).number; n++) {
      const line = doc.line(n);
      if (line.text.trim()) {
        start ??= Math.max(line.from, found.from);
        end = Math.min(line.to, found.to);
      } else if (start !== null) {
        draw(start, end);
        start = null;
      }
    }
    if (start !== null) draw(start, end);
    return markers;
  },
});

// Hard-wrapped paragraphs flow to the width of the pane: their line breaks show as spaces (see
// reflow.js). In the task editor a break typed with Enter shows as a real line until the editor
// loses focus, so pressing Enter visibly starts a new line while the rest still flows.
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

// Where the breaks typed with Enter are after this change. A pasted text flows like any other.
function typedBreaksAfter(breaks, tr) {
  if (!tr.docChanged) return breaks;
  const kept = breaks.map((at) => tr.changes.mapPos(at, 1)).filter((at) => tr.newDoc.sliceString(at, at + 1) === "\n");
  if (tr.isUserEvent("input") && !tr.isUserEvent("input.paste") && !tr.isUserEvent("input.drop")) {
    tr.changes.iterChanges((_fromA, _toA, from, _to, text) => {
      const at = text.toString().indexOf("\n");
      if (at >= 0) kept.push(from + at);
    });
  }
  return kept;
}

const notProse = Decoration.line({ class: "cm-not-prose" });

function reflow({ editable }) {
  const build = (state, focused, typedBreaks) => {
    const lines = state.doc.toJSON();
    const cursorLine = editable && focused ? state.doc.lineAt(state.selection.main.head).number - 1 : -1;
    const [lo, hi] = cursorLine >= 0 ? paragraphAt(lines, cursorLine) : [-1, -1];
    const editing = (i) => i >= lo && i <= hi;
    const decos = [];
    const lists = listStructure(lines);
    const inList = new Set([...lists.items, ...lists.continuations].map((entry) => entry.line));
    for (const i of softBreaks(lines)) {
      const end = state.doc.line(i + 1).to;
      if (typedBreaks.includes(end) && !inList.has(i)) continue; // Enter there makes a new item, so a list stays as it is
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
    for (const i of linesNotProse(lines, inList)) decos.push(notProse.range(state.doc.line(i + 1).from));
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
    create: (state) => ({ focused: false, typedBreaks: [], decos: build(state, false, []) }),
    update(value, tr) {
      let focused = value.focused;
      for (const e of tr.effects) if (e.is(setFocused)) focused = e.value;
      if (!tr.docChanged && focused === value.focused && !(editable && focused && tr.selection)) return value;
      const typedBreaks = focused ? typedBreaksAfter(value.typedBreaks, tr) : [];
      return { focused, typedBreaks, decos: build(tr.state, focused, typedBreaks) };
    },
    provide: (f) => EditorView.decorations.from(f, (value) => value.decos),
  });
  return editable ? [field, EditorView.focusChangeEffect.of((_, focusing) => setFocused.of(focusing))] : field;
}

// ---------- find in a pane ----------

// Each editor has its own box. The browser's own find can't do this: an editor only keeps the text
// near the screen on the page, so the browser would miss the rest of the document.
const FIND_COUNT_LIMIT = 5000;
const findState = new WeakMap(); // view -> { text, anchor }: what it looks for, and where from

function findQuery(text) {
  // Any white space in the words matches any white space in the text, line breaks too: a hard
  // wrapped paragraph shows as one line, so two words may look adjacent across a break.
  const pattern = text.trim().replace(/[.*+?^${}()|[\]\\/]/g, "\\$&").replace(/\s+/g, "\\s+");
  return new SearchQuery({ search: pattern, regexp: true, caseSensitive: false });
}

function findCount(view, query) {
  if (!query.valid) return "";
  const { from, to } = view.state.selection.main;
  const matches = query.getCursor(view.state);
  let total = 0;
  let current = 0;
  for (let m = matches.next(); !m.done && total < FIND_COUNT_LIMIT; m = matches.next()) {
    total += 1;
    if (m.value.from === from && m.value.to === to) current = total;
  }
  if (!total) return "No matches";
  const shown = total >= FIND_COUNT_LIMIT ? `${total.toLocaleString()}+` : total.toLocaleString();
  return current ? `${current.toLocaleString()} of ${shown}` : `${shown} ${total === 1 ? "match" : "matches"}`;
}

function findPanel(view, label) {
  const remembered = () => findState.get(view) ?? { text: "", anchor: view.state.selection.main.from };
  const count = h("span", { class: "find-count", "aria-live": "polite" });
  const input = h("input", {
    type: "text", spellcheck: "false", "main-field": "true", placeholder: label, "aria-label": label,
    oninput: () => {
      const { anchor } = remembered();
      findState.set(view, { text: input.value, anchor });
      view.dispatch({ selection: EditorSelection.cursor(Math.min(anchor, view.state.doc.length)), effects: setSearchQuery.of(findQuery(input.value)) });
      if (input.value.trim()) {
        findNext(view);
        input.setSelectionRange(input.value.length, input.value.length); // it selects the box's text
      }
    },
    onkeydown: (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        (e.shiftKey ? findPrevious : findNext)(view);
      } else if (e.key === "Escape") {
        e.preventDefault();
        closeFind(view);
      }
    },
  });
  input.value = remembered().text;
  const button = (text, title, onclick) => h("button", { type: "button", title, "aria-label": title, onclick }, text);
  const dom = h("div", { class: "find-panel" }, input, count,
    button("↑", "Previous match (Shift+Enter)", () => findPrevious(view)),
    button("↓", "Next match (Enter)", () => findNext(view)),
    button("×", "Close (Esc)", () => closeFind(view)));
  const show = (state) => {
    if (input.value !== remembered().text) input.value = remembered().text;
    count.textContent = input.value.trim() ? findCount({ state }, findQuery(input.value)) : "";
    count.classList.toggle("none", count.textContent === "No matches");
  };
  return {
    dom,
    top: true,
    mount: () => {
      show(view.state);
      input.focus();
      input.select();
    },
    update: (update) => {
      if (update.docChanged || update.selectionSet || update.transactions.some((tr) => tr.effects.some((e) => e.is(setSearchQuery)))) show(update.state);
    },
  };
}

function closeFind(view) {
  closeSearchPanel(view);
  view.focus();
}

function findInPane(label) {
  return [
    search({
      top: true,
      createPanel: (view) => findPanel(view, label),
      scrollToMatch: (range) => EditorView.scrollIntoView(range, { y: "center" }),
    }),
    keymap.of([{ key: "Escape", run: closeSearchPanel }]),
  ];
}

function openFind(view) {
  const { from, to } = view.state.selection.main;
  const selected = view.state.sliceDoc(from, to);
  const usable = selected.trim() && selected.length <= 200 && !selected.includes("\n");
  const text = usable ? selected : findState.get(view)?.text ?? "";
  findState.set(view, { text, anchor: from });
  openSearchPanel(view); // first, as it sets a query of its own
  view.dispatch({ effects: setSearchQuery.of(findQuery(text)) });
}

// Read-only, whatever tries to change it. Not every command checks EditorState.readOnly (the markdown
// mode's Backspace took a list marker away), so a filter refuses every change, and the browser is not
// let to edit what it shows either. A new document comes with a new state, not as a change.
const readOnly = [
  EditorState.readOnly.of(true),
  EditorState.changeFilter.of(() => false),
  EditorView.domEventHandlers({ beforeinput: (event) => { event.preventDefault(); return true; } }),
];

function readerState(doc) {
  return EditorState.create({
    doc,
    extensions: [
      readOnly,
      lineNumbers(),
      drawSelection(),
      EditorView.lineWrapping,
      markdown({ extensions: envelopedBlocks, addKeymap: false }), // its keys edit list markup
      syntaxHighlighting(markdownStyle),
      theme,
      reflow({ editable: false }),
      findInPane("Find in this document"),
      hitSlot.field,
      hitFill,
      placeholder("Open a document from Files or click a search result."),
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
    // The cursor stays on show in red while you work elsewhere, so you can see where an added passage goes.
    // It blinks only while the task has the focus.
    EditorView.theme({
      "&:not(.cm-focused) .cm-cursorLayer": { animation: "none !important" }, // CodeMirror sets the blink inline
      "&:not(.cm-focused) .cm-cursorLayer .cm-cursor": { display: "block", borderLeftColor: "var(--accent)" },
    }),
    reflow({ editable: true }),
    findInPane("Find in this task"),
    addedSlot.field,
    placeholder(
      "Collect passages here. Select text in a document and press Ctrl/⌘+Enter or click + on a result. " +
        "Each passage goes in at your cursor. Add your own notes around them. Related uses this whole task to suggest more passages."
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
  extensions: [theme, readOnly, placeholder("Click + at the top left to create a task.")],
});

const reader = new EditorView({ parent: $("#reader-editor"), state: readerState("") });
const taskView = new EditorView({ parent: $("#task-editor"), state: noTaskState() });

measureWhenScrollbarPressed(reader);
measureWhenScrollbarPressed(taskView);

// ---------- state ----------

const SHARED_HIDDEN_KEY = "hidden-search";
const MAX_HIDDEN_DOCS = 50; // the list travels in the URL, which a public server limits in length

// The recent searches are kept for this tab, through reloads, and shared by its tasks. A new tab
// starts without any.
const HISTORY_KEY = "isis.search-history";

const state = {
  model: null, // the meaning model this browser uses, by id; null with keywords only
  indexStatus: null, // the last answer of api/status
  task: null, // active task name
  tasks: [], // [{name, mtime}]
  docs: new Map(), // task name -> {state, mtime, dirty, saving, again, conflictMtime}
  hiddenDocs: cleanHiddenDocs(store.json(SHARED_HIDDEN_KEY, [])), // documents left out of the active task's Search
  presets: loadPresets(), // panel -> [{name, settings}]
  activePreset: { search: null, related: null }, // the preset last chosen or saved in each panel
  prefs: null, // what the task remembers, as last shown
  prefsFor: null, // the task that is about
  folders: [],
  readerPath: null,
  readerSeq: 0, // the latest document asked for: a slower answer for an earlier one is dropped
  fileToReveal: false, // Files can't scroll while hidden, so this waits until it shows
  pendingView: null, // the task view on its way to the screen
  viewTurn: 0,
  readerScroll: store.json("reader-scroll", {}), // document path -> where its reader view was left
  openedHit: null, // the result last opened in the reader, which stays marked
  readerLocation: null, // the passage highlighted in the reader
  files: [],
  openFolders: new Set(), // folders open in the Files panel; all are closed at first
  skipped: loadSkipped(), // corpus folders, and sources of osiris archives, left out of Search and Related alike
  sources: [], // the sources of the osiris archives, as /api/status lists them
  panel: "search",
  relatedStale: false, // something changed since the suggestions on show were made
  relatedFor: null, // the task they are for
  relatedTerms: [],
  relatedLanes: [],
  likedIds: new Map(), // "path:line" -> id, of the passages marked More like this
  avoidedKeys: new Set(), // "path:line" of the passages marked Less like this
  hiddenPassages: [], // passages left out of the active task's Search: [{path, start_line, end_line, text}]
  searchSeq: 0,
  searchAsTyped: null, // the query to search for without correcting its spelling
  searchTop: 0, // where the Search panel is scrolled to, for the active task
  searchPending: false, // the task's search results aren't back yet, and then go to searchTop
  searchRunning: null, // the query whose results are on their way
  searchedOk: null, // the query of the results on show, when it searched without an error
  searchToKeep: null, // a settled query that joins the recent searches once it has searched
  searchHistory: loadHistory(), // recent searches in this tab, newest first
  historyAt: -1, // the recent search picked with the arrow keys, or -1
  sideScroll: null, // where the Files or Related panel was before a reload, until its content is back
  taskPlaces: {}, // task name -> { top, cursor }
  indexUpdated: null,
  pageVersion: null, // of the files this page loaded
  paramSpecs: { search: [], related: [] }, // from /api/params
  params: loadParams(), // this browser's overrides: {search: {name: value}, related: {...}}
  meaningWeightBefore: null, // what the Meaning button restores
};

// ---------- tasks ----------

// The tabs are updated in place while the list of names stays the same, so a double-click
// lands on the tab its first click switched to.
function renderTaskTabs() {
  const list = $("#task-tabs");
  const names = state.tasks.map((t) => t.name);
  const tabs = [...list.querySelectorAll(".task-tab")];
  const same = tabs.length === names.length && tabs.every((li, i) => li.dataset.name === names[i]);
  if (!same) {
    list.replaceChildren();
    if (!names.length) list.append(h("li", { class: "tasks-empty" }, "No tasks yet. Click + to create one."));
    for (const name of names) {
      list.append(
        h("li", { class: "task-tab", role: "presentation", "data-name": name },
          h("button", { class: "name", role: "tab", title: name }, name),
          h("button", { class: "task-x", tabindex: "-1", title: "Delete this task", "aria-label": `Delete ${name}`, onclick: () => deleteTask(name).catch(fail) }, "×"),
        ),
      );
    }
  }
  for (const li of list.querySelectorAll(".task-tab")) {
    const active = li.dataset.name === state.task;
    li.classList.toggle("active", active);
    li.querySelector(".name")?.setAttribute("aria-selected", String(active));
  }
}

const tabFor = (name) => [...document.querySelectorAll("#task-tabs .task-tab")].find((li) => li.dataset.name === name);
const focusTab = (name) => tabFor(name)?.querySelector(".name")?.focus();

// Double-click or F2 turns the name into a box. Enter or leaving the box renames and Escape doesn't.
function startRename(name) {
  const li = tabFor(name);
  const button = li?.querySelector(".name");
  if (!button) return;
  const input = h("input", { class: "rename", value: name, spellcheck: "false", "aria-label": `New name for ${name}` });
  let done = false;
  const finish = async (rename) => {
    if (done) return;
    done = true;
    const newName = input.value.trim();
    input.replaceWith(button);
    let focus = name;
    if (rename && newName && newName !== name) {
      try {
        focus = await renameTask(name, newName);
      } catch (e) {
        fail(e);
      }
    }
    focusTab(focus);
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
  button.replaceWith(input);
  input.focus();
  input.select();
}

// Puts a task just in front of `before`, or at the end of the list when that is null.
async function moveTask(name, before) {
  const moving = state.tasks.find((t) => t.name === name);
  if (!moving || before === name) return;
  const rest = state.tasks.filter((t) => t.name !== name);
  const at = rest.findIndex((t) => t.name === before);
  rest.splice(at < 0 ? rest.length : at, 0, moving);
  if (rest.every((t, i) => t === state.tasks[i])) return;
  state.tasks = rest; // at once, so the tab doesn't jump back while the server answers
  renderTaskTabs();
  try {
    state.tasks = await api("POST", `api/tasks/${enc(name)}/move`, { before });
    renderTaskTabs();
  } catch (e) {
    await loadTasks().catch(() => {});
    throw e;
  }
}

function onTabKey(e) {
  const li = e.target.closest(".task-tab");
  if (!li || e.target.tagName !== "BUTTON") return;
  const name = li.dataset.name;
  const names = state.tasks.map((t) => t.name);
  const at = names.indexOf(name);
  if (e.key === "F2") {
    e.preventDefault();
    startRename(name);
  } else if (e.key === "Delete" || e.key === "Backspace") { // a Mac's delete key is Backspace
    e.preventDefault();
    deleteTask(name).then(() => focusTab(tabFor(name) ? name : state.task)).catch(fail); // the same tab if the delete was called off
  } else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && e.altKey) {
    e.preventDefault();
    const before = e.key === "ArrowUp" ? names[at - 1] : names[at + 2] ?? null;
    if (at === 0 && e.key === "ArrowUp") return;
    moveTask(name, before).then(() => focusTab(name)).catch(fail);
  } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
    e.preventDefault();
    const next = names[at + (e.key === "ArrowUp" ? -1 : 1)];
    if (next) focusTab(next);
  }
}

function bindTaskTabs() {
  const list = $("#task-tabs");
  const nameAt = (e) => (e.target.closest?.(".task-tab:not(:has(input))") ?? null)?.dataset.name;
  list.addEventListener("click", (e) => {
    const name = nameAt(e);
    if (name && e.target.closest(".name")) switchTask(name).catch(fail);
  });
  list.addEventListener("dblclick", (e) => {
    if (e.target.closest(".task-x")) return;
    const name = nameAt(e);
    if (name) startRename(name);
  });
  list.addEventListener("keydown", onTabKey);

  // Dragging, done by hand rather than with the browser's drag and drop, whose picture of the
  // moving tab can't be styled. A copy of the tab follows the mouse, and a line shows where it
  // will go. Escape or letting go outside the task list calls it off.
  const line = h("div", { class: "drop-line", hidden: true });
  let drag = null; // { name, li, x, y, dx, dy, ghost, before }
  let justDragged = false;
  const stop = () => {
    drag?.ghost?.remove();
    drag?.li.classList.remove("dragging");
    document.body.classList.remove("dragging-task");
    line.hidden = true;
    drag = null;
  };
  const place = (e) => {
    const listBox = list.getBoundingClientRect();
    const inside = e.target.closest?.("#tasks") || (e.clientX >= listBox.left && e.clientX <= listBox.right && e.clientY >= listBox.top && e.clientY <= listBox.bottom);
    const tabs = [...list.querySelectorAll(".task-tab")];
    if (!inside || !tabs.length) {
      drag.before = undefined;
      line.hidden = true;
      return;
    }
    if (e.clientY < listBox.top + 24) list.scrollTop -= 8;
    else if (e.clientY > listBox.bottom - 24) list.scrollTop += 8;
    const next = tabs.find((li) => { const box = li.getBoundingClientRect(); return e.clientY < box.top + box.height / 2; });
    drag.before = next?.dataset.name ?? null;
    const edge = next ? next.getBoundingClientRect().top : tabs.at(-1).getBoundingClientRect().bottom;
    line.style.top = `${Math.max(0, edge - listBox.top + list.scrollTop - 1)}px`; // centred on the edge, but not cut off above the first tab
    line.hidden = false;
  };
  list.addEventListener("pointerdown", (e) => {
    const button = e.target.closest(".task-tab > .name");
    if (!button || e.button !== 0) return;
    drag = { name: button.parentElement.dataset.name, li: button.parentElement, x: e.clientX, y: e.clientY, ghost: null, before: undefined };
  });
  window.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (!drag.ghost) {
      if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 5) return; // still a click
      const box = drag.li.getBoundingClientRect();
      drag.dx = drag.x - box.left;
      drag.dy = drag.y - box.top;
      drag.ghost = h("div", { class: `task-ghost${drag.li.classList.contains("active") ? " active" : ""}` }, drag.name);
      drag.ghost.style.width = `${box.width}px`;
      drag.ghost.style.height = `${box.height}px`;
      document.body.append(drag.ghost);
      drag.li.classList.add("dragging");
      document.body.classList.add("dragging-task");
      list.append(line);
    }
    drag.ghost.style.transform = `translate(${e.clientX - drag.dx}px, ${e.clientY - drag.dy}px)`;
    place(e);
  });
  window.addEventListener("pointerup", () => {
    if (!drag) return;
    const { name, ghost, before } = drag;
    stop();
    if (!ghost) return;
    justDragged = true; // the click that ends a drag doesn't switch tasks
    setTimeout(() => (justDragged = false), 0);
    if (before !== undefined) moveTask(name, before).catch(fail);
  });
  window.addEventListener("pointercancel", stop);
  window.addEventListener("blur", stop);
  window.addEventListener("keydown", (e) => {
    if (drag?.ghost && e.key === "Escape") {
      e.preventDefault();
      stop();
    }
  }, true);
  list.addEventListener("click", (e) => {
    if (justDragged) e.stopImmediatePropagation();
  }, true);
}

async function loadTasks() {
  state.tasks = await api("GET", "api/tasks");
  renderTaskTabs();
}

// `view`: what to show in the reader and in Search. A first task made while no task is open has
// none, and takes over what is on show.
async function switchTask(name, view = loadTaskView(name)) {
  if (state.task === name) return;
  const leaving = state.task;
  if (state.task && state.docs.has(state.task)) {
    rememberTaskPlace();
    saveTaskView();
    state.docs.get(state.task).state = taskView.state;
    saveTask(state.task).catch(fail); // in the background, so the switch doesn't wait
  }
  let doc = state.docs.get(name);
  const opening = !doc || doc.preloaded;
  if (!doc) {
    const task = await api("GET", `api/tasks/${enc(name)}`);
    doc = { state: taskState(task.content), mtime: task.mtime, dirty: false };
    state.docs.set(name, doc);
  } else if (doc.preloaded) {
    delete doc.preloaded;
    recheckTask(name, doc).catch(() => {});
  }
  state.task = name;
  store.set("task", name);
  taskView.setState(doc.state);
  measureWholeDocument(taskView);
  restoreTaskPlace(name, opening);
  $("#task-name").textContent = name;
  $("#conflict").hidden = true;
  setSaveState(doc.dirty ? "unsaved" : "saved");
  renderTaskTabs();
  state.relatedStale = false;
  updateRefreshButton();
  if (state.panel === "related") refreshRelated().catch(fail); // what is on show is another task's
  if (view || leaving) await showView(view ?? {});
}

// After the page starts, the other tasks' text loads in the background, one at a time, so that
// opening one doesn't wait for the server.
async function preloadTasks() {
  for (const { name } of [...state.tasks]) {
    if (state.docs.has(name)) continue;
    try {
      const task = await api("GET", `api/tasks/${enc(name)}`);
      if (!state.docs.has(name) && state.tasks.some((t) => t.name === name)) {
        state.docs.set(name, { state: taskState(task.content), mtime: task.mtime, dirty: false, preloaded: true });
      }
    } catch {} // it loads when it's opened instead
  }
}

// A preloaded task is shown as it was loaded, then checked in case it changed since, such as in
// another browser. If it did and it hasn't been edited here, the newer text replaces it.
async function recheckTask(name, doc) {
  const task = await api("GET", `api/tasks/${enc(name)}`);
  if (state.docs.get(name) !== doc || doc.dirty || doc.saving || task.mtime === doc.mtime) return;
  doc.state = taskState(task.content);
  doc.mtime = task.mtime;
  if (state.task === name) {
    taskView.setState(doc.state);
    measureWholeDocument(taskView);
    restoreTaskPlace(name, true);
  }
}

// ---------- each task's reader and search ----------

// Each task has its own document in the reader and its own search. They are kept in this browser.
const viewKey = (name) => `task-view.${name}`;

function loadTaskView(name) {
  return store.json(viewKey(name), null);
}

function saveTaskView() {
  if (state.task) store.set(viewKey(state.task), JSON.stringify(currentView()));
}

function currentView() {
  if (state.pendingView) return state.pendingView;
  rememberSearchTop();
  return {
    reader: state.readerPath,
    readerLocation: state.readerLocation,
    readerTop: state.readerPath ? topPlace(reader) : 0,
    openedHit: state.openedHit,
    query: $("#search-input").value,
    asTyped: state.searchAsTyped,
    searchTop: state.searchTop,
    hiddenDocs: [...state.hiddenDocs],
    hiddenPassages: state.hiddenPassages,
  };
}

async function showView(view) {
  const turn = ++state.viewTurn;
  state.pendingView = view;
  state.openedHit = view.openedHit ?? null;
  for (const card of document.querySelectorAll(".hit")) card.classList.toggle("opened", card.dataset.key === state.openedHit);
  showViewSearch(view);
  try {
    if (view.reader) await openFile(view.reader, view.readerLocation ?? null, { place: view.readerTop ?? state.readerScroll[view.reader] ?? 0 });
    else closeReader();
  } catch {
    if (turn === state.viewTurn) closeReader(); // the document may be gone
  }
  if (turn === state.viewTurn) state.pendingView = null;
}

// The results come from the server only once the panel is on show, as in Related.
function showViewSearch(view) {
  runSearch.cancel();
  closeHistory();
  state.searchToKeep = null;
  state.searchSeq++;
  stopSearchClock();
  state.searchAsTyped = view.asTyped ?? null;
  state.searchTop = view.searchTop ?? 0;
  state.hiddenDocs = cleanHiddenDocs(view.hiddenDocs);
  state.hiddenPassages = cleanHiddenPassages(view.hiddenPassages);
  renderHiddenDocs();
  $("#search-input").value = view.query ?? "";
  $("#search-status").textContent = "";
  $("#search-results").replaceChildren();
  renderPresets("search");
  state.searchPending = Boolean($("#search-input").value.trim());
  if (state.panel === "search" && state.searchPending) searchNow();
}

function rememberSearchTop() {
  if (state.panel === "search" && !state.searchPending) state.searchTop = $("#panel-search").scrollTop;
}

function searchArrived() {
  if (!state.searchPending) return;
  state.searchPending = false;
  if (state.panel === "search") $("#panel-search").scrollTop = state.searchTop;
}

// Each task's place, kept through a switch and a reload. An editor keeps its cursor through a
// switch by itself, but after a reload it starts anew.
function rememberTaskPlace() {
  if (state.task) state.taskPlaces[state.task] = { top: topPlace(taskView), cursor: taskView.state.selection.main.head };
}

function restoreTaskPlace(name, opening) {
  const place = state.taskPlaces[name];
  if (!place) return;
  const doc = taskView.state.doc;
  taskView.dispatch({
    selection: opening ? EditorSelection.cursor(Math.min(place.cursor, doc.length)) : undefined,
    effects: scrollToPlace(place.top, doc),
  });
}

function showNoTask() {
  state.task = null;
  state.relatedFor = null;
  state.relatedStale = false;
  updateRefreshButton();
  $("#task-name").textContent = "No task";
  setSaveState("none");
  taskView.setState(noTaskState());
  renderTaskTabs();
  renderRelatedEmpty(NO_TASK_RELATED);
}

async function newTask() {
  const name = prompt("Name of the new task");
  if (!name || !name.trim()) return null;
  const task = await api("POST", "api/tasks", { name });
  // The answer has all the page needs, so the list and the task's text aren't asked for again.
  if (!state.tasks.some((t) => t.name === task.name)) state.tasks.push({ name: task.name, mtime: task.mtime });
  state.docs.set(task.name, { state: taskState(task.content), mtime: task.mtime, dirty: false });
  renderTaskTabs();
  await switchTask(task.name);
  return task.name;
}

// Moves what the page keeps about a task to a new name.
function carryTaskState(from, to) {
  if (from === to) return;
  const doc = state.docs.get(from);
  state.docs.delete(from);
  if (doc) state.docs.set(to, doc);
  if (state.taskPlaces[from]) state.taskPlaces[to] = state.taskPlaces[from];
  delete state.taskPlaces[from];
  const view = store.get(viewKey(from));
  if (view) store.set(viewKey(to), view);
  store.remove(viewKey(from));
  if (state.relatedFor === from) state.relatedFor = to;
  for (const task of state.tasks) if (task.name === from) task.name = to;
  if (state.task === from) {
    state.task = to;
    store.set("task", to);
    $("#task-name").textContent = to;
  }
  renderTaskTabs();
}

// The new name shows at once and the server catches up. Saves of the task wait for it, since the
// task has its old name there until then. If the server cleans the name up, the cleaned name
// replaces it, and if the server refuses, the old name comes back. Returns the name the task has.
async function renameTask(name, newName) {
  while (renaming) { // one rename at a time
    const { from, to } = await renaming;
    if (name === from) name = to;
  }
  newName = newName.trim();
  if (state.tasks.some((t) => t.name === newName)) throw new Error(`a task named '${newName}' already exists`);
  carryTaskState(name, newName);
  let done;
  renaming = new Promise((resolve) => (done = resolve));
  let final = name;
  try {
    await saveTask(newName, { beforeRename: name });
    const task = await api("POST", `api/tasks/${enc(name)}/rename`, { new_name: newName }, { duringRename: true });
    final = task.name;
    const doc = state.docs.get(newName);
    if (doc) doc.mtime = task.mtime;
    carryTaskState(newName, final);
    return final;
  } catch (error) {
    carryTaskState(newName, name);
    throw error;
  } finally {
    renaming = null;
    done({ from: newName, to: final });
  }
}

// The task leaves the list at once and the server catches up. If it refuses, the task comes back.
async function deleteTask(name) {
  if (!confirm(`Delete the task “${name}”?`)) return;
  const kept = { doc: state.docs.get(name), place: state.taskPlaces[name], view: store.get(viewKey(name)) };
  const wasActive = state.task === name;
  state.docs.delete(name);
  delete state.taskPlaces[name];
  store.remove(viewKey(name));
  state.tasks = state.tasks.filter((t) => t.name !== name);
  renderTaskTabs();
  const deleting = api("DELETE", `api/tasks/${enc(name)}`);
  deleting.catch(() => {}); // handled below, after the switch
  if (wasActive) {
    state.task = null;
    if (state.tasks.length) await switchTask(state.tasks[0].name, loadTaskView(state.tasks[0].name) ?? {}).catch(fail);
    else {
      showNoTask();
      await showView({});
    }
  }
  try {
    await deleting;
  } catch (error) {
    if (kept.doc) state.docs.set(name, kept.doc);
    if (kept.place) state.taskPlaces[name] = kept.place;
    if (kept.view) store.set(viewKey(name), kept.view);
    await loadTasks().catch(() => {});
    throw error;
  }
}

// ---------- saving ----------

const SAVE_TEXT = { none: "", saved: "Saved", unsaved: "Not saved", saving: "Saving…", failed: "Couldn't save" };

function setSaveState(state) {
  $("#save-state").textContent = SAVE_TEXT[state];
  $("#save-state").dataset.state = state;
}

const scheduleSave = debounce(() => saveTask(state.task).catch(fail), 800);

// `beforeRename`: the name the task still has on the server, while a rename is on its way there.
async function saveTask(name, { keepalive = false, beforeRename = null } = {}) {
  const doc = name && state.docs.get(name);
  if (!doc) return;
  if (doc.saving) {
    doc.again = true;
    return doc.saving;
  }
  if (!doc.dirty) return;
  const content = (name === state.task ? taskView.state : doc.state).doc.toString();
  doc.dirty = false;
  if (name === state.task) setSaveState("saving");
  doc.saving = (async () => {
    try {
      const body = { content, base_mtime: doc.mtime };
      // Browsers refuse keepalive requests over 64 KiB; a larger task saves as a normal request.
      const small = new TextEncoder().encode(JSON.stringify(body)).length < 60000;
      const result = await api("PUT", `api/tasks/${enc(beforeRename ?? name)}`, body, { keepalive: keepalive && small, duringRename: Boolean(beforeRename) });
      doc.mtime = result.mtime;
      if (name === state.task) setSaveState(doc.dirty ? "unsaved" : "saved");
    } catch (error) {
      doc.dirty = true;
      if (error.status === 409) {
        doc.conflictMtime = error.data.mtime;
        if (name === state.task) {
          $("#conflict").hidden = false;
          setSaveState("unsaved");
        }
      } else {
        if (name === state.task) setSaveState("failed");
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
    return saveTask(name, { keepalive, beforeRename });
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
    const task = await api("GET", `api/tasks/${enc(name)}`);
    doc.state = taskState(task.content);
    doc.mtime = task.mtime;
    doc.dirty = false;
    doc.conflictMtime = null;
    taskView.setState(doc.state);
    measureWholeDocument(taskView);
    setSaveState("saved");
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
  setSaveState("unsaved");
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
    api("POST", `api/tasks/${enc(state.task)}/feedback`, { action: "add", lane: origin.lane, path: origin.path }).catch(fail);
  }
}

function addReaderSelection() {
  const sel = reader.state.selection.main;
  if (sel.empty) return;
  addToTask(reader.state.sliceDoc(sel.from, sel.to)).catch(fail);
}

// ---------- reader ----------

// Where a document was left is kept as a place in its text, not in pixels: the panel may have a
// different width when the document opens again, and then the same text sits at another height.
// The first character on show in an editor: unlike a pixel position, it survives a new layout.
function topPlace(view) {
  const box = view.scrollDOM.getBoundingClientRect();
  const left = view.contentDOM.getBoundingClientRect().left;
  return view.posAtCoords({ x: left + 1, y: box.top + 1 }, false);
}

const scrollToPlace = (place, doc) => EditorView.scrollIntoView(Math.min(place, doc.length), { y: "start", yMargin: 0 }); // the file may have become shorter

function rememberReaderScroll() {
  if (!state.readerPath) return;
  const place = topPlace(reader);
  if (place > 0) state.readerScroll[state.readerPath] = place;
  else delete state.readerScroll[state.readerPath];
  store.set("reader-scroll", JSON.stringify(state.readerScroll));
}

function restoreReaderScroll(path) {
  const place = state.readerScroll[path];
  reader.scrollDOM.scrollTop = 0;
  if (!Number.isFinite(place) || place <= 0) return;
  reader.dispatch({ effects: scrollToPlace(place, reader.state.doc) });
}

function closeReader() {
  state.readerSeq++;
  if (state.readerPath) {
    rememberReaderScroll();
    reader.setState(readerState(""));
  }
  state.readerPath = null;
  state.readerLocation = null;
  $("#reader-path").textContent = "No document open";
  $("#reader-path").title = "";
  $("#reader-add").hidden = true;
  markCurrentFile();
}

// The editor centers a passage that fits the pane. One taller than the pane it lines up by one end,
// so there the middle is put in the middle by hand, once the editor has scrolled.
function centerWhenTall(passage) {
  const seq = state.readerSeq;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (seq !== state.readerSeq) return;
    reader.requestMeasure({
      read(view) {
        const top = view.coordsAtPos(passage.from, 1), bottom = view.coordsAtPos(passage.to, -1);
        if (!top || !bottom) return 0;
        const box = view.scrollDOM.getBoundingClientRect();
        return bottom.bottom - top.top > box.height ? (top.top + bottom.bottom - box.top - box.bottom) / 2 : 0;
      },
      write(off, view) {
        if (off) view.scrollDOM.scrollTop += off;
      },
    });
  }));
}

// `location`: the passage to highlight. It is scrolled to, unless `place` says where the document
// goes instead: the top of a task's reader as that task left it.
async function openFile(path, location = null, { place = null } = {}) {
  const seq = ++state.readerSeq;
  if (path !== state.readerPath) {
    const { content } = await api("GET", `api/file?path=${enc(path)}`);
    if (seq !== state.readerSeq) return;
    rememberReaderScroll();
    const finding = Boolean(reader.dom.querySelector(".find-panel"));
    reader.setState(readerState(content));
    if (finding) { // a new state starts without the box: keep looking for the same words
      const text = findState.get(reader)?.text ?? "";
      findState.set(reader, { text, anchor: 0 });
      openSearchPanel(reader);
      reader.dispatch({ effects: setSearchQuery.of(findQuery(text)) });
    }
    measureWholeDocument(reader);
    if (!location && place === null) restoreReaderScroll(path);
    state.readerPath = path;
    $("#reader-path").textContent = path;
    $("#reader-path").title = path;
    $("#reader-add").hidden = true;
  }
  revealCurrentFile({ showHidden: place === null });
  state.readerLocation = location;
  const doc = reader.state.doc;
  if (!location) {
    reader.dispatch({ effects: [hitSlot.set.of(Decoration.none), ...(place === null ? [] : [scrollToPlace(place, doc)])] });
    return;
  }
  const line = doc.line(Math.min(location.line + 1, doc.lines));
  const from = line.from + Math.min(location.ch ?? 0, line.length);
  const decos = [];
  let passage = EditorSelection.cursor(from);
  if (location.start !== undefined) {
    // The passage itself, exactly: it may start or end inside a paragraph, which flows as one line.
    const start = doc.line(Math.min(location.start + 1, doc.lines)).from;
    const end = doc.line(Math.min(location.end + 1, doc.lines)).to;
    if (end > start) {
      decos.push(Decoration.mark({ class: "cm-hit-range" }).range(start, end));
      passage = EditorSelection.range(start, end);
    }
  }
  for (const [n, s, e] of location.marks ?? []) {
    if (n >= doc.lines) continue;
    const l = doc.line(n + 1);
    const a = l.from + Math.min(s, l.length);
    const b = l.from + Math.min(e, l.length);
    if (b > a) decos.push(Decoration.mark({ class: "cm-hit-match" }).range(a, b));
  }
  if (place === null) centerWhenTall(passage);
  reader.dispatch({
    selection: EditorSelection.cursor(from),
    effects: [hitSlot.set.of(Decoration.set(decos, true)), place === null ? EditorView.scrollIntoView(passage, { y: "center" }) : scrollToPlace(place, doc)],
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
  if (more) list.append(h("li", { class: "folder" }, `…and ${more.toLocaleString()} more. Narrow your filter to see them.`));
  if (!matches.length) list.append(h("li", { class: "folder" }, state.files.length ? "No files match." : "No Markdown files found."));
}

function toggleFolder(folder) {
  if (!state.openFolders.delete(folder)) state.openFolders.add(folder);
  renderFiles();
  $(`#file-list [data-dir="${CSS.escape(folder)}"]`)?.focus();
}

// The file in the reader stays findable: opening one, from anywhere, opens the folders above it
// and brings it into view in Files. `showHidden` also clears a filter that leaves it out, which a
// reload or a change of task doesn't do: they only bring back what was there.
function revealCurrentFile({ showHidden = true } = {}) {
  const filter = $("#file-filter").value.trim().toLowerCase();
  const filtered = showHidden && filter && !state.readerPath.toLowerCase().includes(filter);
  if (filtered) $("#file-filter").value = "";
  const closed = ancestors(state.readerPath).filter((folder) => !state.openFolders.has(folder));
  for (const folder of closed) state.openFolders.add(folder);
  if (filtered || closed.length) renderFiles();
  else markCurrentFile();
  if (state.panel === "files") $("#file-list .file.current")?.scrollIntoView({ block: "nearest" });
  else state.fileToReveal = true;
}

function markCurrentFile() {
  for (const el of document.querySelectorAll("#file-list .file")) el.classList.toggle("current", el.dataset.path === state.readerPath);
}

async function loadFiles() {
  state.files = await api("GET", "api/files");
  renderFiles();
}

// ---------- results ----------

const passageKey = (passage) => `${passage.path}:${passage.start_line}`;

// Snippet lines flow as prose (hard-wrapped text would otherwise wrap twice in a narrow
// panel); blank lines start a new paragraph. Context lines around the passage are dimmed.
// The dimmed text around a passage, at most this long on each side. It is counted in characters
// because a line can be a whole paragraph, in a document with no line breaks of its own.
const CONTEXT_CHARS = 90;

// What is left of the lines on one side of a passage, nearest first. `cut` keeps the part of a line
// nearest the passage. A line with a highlight, such as a heading that holds the match, stays whole.
function trimContext(lines, cut) {
  let room = CONTEXT_CHARS;
  const kept = [];
  for (const line of lines) {
    if (line.marks.length || !line.text.trim()) {
      kept.push(line);
    } else if (room > 0) {
      const fits = line.text.length <= room;
      kept.push(fits ? line : { ...line, text: cut(line.text, room) });
      room = fits ? room - line.text.length : 0;
    }
  }
  return kept;
}

function shortenedSnippet(snippet) {
  const first = snippet.findIndex((line) => line.in_passage);
  if (first < 0) return snippet;
  const last = snippet.findLastIndex((line) => line.in_passage);
  const before = trimContext(snippet.slice(0, first).reverse(), (text, room) => `…${text.slice(-room).replace(/^\S*\s+/, " ")}`);
  const after = trimContext(snippet.slice(last + 1), (text, room) => `${text.slice(0, room).replace(/\s+\S*$/, " ")}…`);
  return [...before.reverse(), ...snippet.slice(first, last + 1), ...after];
}

function renderSnippet(hit) {
  const body = h("div", { class: "hit-body" });
  let para = null;
  for (const line of shortenedSnippet(hit.snippet)) {
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
  semantic_z: (v) => `meaning score ${v}`,
  lexical_rank: (v) => `keyword rank ${v}`,
  obviousness: (v) => `keyword score ${Math.round(v * 100)}%`,
  themes: (v) => v.join(" ↔ "),
  rarity: (v) => `the topics appear together in ${Math.round((1 - v) * 100)}% of their passages`,
  collected_in_theme: (v) => `${v} collected in this topic`,
  matches_task_line: (v) => `closest to line ${v + 1} of your task`,
};

// For search hits found with Meaning on: where each signal ranked the passage.
function describeSignals(why) {
  if (!("meaning_rank" in why)) return "";
  return [
    why.keyword_rank ? `keywords #${why.keyword_rank}` : "not a top keyword result",
    why.embedded === false ? "no meaning data" : why.meaning_rank ? `meaning #${why.meaning_rank}` : "not a top meaning result",
  ].join(" · ");
}

function describeWhy(why) {
  return Object.entries(why)
    .filter(([k, v]) => v !== null && WHY_LABELS[k])
    .map(([k, v]) => WHY_LABELS[k](v))
    .join(" · ");
}

// The text a result shows when the pointer is on it: what its numbers mean.
function describeHover(why) {
  const lines = ["Click to open this passage in its document."];
  if (why.semantic_z !== undefined && why.semantic_z !== null) lines.push(`The meaning score is ${why.semantic_z}. A typical passage scores 0. A higher score means closer to your task.`);
  if (why.lexical_rank) lines.push(`The keyword rank is ${why.lexical_rank}. Rank 1 is the best match for your task's keywords.`);
  if (why.keyword_rank) lines.push(`The keyword rank is ${why.keyword_rank}. Rank 1 is the best match for your words.`);
  if (why.meaning_rank) lines.push(`The meaning rank is ${why.meaning_rank}. Rank 1 is the closest to your words in meaning.`);
  if (why.obviousness !== undefined) lines.push(`The keyword score is ${Math.round(why.obviousness * 100)}% of the best keyword result's score.`);
  if (why.themes) lines.push(`The related topics in your task are ${why.themes.join(" and ")}.`);
  if (why.rarity !== undefined) lines.push(`These topics appear together in ${Math.round((1 - why.rarity) * 100)}% of their passages. The lower the number, the more unusual the connection.`);
  if (why.collected_in_theme !== undefined) lines.push(`You've collected ${why.collected_in_theme} passages for this topic.`);
  if (why.matches_task_line !== undefined && why.matches_task_line !== null) lines.push(`The closest text in your task is on line ${why.matches_task_line + 1}.`);
  return lines.join("\n");
}

// The document's title and, when it says something more, the heading the passage is under.
function hitTitle(hit, file) {
  if (!hit.title) return hit.heading || file;
  return hit.heading && hit.heading !== hit.title ? `${hit.title} · ${hit.heading}` : hit.title;
}

function renderHit(hit, { lane = null } = {}) {
  const file = withoutExtension(hit.path.split("/").pop());
  const key = passageKey(hit);
  const open = (event) => {
    if (event.target.closest("button, a")) return;
    if (!lane) searchSettled();
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
  // The title opens the whole document from its start, where the rest of the card opens the passage.
  const openDocument = (event) => {
    event.preventDefault();
    if (!lane) searchSettled();
    state.openedHit = null;
    for (const card of document.querySelectorAll(".hit.opened")) card.classList.remove("opened");
    openFile(hit.path, null, { place: 0 }).catch(fail);
  };
  const actions = h("span", { class: "hit-actions" },
    h("button", { class: "add", title: "Add this passage to your task, at your cursor.", onclick: () => { if (!lane) searchSettled(); addToTask(hit.text, lane && { lane, path: hit.path }).catch(fail); } }, icon("plus"), "Add"),
    h("button", { class: "hit-menu-btn", title: "More actions", "aria-label": "More actions", "aria-haspopup": "menu", "aria-expanded": "false", onclick: (e) => openHitMenu(e.currentTarget, hit, lane) }, icon("menu")),
  );
  const why = lane ? describeWhy(hit.why) : describeSignals(hit.why);
  const mark = lane ? markOf(key) : null;
  const marked = mark === "more" ? " liked" : mark === "less" ? " avoided" : "";
  return h("article", { class: `hit${key === state.openedHit ? " opened" : ""}${marked}`, "data-key": key, "data-path": hit.path, "data-meaning-rank": hit.why.meaning_rank, onclick: open, title: describeHover(hit.why) },
    h("div", { class: "hit-head" }, h("a", { class: "hit-title", href: "#", title: hitTitle(hit, file), onclick: openDocument }, hitTitle(hit, file)), actions),
    h("div", { class: "hit-loc", title: `${hit.path}:${hit.focus_line + 1}` }, `${hit.path}:${hit.focus_line + 1}`),
    renderSnippet(hit),
    why && h("div", { class: "hit-why" }, why),
  );
}

// ---------- a result's menu ----------

const MORE_TITLE = "Show more passages like this one, in meaning and in words. This doesn't add it to your task.";
const LESS_TITLE = "Show fewer passages like this one. When you next refresh, it goes away along with passages close to it in meaning.";

function hitMenuItems(hit, lane, card) {
  if (!lane) {
    return [
      { icon: "hide", label: "Hide this passage", title: "Hide this passage from this task's Search. You can bring it back under Hidden in Search.", run: () => hideSearchPassage(hit) },
      { icon: "doc", label: "Hide this whole document", title: `Hide ${hit.path} from this task's Search. You can bring it back under Hidden in Search.`, run: () => hideSearchDoc(hit.path) },
    ];
  }
  const mark = markOf(passageKey(hit));
  return [
    { icon: "more", label: "More like this", checked: mark === "more", title: mark === "more" ? "Click to stop showing more passages like this one." : MORE_TITLE, run: () => markHit(hit, lane, mark === "more" ? null : "more") },
    { icon: "less", label: "Less like this", checked: mark === "less", title: mark === "less" ? "Click to take this back. The passage and those like it can appear again." : LESS_TITLE, run: () => markHit(hit, lane, mark === "less" ? null : "less") },
    null,
    { icon: "hide", label: "Hide this passage", title: "Remove this result. It won't appear again for this task.", run: () => dismiss(hit, lane, card) },
    { icon: "doc", label: "Hide this whole document", title: `Remove all passages from ${hit.path} for this task. Search can still find them.`, run: () => excludeDoc(hit.path) },
  ];
}

let hitMenu = null; // { el, button } while a result's menu is open

function closeHitMenu({ focus = false } = {}) {
  if (!hitMenu) return;
  const { el, button } = hitMenu;
  hitMenu = null;
  el.remove();
  button.setAttribute("aria-expanded", "false");
  if (focus && button.isConnected) button.focus();
}

function openHitMenu(button, hit, lane) {
  const again = hitMenu?.button === button;
  closeHitMenu();
  if (again) return; // a second click closes it
  const items = hitMenuItems(hit, lane, button.closest(".hit")).map((item) => item
    ? h("button", {
      type: "button",
      class: "menu-item",
      role: item.checked === undefined ? "menuitem" : "menuitemcheckbox",
      "aria-checked": item.checked === undefined ? null : String(item.checked),
      title: item.title,
      onclick: () => {
        closeHitMenu({ focus: true }); // back to ⋯, unless the choice removes its card
        Promise.resolve().then(item.run).catch(fail);
      },
    }, icon(item.icon), h("span", { class: "menu-label" }, item.label), item.checked && icon("check"))
    : h("div", { class: "menu-sep", role: "separator" }));
  const el = h("div", { class: "hit-menu", role: "menu" }, items);
  document.body.append(el);
  const box = button.getBoundingClientRect();
  const left = Math.max(8, Math.min(box.right - el.offsetWidth, innerWidth - el.offsetWidth - 8));
  let top = box.bottom + 4;
  if (top + el.offsetHeight > innerHeight - 8) top = Math.max(8, box.top - el.offsetHeight - 4);
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
  button.setAttribute("aria-expanded", "true");
  hitMenu = { el, button };
  el.querySelector(".menu-item").focus();
}

function bindHitMenu() {
  document.addEventListener("pointerdown", (e) => {
    if (hitMenu && !hitMenu.el.contains(e.target) && !hitMenu.button.contains(e.target)) closeHitMenu();
  }, true);
  document.addEventListener("scroll", (e) => {
    if (hitMenu && !hitMenu.el.contains(e.target)) closeHitMenu();
  }, true);
  window.addEventListener("resize", () => closeHitMenu());
  window.addEventListener("blur", () => closeHitMenu());
  document.addEventListener("keydown", (e) => {
    if (!hitMenu) return;
    const items = [...hitMenu.el.querySelectorAll(".menu-item")];
    const at = items.indexOf(document.activeElement);
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      closeHitMenu({ focus: true });
    } else if (e.key === "Tab") {
      closeHitMenu({ focus: true });
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      items[(at + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      items[e.key === "Home" ? 0 : items.length - 1].focus();
    }
  }, true);
}

// ---------- search panel ----------

const SEARCH_LIMIT = 50;
let stopSearchClock = () => {};

async function searchNow() {
  const q = $("#search-input").value;
  const seq = ++state.searchSeq;
  stopSearchClock();
  state.searchedOk = null;
  if (!q.trim()) {
    state.searchRunning = null;
    $("#search-status").textContent = "";
    $("#search-results").replaceChildren();
    $("#search-results").classList.remove("stale");
    searchArrived();
    return;
  }
  state.searchRunning = q;
  $("#search-results").classList.add("stale");
  const stop = stopSearchClock = stopwatch((t) => { $("#search-status").textContent = `Searching… ${t}`; });
  try {
    const tuned = Object.keys(state.params.search).length ? `&params=${enc(JSON.stringify(state.params.search))}` : "";
    const skip = state.skipped.size ? `&skip=${enc(JSON.stringify([...state.skipped]))}` : "";
    const hide = state.hiddenDocs.size ? `&hide=${enc(JSON.stringify([...state.hiddenDocs]))}` : "";
    // Always hybrid: Meaning off is meaning_weight 0, which the server ranks by keywords alone.
    const asTyped = state.searchAsTyped === q ? "&as_typed=true" : "";
    const model = state.model ? `&model=${enc(state.model)}` : "";
    const hidden = new Set(state.hiddenPassages.map(passageKey));
    const limit = Math.min(SEARCH_LIMIT + hidden.size, 200); // the server's most
    const result = await api("GET", `api/search?q=${enc(q)}&mode=hybrid&limit=${limit}${tuned}${skip}${hide}${asTyped}${model}`);
    if (seq !== state.searchSeq) return;
    const hits = result.hits.filter((hit) => !hidden.has(passageKey(hit))).slice(0, SEARCH_LIMIT);
    const n = hits.length;
    $("#search-results").replaceChildren(...hits.map((hit) => renderHit(hit)));
    const took = stop();
    const byMeaning = hits.some((hit) => "meaning_rank" in hit.why);
    const how = !byMeaning ? "" : hits.some((hit) => hit.why.keyword_rank) ? " by keywords and meaning" : " by meaning only";
    const summary = n ? `${n}${n === SEARCH_LIMIT ? "+" : ""} ${n === 1 ? "passage" : "passages"}${how} · ${took}` : `No passages found · ${took}`;
    const correction = result.corrected && [
      "Showing results for ",
      h("strong", {}, result.corrected),
      ". ",
      h("button", { type: "button", class: "as-typed", onclick: () => { state.searchAsTyped = q; runSearch(); } }, `Search for ${q} instead`),
      " · ",
    ];
    $("#search-status").replaceChildren(...(correction || []), summary);
    $("#search-results").classList.remove("stale");
    state.searchRunning = null;
    state.searchedOk = q;
    keepIfSearched();
    searchArrived();
  } catch (error) {
    if (seq !== state.searchSeq) return;
    stop();
    state.searchRunning = null;
    if (error.status !== 400) fail(error);
    $("#search-status").textContent = error.status === 400 ? `⚠ ${error.message}` : "";
    $("#search-results").replaceChildren();
    $("#search-results").classList.remove("stale");
    searchArrived();
  }
}

const runSearch = debounce(searchNow, 200);

// ---------- recent searches ----------

function loadHistory() {
  try { return cleanHistory(JSON.parse(sessionStorage.getItem(HISTORY_KEY))); } catch { return []; }
}

function saveHistory() {
  try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(state.searchHistory)); } catch {}
}

// The box searches while you type, so a query joins the recent searches only once it is settled:
// on Enter, when a result is opened or added, or when the box is left. Even then it waits until it
// has searched without an error, so a regex still being written is not kept.
function searchSettled() {
  const q = $("#search-input").value;
  if (!q.trim()) return;
  state.searchToKeep = q;
  if (state.searchedOk === q) keepIfSearched();
  else if (state.searchRunning !== q) {
    runSearch.cancel();
    searchNow();
  }
}

function keepIfSearched() {
  if (state.searchToKeep === null || state.searchToKeep !== state.searchedOk) return;
  state.searchHistory = remember(state.searchHistory, state.searchToKeep);
  state.searchToKeep = null;
  saveHistory();
}

const historyOpen = () => !$("#search-history").hidden;

function openHistory(at = -1) {
  if (!state.searchHistory.length) return closeHistory();
  state.historyAt = at;
  renderHistory();
  $("#search-history").hidden = false;
  $("#search-input").setAttribute("aria-expanded", "true");
}

function closeHistory() {
  state.historyAt = -1;
  $("#search-history").hidden = true;
  $("#search-input").setAttribute("aria-expanded", "false");
  $("#search-input").removeAttribute("aria-activedescendant");
}

function renderHistory() {
  const list = $("#search-history");
  const items = state.searchHistory.map((q, i) => h("li", {
    id: `search-history-${i}`,
    class: "history-item",
    role: "option",
    "aria-selected": String(i === state.historyAt),
    title: q,
    onclick: (e) => { if (!e.target.closest("button")) useHistory(q); },
  },
  h("span", { class: "history-query" }, q),
  h("button", { type: "button", class: "history-forget", tabindex: "-1", title: "Remove this search from the list. Delete does the same for the one picked with the arrow keys.", "aria-label": `Remove ${q} from recent searches`, onclick: () => forgetSearch(q) }, "×")));
  list.replaceChildren(h("li", { class: "history-head", role: "presentation" }, "Recent searches in this tab"), ...items);
  const input = $("#search-input");
  if (state.historyAt < 0) input.removeAttribute("aria-activedescendant");
  else {
    input.setAttribute("aria-activedescendant", items[state.historyAt].id);
    items[state.historyAt].scrollIntoView({ block: "nearest" });
  }
}

function moveInHistory(step) {
  const n = state.searchHistory.length;
  state.historyAt = state.historyAt < 0 ? (step > 0 ? 0 : n - 1) : (state.historyAt + step + n) % n;
  renderHistory();
}

function useHistory(q) {
  closeHistory();
  $("#search-input").value = q;
  renderPresets("search");
  state.searchToKeep = q; // moves it to the top once it has searched
  runSearch.cancel();
  searchNow();
}

function forgetSearch(q) {
  state.searchHistory = forget(state.searchHistory, q);
  saveHistory();
  if (!state.searchHistory.length) return closeHistory();
  state.historyAt = Math.min(state.historyAt, state.searchHistory.length - 1);
  renderHistory();
}

function bindSearchHistory() {
  const input = $("#search-input");
  $("#search-history").addEventListener("mousedown", (e) => e.preventDefault()); // the box keeps the focus
  input.addEventListener("input", () => {
    state.searchToKeep = null;
    if (!input.value && document.activeElement === input) openHistory();
    else closeHistory();
  });
  input.addEventListener("click", () => {
    if (!input.value && !historyOpen()) openHistory();
  });
  input.addEventListener("blur", closeHistory);
  input.addEventListener("change", searchSettled); // on leaving the box after a change
  input.addEventListener("keydown", (e) => {
    if (e.isComposing || e.ctrlKey || e.metaKey) return;
    const open = historyOpen();
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (open) moveInHistory(e.key === "ArrowDown" ? 1 : -1);
      else if (e.key === "ArrowDown" && state.searchHistory.length) openHistory(0);
      else return;
      e.preventDefault();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (open && state.historyAt >= 0) useHistory(state.searchHistory[state.historyAt]);
      else {
        closeHistory();
        searchSettled();
      }
    } else if (e.key === "Escape" && open) {
      e.preventDefault(); // a search field would clear itself
      closeHistory();
    } else if (e.key === "Delete" && open && state.historyAt >= 0) {
      e.preventDefault();
      forgetSearch(state.searchHistory[state.historyAt]);
    }
  });
}

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

// After Refresh, the first group of results starts at the top of the panel. With few results the
// panel couldn't scroll that far, so the list gets at least the panel's height.
function scrollToBestResults() {
  const panel = $("#panel-related");
  const lanes = $("#related-lanes");
  const heading = lanes.querySelector(".lane h3");
  if (!heading) return;
  lanes.style.minHeight = `${panel.clientHeight}px`;
  panel.scrollTop += heading.getBoundingClientRect().top - panel.getBoundingClientRect().top;
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
      restoreSideScroll("related");
    } catch (error) {
      state.relatedStale = true;
      $("#related-status").textContent = `⚠ ${error.message}`;
      throw error;
    } finally {
      $("#related-status").removeAttribute("aria-busy");
      for (const id of ["#related-lanes", "#related-terms"]) $(id).classList.remove("stale");
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
  $("#related-status").setAttribute("aria-busy", "true");
  for (const id of ["#related-lanes", "#related-terms"]) $(id).classList.add("stale");
  // The time beside Best results counts too, above the fade, as that is where the reader may be.
  const shown = $("#related-lanes .took");
  const stop = stopwatch((t) => {
    $("#related-status").textContent = `Refreshing… ${t}`;
    if (shown) shown.textContent = t;
  });
  let result;
  try {
    result = await api("POST", `api/tasks/${enc(name)}/related`, {
      text: taskView.state.doc.toString(),
      params: state.params.related,
      skip_folders: [...state.skipped],
      model: state.model ?? "",
    });
  } finally {
    stop();
  }
  if (name !== state.task) return;

  state.relatedFor = name;
  state.relatedTerms = result.terms;
  state.relatedLanes = result.lanes;
  rememberLikes(result.prefs);
  // The time taken goes beside the first heading, where Refresh scrolls to.
  const took = stop();
  const lanes = result.lanes.map((lane, i) =>
    h("section", { class: "lane" },
      h("h3", {}, lane.title, h("span", { class: "count" }, String(lane.items.length)),
        i === 0 && h("span", { class: "took", title: "How long these results took" }, took)),
      h("p", { class: "lane-desc" }, lane.description),
      lane.note && h("p", { class: "lane-note" }, lane.note),
      lane.items.map((hit) => renderHit(hit, { lane: lane.name })),
    ),
  );
  renderChips(state.relatedTerms);
  $("#related-lanes").replaceChildren(...lanes);
  showPrefs(result.prefs);
  const s = result.stats;
  const summary = s.gated === undefined
    ? (result.lanes[0]?.note || "")
    : `${s.gated} of ${s.passages.toLocaleString()} passages are clearly related`;
  $("#related-status").textContent = result.lanes.length ? summary : [summary, took].filter(Boolean).join(" · ");
}

// The keywords taken from the task. Boosting writes the word into the steer line; ignoring takes
// a keyword out of the query without hiding the passages that contain it.
function renderChips(terms) {
  const chip = (t) => {
    const [title, term, action] = t.ignored
      ? ["This keyword isn't being used. Click to use it again.", t.term, "unignore"]
      : t.boosted
        ? ["Your word rules boost this keyword. Click to remove it from them.", t.term, "unboost"]
        : ["Boost this keyword. It will always be used and it's added to your word rules.", t.label, "boost"];
    return h("span", { class: `chip${t.boosted ? " boosted" : ""}${t.ignored ? " ignored" : ""}`, style: `--w: ${t.weight.toFixed(2)}` },
      h("button", { class: "chip-label", title, onclick: () => termAction(t, term, action).catch(fail) }, t.label),
      t.ignored
        ? h("button", { class: "chip-x", title: `Hide passages that contain “${t.label}”. This adds -${t.label} to your word rules.`, onclick: () => termAction(t, t.label, "exclude").catch(fail) }, "⊘")
        : h("button", { class: "chip-x", title: "Stop using this keyword. Passages that contain it can still appear. To hide them too, click ⊘ next.", onclick: () => termAction(t, t.term, "ignore").catch(fail) }, "×"),
    );
  };
  $("#related-terms").replaceChildren(...terms.map(chip));
}

// A chip changes at once and the server catches up. If it refuses, the chip goes back.
async function termAction(chip, term, action) {
  if (!chip) { // from the drawer: the word may not be among the task's keywords any more
    await changePrefs("POST", "/terms", { term, action });
    return;
  }
  const before = { boosted: chip.boosted, ignored: chip.ignored, terms: state.relatedTerms };
  chip.boosted = action === "boost";
  chip.ignored = action === "ignore";
  if (action === "exclude") state.relatedTerms = state.relatedTerms.filter((t) => t !== chip);
  renderChips(state.relatedTerms);
  const shown = state.relatedTerms;
  try {
    await changePrefs("POST", "/terms", { term, action });
  } catch (error) {
    if (state.relatedTerms === shown) { // not replaced by a refresh meanwhile
      chip.boosted = before.boosted;
      chip.ignored = before.ignored;
      state.relatedTerms = before.terms;
      renderChips(state.relatedTerms);
    }
    throw error;
  }
}

// Change what the task remembers and show the result, unless the user has moved to another task
// meanwhile: its panel must not get this task's steer line and feedback. Null in that case.
async function changePrefs(method, path, body) {
  const name = state.task;
  const prefs = await api(method, `api/tasks/${enc(name)}${path}`, body);
  if (name !== state.task) return null;
  showPrefs(prefs);
  requestRelated();
  return prefs;
}

// What the panel remembers changed: show it now, whether or not the suggestions are refreshed.
function showPrefs(prefs) {
  state.prefs = prefs;
  state.prefsFor = state.task;
  renderPresets("related");
  renderSteer(prefs.steer);
  renderFeedback(prefs, state.relatedTerms, state.relatedLanes);
  rememberLikes(prefs);
  paintMarks();
}

function paintMarks() {
  for (const card of document.querySelectorAll("#related-lanes .hit")) {
    const mark = markOf(card.dataset.key);
    card.classList.toggle("liked", mark === "more");
    card.classList.toggle("avoided", mark === "less");
  }
}

function rememberLikes(prefs) {
  state.likedIds = new Map(prefs.liked.map((d) => [passageKey(d), d.id]));
  state.avoidedKeys = new Set(prefs.dismissed.filter((d) => d.avoid).map(passageKey));
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

async function dismiss(hit, lane, element) {
  element?.remove();
  await changePrefs("POST", "/feedback", { action: "dismiss", ...passageFields(hit, lane) });
}

// More like this and Less like this show on the card at once, as its colour, and the server
// catches up. Choices for one passage go to the server in turn, since taking a mark back needs
// what the server answered when it was made.
const markRequests = new Map(); // "path:line" -> the latest request for it
const markWanted = new Map(); // "path:line" -> "more", "less" or null, while requests are on their way

const serverMark = (key) => (state.likedIds.has(key) ? "more" : state.avoidedKeys.has(key) ? "less" : null);
const markOf = (key) => (markWanted.has(key) ? markWanted.get(key) : serverMark(key));

async function markHit(hit, lane, mark) {
  const key = passageKey(hit);
  markWanted.set(key, mark);
  paintMarks();
  const previous = markRequests.get(key);
  const request = (async () => {
    await previous?.catch(() => {});
    const now = serverMark(key);
    if (now === mark) return;
    const fields = passageFields(hit, lane);
    if (now === "more" && mark === null) await changePrefs("POST", "/feedback", { action: "unlike", id: state.likedIds.get(key) });
    if (now === "less") await changePrefs("POST", "/feedback", { action: "restore", path: hit.path, text: hit.text });
    if (mark === "more") await changePrefs("POST", "/feedback", { action: "like", ...fields });
    if (mark === "less") await changePrefs("POST", "/feedback", { action: "avoid", ...fields }); // this also takes back a like
  })();
  markRequests.set(key, request);
  try {
    await request;
  } finally {
    if (markRequests.get(key) === request) {
      markRequests.delete(key);
      markWanted.delete(key);
      paintMarks(); // what the server has, also after a failure
    }
  }
}

// ---------- documents hidden from Search ----------

// They are a setting of Search, kept in this browser, because a search needs no task. Related keeps
// its own list for each task.
function cleanHiddenDocs(paths) {
  return new Set(Array.isArray(paths) ? paths.filter((path) => typeof path === "string").slice(0, MAX_HIDDEN_DOCS) : []);
}

function setHiddenDocs(paths) {
  state.hiddenDocs = new Set(paths);
  saveTaskView();
  renderHiddenDocs();
}

// The list used to be one for all tasks: every task that exists then keeps it as its own.
function shareOutHiddenDocs() {
  const shared = store.json(SHARED_HIDDEN_KEY, null);
  if (!shared || !state.tasks.length) return;
  for (const { name } of state.tasks) {
    const view = loadTaskView(name) ?? {};
    if (!view.hiddenDocs) store.set(viewKey(name), JSON.stringify({ ...view, hiddenDocs: shared }));
  }
  store.remove(SHARED_HIDDEN_KEY);
}

function hideSearchDoc(path) {
  if (state.hiddenDocs.size >= MAX_HIDDEN_DOCS) {
    flash(`You can hide up to ${MAX_HIDDEN_DOCS} documents. Bring one back first.`);
    return;
  }
  setHiddenDocs([...state.hiddenDocs, path]);
  for (const card of document.querySelectorAll("#search-results .hit")) {
    if (card.dataset.path === path) card.remove();
  }
  settingChanged("search");
}

const MAX_HIDDEN_PASSAGES = 100;

function cleanHiddenPassages(list) {
  if (!Array.isArray(list)) return [];
  return list.filter((d) => d && typeof d.path === "string" && Number.isInteger(d.start_line)).slice(0, MAX_HIDDEN_PASSAGES)
    .map((d) => ({ path: d.path, start_line: d.start_line, end_line: Number.isInteger(d.end_line) ? d.end_line : d.start_line, text: String(d.text ?? "") }));
}

function setHiddenPassages(list) {
  state.hiddenPassages = list;
  saveTaskView();
  renderHiddenDocs();
}

// Hidden passages are left out by the page, which asks for that many more results to make up.
function hideSearchPassage(hit) {
  if (state.hiddenPassages.length >= MAX_HIDDEN_PASSAGES) {
    flash(`You can hide up to ${MAX_HIDDEN_PASSAGES} passages. Bring one back first.`);
    return;
  }
  const text = hit.text.replace(/\s+/g, " ").trim().slice(0, 200);
  setHiddenPassages([...state.hiddenPassages, { path: hit.path, start_line: hit.start_line, end_line: hit.end_line, text }]);
  for (const card of document.querySelectorAll("#search-results .hit")) {
    if (card.dataset.key === passageKey(hit)) card.remove();
  }
  settingChanged("search");
}

function showSearchPassage(key) {
  setHiddenPassages(state.hiddenPassages.filter((d) => passageKey(d) !== key));
  settingChanged("search");
}

function showSearchDoc(path) {
  setHiddenDocs([...state.hiddenDocs].filter((p) => p !== path));
  settingChanged("search");
}

function renderHiddenDocs() {
  const drawer = $("#search-hidden");
  const paths = [...state.hiddenDocs].sort();
  const passages = state.hiddenPassages;
  drawer.hidden = !paths.length && !passages.length;
  const openLink = (path, d) => h("button", {
    class: "feedback-open",
    title: d ? `Open this document at line ${d.start_line + 1}` : "Open this document",
    onclick: () => openFile(path, d ? { line: d.start_line, start: d.start_line, end: d.end_line } : null).catch(fail),
  }, d ? `${path}:${d.start_line + 1}` : path);
  drawer.replaceChildren(
    sectionHead("Hidden in Search", "Documents and passages you've hidden from Search for this task. This browser remembers them.",
      h("button", { class: "tune-reset", type: "button", title: "Bring back everything you've hidden.", onclick: () => {
        state.hiddenPassages = [];
        setHiddenDocs([]);
        settingChanged("search");
      } }, "Show all"), ` · ${paths.length + passages.length}`),
    h("div", { class: "tune-body" },
      paths.length > 0 && [
        h("div", { class: "tune-group" }, "Documents"),
        h("ul", { class: "feedback-list" }, paths.map((path) =>
          h("li", { class: "feedback-item" },
            h("div", { class: "feedback-head" },
              openLink(path),
              h("button", { class: "undo", title: "Bring this document back into Search.", onclick: () => showSearchDoc(path) }, "Show again"),
            )))),
      ],
      passages.length > 0 && [
        h("div", { class: "tune-group" }, "Passages"),
        h("ul", { class: "feedback-list" }, [...passages].reverse().map((d) =>
          h("li", { class: "feedback-item" },
            h("div", { class: "feedback-head" },
              openLink(d.path, d),
              h("button", { class: "undo", title: "Bring this passage back into Search.", onclick: () => showSearchPassage(passageKey(d)) }, "Show again"),
            ),
            h("div", { class: "feedback-text" }, d.text.length > 160 ? `${d.text.slice(0, 160).trimEnd()}…` : d.text)))),
      ],
    ),
  );
}

async function excludeDoc(path) {
  for (const card of document.querySelectorAll("#related-lanes .hit")) {
    if (card.dataset.path === path) card.remove();
  }
  await changePrefs("POST", "/feedback", { action: "exclude_doc", path });
}

async function includeDoc(path, element) {
  element?.remove();
  await changePrefs("POST", "/feedback", { action: "include_doc", path });
}

// From the drawer: "restore" a dismissed passage, or "unlike" a liked one.
async function undoFeedback(action, id, element) {
  element?.remove();
  await changePrefs("POST", "/feedback", { action, id });
}

// ---------- feedback drawer ----------

// The header of a collapsible section. Its reset button stays in view when the section is closed.
// Text where *a name* refers to a button, menu item or group on the page, which shows in italics.
function emphasized(text) {
  return text.split(/\*([^*]+)\*/).map((part, i) => (i % 2 ? h("em", {}, part) : part));
}

function sectionHead(name, title, reset, count = "") {
  return h("summary", { title }, h("span", { class: "tune-title" }, name, h("span", { class: "tune-count" }, count)), reset);
}

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
  const added = Math.max(counted.reduce((sum, [, [adds]]) => sum + adds, 0) - prefs.liked.length, 0); // More like this also counts for its lane
  const summary = [
    prefs.liked.length && `${prefs.liked.length} more of this`,
    prefs.dismissed.length - avoided && `${prefs.dismissed.length - avoided} removed`,
    avoided && `${avoided} removed with nearby passages`,
    prefs.excluded_docs.length && `${prefs.excluded_docs.length} ${prefs.excluded_docs.length === 1 ? "document" : "documents"} removed`,
    prefs.steer && "word rules",
    added && `${added} added`,
    ignored.length && `${ignored.length} ${ignored.length === 1 ? "keyword" : "keywords"} not in use`,
  ].filter(Boolean);
  const empty = !prefs.liked.length && !prefs.dismissed.length && !prefs.excluded_docs.length && !prefs.steer && !ignored.length && !counted.length;

  const passageItem = (kind, d, undo, tag) => {
    const known = d.start_line !== undefined && d.start_line !== null;
    const location = known ? { line: d.start_line, start: d.start_line, end: d.end_line ?? d.start_line } : null;
    const text = d.text.replace(/\s+/g, " ").trim();
    const where = known ? `${d.path}:${d.start_line + 1}` : d.path;
    return h("li", { class: "feedback-item", "data-kind": kind },
      h("div", { class: "feedback-head" },
        h("button", { class: "feedback-open", title: known ? `Open this document at line ${d.start_line + 1}` : "Open this document", onclick: () => openFile(d.path, location).catch(fail) }, where),
        undo,
      ),
      h("div", { class: "feedback-text" }, tag, text.length > 160 ? `${text.slice(0, 160).trimEnd()}…` : text),
    );
  };
  const dismissed = [...prefs.dismissed].reverse().map((d) => passageItem("removed", d,
    h("button", { class: "undo", title: "Let this passage appear again.", onclick: (e) => undoFeedback("restore", d.id, e.target.closest("li")).catch(fail) }, "Show again"),
    d.avoid && h("span", { class: "feedback-tag", title: "You chose Less like this for this passage, so passages close to it in meaning are removed too." }, "Less like this")));
  const liked = [...prefs.liked].reverse().map((d) => passageItem("liked", d,
    h("button", { class: "undo", title: "Stop looking for more passages like this one.", onclick: (e) => undoFeedback("unlike", d.id, e.target.closest("li")).catch(fail) }, "Stop")));
  const excludedDocs = [...prefs.excluded_docs].reverse().map((path) =>
    h("li", { class: "feedback-item" },
      h("div", { class: "feedback-head" },
        h("button", { class: "feedback-open", title: "Open this document", onclick: () => openFile(path).catch(fail) }, path),
        h("button", { class: "undo", title: "Let passages from this document appear again.", onclick: (e) => includeDoc(path, e.target.closest("li")).catch(fail) }, "Show again"),
      ),
    ));
  const ignoredChip = (term) =>
    h("span", { class: "chip ignored" },
      h("span", { class: "chip-label" }, labels.get(term) ?? term),
      h("button", { class: "chip-x", title: "Use this keyword again.", onclick: () => termAction(state.relatedTerms.find((t) => t.term === term), term, "unignore").catch(fail) }, "×"),
    );
  const laneCounts = counted.map(([lane, [adds, dismisses]]) =>
    `${laneTitles.get(lane) ?? lane} (${[adds && `${adds} ${prefs.liked.length ? "added or liked" : "added"}`, dismisses && `${dismisses} removed`].filter(Boolean).join(", ")})`);

  drawer.replaceChildren(
    sectionHead("Your changes", "Everything you've told Related about this task. It stays when you edit the task's text.",
      h("button", {
        class: "tune-reset",
        type: "button",
        disabled: empty,
        title: "Clear everything in this list for this task.",
        onclick: () => forgetFeedback().catch(fail),
      }, "Remove all changes"), summary.length ? ` · ${summary.join(" · ")}` : ""),
    h("div", { class: "tune-body" },
      empty && h("p", { class: "feedback-note" },
        "No changes for this task yet. This list will show passages you add or want more or less of, passages and documents you hide, your word rules and keywords you stop using. They stay when you edit the task's text."),
      liked.length > 0 && [
        h("div", { class: "tune-group" }, "More of this"),
        h("p", { class: "feedback-note" }, emphasized("Related looks for more passages like these, in meaning and in keywords. They aren't part of your task. “How strong *More like this* is” in *Adjustments* sets how much they count.")),
        h("ul", { class: "feedback-list" }, liked),
      ],
      dismissed.length > 0 && [
        h("div", { class: "tune-group" }, "Removed passages"),
        h("p", { class: "feedback-note" }, emphasized("These passages won't appear again. A passage marked *Less like this* takes nearby passages with it. “How near for *Less like this*” and “Effect of removed passages” in *Adjustments* control this.")),
        h("ul", { class: "feedback-list" }, dismissed),
      ],
      excludedDocs.length > 0 && [
        h("div", { class: "tune-group" }, "Removed documents"),
        h("p", { class: "feedback-note" }, "Passages from these documents won't appear for this task. Search can still find them."),
        h("ul", { class: "feedback-list" }, excludedDocs),
      ],
      prefs.steer && [
        h("div", { class: "tune-group" }, "Word rules"),
        h("code", { class: "feedback-steer" }, prefs.steer),
        h("p", { class: "feedback-note" }, "Change your word rules in the box above the results."),
      ],
      ignored.length > 0 && [
        h("div", { class: "tune-group" }, "Keywords not in use"),
        h("p", { class: "feedback-note" }, "These keywords aren't being used. Passages that contain them can still appear."),
        h("div", { class: "chips" }, ignored.map(ignoredChip)),
      ],
      counted.length > 0 && [
        h("div", { class: "tune-group" }, "Result groups"),
        h("p", { class: "feedback-note" }, `${laneCounts.join(" · ")}. Groups you add passages from usually get more of the results.`),
      ],
    ),
  );
}

async function forgetFeedback() {
  const name = state.task;
  if (!confirm(`Clear all your changes for “${name}”? Removed passages and documents can appear again, your word rules and unused keywords are cleared and all result groups go back to equal.`)) return;
  const prefs = await changePrefs("DELETE", "/prefs");
  flash(`Cleared your changes for “${name}”`);
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
  renderPresets(kind);
  if (kind === "search") runSearch();
  else requestRelated();
}

function formatValue(spec, value) {
  if (spec.type !== "float") return String(value);
  const digits = String(spec.step).split(".")[1]?.length ?? 0;
  return Number(value).toFixed(digits);
}

function paramRow(kind, spec) {
  const value = state.params[kind][spec.name] ?? spec.default;
  const output = h("output", {}, formatValue(spec, value));
  const input = spec.type === "choice"
    ? h("select", {}, spec.choices.map((c) => h("option", { value: c, selected: c === value }, spec.choice_labels[c])))
    : h("input", { type: "range", min: spec.min, max: spec.max, step: spec.step, value });
  // The track is drawn in style.css, which reads how far along it to fill from --fill, from 0 to 1.
  const fill = () => spec.type !== "choice" && input.style.setProperty("--fill", (input.value - spec.min) / (spec.max - spec.min || 1));
  fill();
  input.addEventListener("input", () => {
    fill();
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
    title: `Put this slider back to its default, ${spec.choice_labels?.[spec.default] ?? formatValue(spec, spec.default)}.`,
    onclick: () => setTuned(kind, spec.name, spec.default),
  }, "default");
  input.id = `tune-${kind}-${spec.name}`;
  return h("div", { class: "tune-row", "data-name": spec.name, title: `${spec.help}\nThe default is ${spec.choice_labels?.[spec.default] ?? formatValue(spec, spec.default)}.` },
    h("label", { class: "tune-label", for: input.id }, emphasized(spec.label)), input, spec.type === "choice" ? null : output, toDefault);
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
      rows.push(h("p", { class: "feedback-note" }, spec.group_help));
    }
    rows.push(paramRow(kind, spec));
  }
  $(`#${kind}-tune`).replaceChildren(
    sectionHead("Adjustments", "Change how passages are found and ranked.",
      h("button", {
        class: "tune-reset",
        type: "button",
        title: "Put every slider in this list back to its default.",
        onclick: () => resetParams(kind),
      }, "Set all to default")),
    h("div", { class: "tune-body" },
      h("p", { class: "feedback-note" }, emphasized(
        `Move a slider to change the results and hover over it to see what it does. The “default” button puts it back. ${kind === "search" ? "Results update right away." : "Click *Refresh* to see the new results."} Your adjustments are saved in this browser.`)),
      rows,
    ),
  );
  updateTuner(kind);
}

function updateTuner(kind) {
  const el = $(`#${kind}-tune`);
  const changed = Object.keys(state.params[kind]).length;
  el.querySelector(".tune-count").textContent = changed ? ` · ${changed} changed` : "";
  el.querySelector(".tune-count").dataset.changed = changed;
  el.querySelector(".tune-reset").disabled = !changed;
  for (const row of el.querySelectorAll(".tune-row")) row.classList.toggle("changed", row.dataset.name in state.params[kind]);
  if (kind === "search") $("#search-meaning").setAttribute("aria-pressed", String(meaningWeight() > 0));
}

// ---------- presets ----------

// A preset is a named copy of a panel's settings, kept in this browser like the settings themselves.
// In Search: the sliders, the Meaning button (a slider underneath), the folders, the hidden documents
// and the search box, with any regex in it. In Related: the sliders, the folders, the word rules, the ignored keywords
// and the hidden documents. Likes and hidden passages stay out: they belong to one task's results.
function loadPresets() {
  const saved = store.json("presets", {});
  const clean = (list) => (Array.isArray(list) ? list.filter((p) => typeof p?.name === "string" && p.settings && typeof p.settings === "object") : []);
  return { search: clean(saved?.search), related: clean(saved?.related) };
}

const savePresets = () => store.set("presets", JSON.stringify(state.presets));
const sortedList = (list) => [...new Set(list ?? [])].sort();

function currentSettings(kind, prefs = state.prefsFor === state.task ? state.prefs : null) {
  const settings = { params: { ...state.params[kind] }, skipped: shownSkipped(state.skipped) };
  if (kind === "search") return { ...settings, query: $("#search-input").value, hidden_docs: sortedList(state.hiddenDocs) };
  return { ...settings, steer: prefs?.steer ?? "", ignored: sortedList(prefs?.ignored), excluded_docs: sortedList(prefs?.excluded_docs) };
}

function settingsKey(settings) {
  const params = Object.keys(settings.params ?? {}).sort().map((name) => [name, settings.params[name]]);
  return JSON.stringify({ ...settings, params, skipped: sortedList(settings.skipped), ignored: sortedList(settings.ignored), excluded_docs: sortedList(settings.excluded_docs), hidden_docs: sortedList(settings.hidden_docs) });
}

// Built-in presets come from the server with the sliders. They set only the sliders, so the folders,
// the search box and the word rules stay as they are. A server without any shows none.
function builtinPresets(kind) {
  const list = state.paramSpecs.presets?.[kind];
  return (Array.isArray(list) ? list : []).map((p) => ({ name: p.name, help: p.help, settings: { params: p.params ?? {} }, builtin: true }));
}

// The sliders that differ from their defaults, so that a slider moved back to its default still matches.
function changedParams(kind, params) {
  const specs = new Map(state.paramSpecs[kind].map((s) => [s.name, s]));
  return Object.fromEntries(Object.entries(params ?? {}).filter(([name, value]) => specs.has(name) && value !== specs.get(name).default));
}

function renderPresets(kind) {
  const bar = $(`#${kind}-presets`);
  if (!bar) return;
  const own = state.presets[kind];
  const builtin = builtinPresets(kind);
  const current = currentSettings(kind);
  const key = settingsKey(current);
  const paramsKey = settingsKey({ params: changedParams(kind, current.params) });
  const same = (preset) => preset.builtin ? settingsKey({ params: changedParams(kind, preset.settings.params) }) === paramsKey : settingsKey({ ...preset.settings, skipped: shownSkipped(preset.settings.skipped) }) === key;
  const active = builtin.find((p) => p.name === state.activePreset[kind]) ?? own.find((p) => p.name === state.activePreset[kind]);
  const showing = active && same(active) ? active : own.find(same) ?? builtin.find(same);
  if (showing) state.activePreset[kind] = showing.name;
  const named = showing ?? active;
  const option = (p) => h("option", { value: p.name, title: p.help }, p.name);
  const select = h("select", {
    id: `${kind}-preset`,
    title: "Choose a preset to restore its settings. A built-in preset sets only the sliders.",
    onchange: (e) => e.target.value && applyPreset(kind, e.target.value).catch(fail),
  },
    !showing && h("option", { value: "" }, active ? `${active.name} (changed)` : own.length || builtin.length ? "Current settings" : "No presets yet"),
    builtin.length ? [h("optgroup", { label: "Built in" }, builtin.map(option)), own.length > 0 && h("optgroup", { label: "Saved in this browser" }, own.map(option))] : own.map(option));
  select.value = showing?.name ?? "";
  const deleteTitle = !named ? "Choose a preset to delete it." : named.builtin ? "Built-in presets can't be deleted." : `Delete the preset “${named.name}”. Your current settings stay as they are.`;
  bar.replaceChildren(
    h("label", { for: select.id }, "Preset"),
    select,
    h("button", { type: "button", class: "preset-save", title: `Save ${kind === "search" ? "the sliders, folders, hidden documents, Meaning button and search box" : "the sliders, folders, word rules, keywords not in use and removed documents"} under a name. Presets are saved in this browser.`, onclick: () => savePreset(kind).catch(fail) }, "Save"),
    h("button", { type: "button", class: "preset-delete", title: deleteTitle, disabled: !named || named.builtin, onclick: () => deletePreset(kind, named.name) }, "Delete"),
  );
}

async function savePreset(kind) {
  const builtin = builtinPresets(kind);
  const suggested = builtin.some((p) => p.name === state.activePreset[kind]) ? "" : state.activePreset[kind] ?? "";
  const name = prompt("Name for this preset", suggested)?.trim().slice(0, 60);
  if (!name) return;
  if (builtin.some((p) => p.name === name)) {
    alert(`“${name}” is the name of a built-in preset. Choose another name.`);
    return;
  }
  const presets = state.presets[kind];
  const existing = presets.find((p) => p.name === name);
  if (existing && name !== state.activePreset[kind] && !confirm(`Replace the preset “${name}”?`)) return;
  // The task's own settings come from the server: the panel may not have shown them yet.
  const prefs = kind === "related" && state.task ? await api("GET", `api/tasks/${enc(state.task)}/prefs`) : undefined;
  const settings = currentSettings(kind, prefs);
  if (existing) existing.settings = settings;
  else presets.push({ name, settings });
  presets.sort((a, b) => a.name.localeCompare(b.name));
  state.activePreset[kind] = name;
  savePresets();
  renderPresets(kind);
  flash(`Saved the preset “${name}”`);
}

async function applyPreset(kind, name) {
  const preset = builtinPresets(kind).find((p) => p.name === name) ?? state.presets[kind].find((p) => p.name === name);
  if (!preset) return;
  const { settings } = preset;
  const known = new Set(state.paramSpecs[kind].map((s) => s.name));
  state.params[kind] = Object.fromEntries(Object.entries(settings.params ?? {}).filter(([param]) => known.has(param)));
  saveParams();
  buildTuner(kind);
  if (preset.builtin) {
    state.activePreset[kind] = name;
    settingChanged(kind);
    return;
  }
  const skippedBefore = settingsKey({ skipped: [...state.skipped] });
  // The preset's boxes, keeping what is left out that has no box now.
  state.skipped = new Set([...(settings.skipped ?? []), ...[...state.skipped].filter((name) => !shownPick(name))]);
  renderFolderPicks(state.folders);
  saveSkipped();
  if (settingsKey({ skipped: [...state.skipped] }) !== skippedBefore) settingChanged(kind === "search" ? "related" : "search");
  state.activePreset[kind] = name;
  if (kind === "search") {
    $("#search-input").value = settings.query ?? "";
    setHiddenDocs(settings.hidden_docs ?? []);
    settingChanged(kind);
    return;
  }
  if (!state.task) {
    settingChanged(kind);
    flash("Open a task to restore this preset's word rules, keywords not in use and removed documents");
    return;
  }
  try {
    const prefs = await changePrefs("PUT", "/settings", { steer: settings.steer ?? "", ignored: settings.ignored ?? [], excluded_docs: settings.excluded_docs ?? [] });
    if (prefs) $("#steer-input").value = prefs.steer; // also with the cursor in the box
  } finally {
    renderPresets(kind);
  }
}

function deletePreset(kind, name) {
  if (!confirm(`Delete the preset “${name}”? Your current settings stay as they are.`)) return;
  state.presets[kind] = state.presets[kind].filter((p) => p.name !== name);
  if (state.activePreset[kind] === name) state.activePreset[kind] = null;
  savePresets();
  renderPresets(kind);
}

function resetParams(kind) {
  state.params[kind] = {};
  saveParams();
  buildTuner(kind);
  settingChanged(kind);
}

// A slider is centred on its row, which is the middle of the line box beside it, but digits sit above or
// below that middle depending on the font. This moves the sliders by the difference, measured on the
// font the numbers are drawn in on this computer, so they are level with the numbers.
function levelSliders() {
  const output = $(".tune-row output");
  if (!output) return;
  const style = getComputedStyle(output);
  const ctx = document.createElement("canvas").getContext("2d");
  ctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  const m = ctx.measureText("0123456789");
  const shift = (m.fontBoundingBoxAscent - m.fontBoundingBoxDescent) / 2 - (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2;
  if (Number.isFinite(shift)) document.documentElement.style.setProperty("--slider-shift", `${shift.toFixed(2)}px`);
}

async function loadParamSpecs() {
  state.paramSpecs = await api("GET", "api/params");
  for (const kind of ["search", "related"]) {
    const known = new Set(state.paramSpecs[kind].map((s) => s.name));
    for (const name of Object.keys(state.params[kind])) if (!known.has(name)) delete state.params[kind][name];
    buildTuner(kind);
    renderPresets(kind);
  }
  saveParams();
  renderHiddenDocs();
  document.fonts.ready.then(levelSliders);
}

// ---------- panels ----------

function showPanel(name) {
  rememberSearchTop();
  state.panel = name;
  store.set("panel", name);
  for (const button of document.querySelectorAll(".side-tabs button")) {
    button.setAttribute("aria-selected", String(button.dataset.panel === name));
  }
  for (const panel of ["files", "search", "related"]) $(`#panel-${panel}`).hidden = panel !== name;
  $("#related-refresh").hidden = name !== "related";
  if (name === "files" && state.fileToReveal) {
    state.fileToReveal = false;
    $("#file-list .file.current")?.scrollIntoView({ block: "center" });
  }
  if (name === "related" && state.task && state.relatedFor !== state.task) refreshRelated().catch(fail);
  if (name === "search") {
    if (state.searchPending) searchNow();
    else $("#panel-search").scrollTop = state.searchTop;
    $("#search-input").focus({ preventScroll: true });
  }
}

// ---------- corpus folders ----------

// When several folders are served, Search and Related both show a box for each, to leave some of
// them out. The boxes are one setting: unticking a folder in either panel leaves it out of both.
// They are remembered in this browser, like the parameters.
function loadSkipped() {
  let saved = store.json("skip", null);
  if (!Array.isArray(saved)) {
    // From when each panel had its own boxes: Search's choice wins.
    saved = [store.json("skip-search", null), store.json("skip-related", null)].find(Array.isArray) ?? [];
    store.set("skip", JSON.stringify(saved));
    store.remove("skip-search");
    store.remove("skip-related");
  }
  return new Set(saved.filter((name) => typeof name === "string"));
}

const saveSkipped = () => store.set("skip", JSON.stringify([...state.skipped]));

// Whether the page has a box for this folder or source now. One left out that the server doesn't
// list at the moment, such as a source with no documents, stays left out for when it comes back.
const shownPick = (name) => state.folders.includes(name) || state.sources.some((s) => s.id === name || s.folder === name);
const shownSkipped = (names) => sortedList(names).filter(shownPick);

// The boxes of an osiris archive's sources: `sources` as /api/status lists them, each with its id
// ("md/youtube:auto"), its folder, its lane ("md/youtube"), the lane's name, its own name and its
// number of documents. The skipped set holds the ids of the sources left out.
function renderFolderPicks(folders, sources = state.sources) {
  state.folders = folders;
  state.sources = sources;
  for (const panel of ["search", "related"]) {
    const box = $(`#${panel}-folders`);
    box.hidden = folders.length < 2 && !sources.length;
    box.replaceChildren(box.hidden ? "" : renderFolderTree(panel, folders, sources));
  }
}

// Leaves these folders or sources out, or takes them back, in Search and Related alike.
function setSkipped(names, skip) {
  for (const name of names) {
    if (skip) state.skipped.add(name);
    else state.skipped.delete(name);
  }
  skippedChanged();
}

// Includes exactly the sources in `on` of an archive's `sources`. With none, the archive's folder is
// left out as a whole, so that a kind of document it gets later stays out too. An archive served
// alone has no folder name to leave out, so then each of its sources is.
function setSourcesOn(folder, sources, on) {
  for (const s of sources) state.skipped.delete(s.id);
  if (on.size === 0 && folder) state.skipped.add(folder);
  else {
    state.skipped.delete(folder);
    for (const s of sources) if (!on.has(s.id)) state.skipped.add(s.id);
  }
  skippedChanged();
}

function skippedChanged() {
  saveSkipped();
  const focused = document.activeElement;
  const panel = focused?.closest?.(".folder-picks")?.id;
  renderFolderPicks(state.folders);
  if (panel) { // the boxes are drawn again: keep the focus on the one just used
    [...document.querySelectorAll(`#${panel} input[data-key]`)].find((box) => box.dataset.key === focused.dataset.key)?.focus();
  }
  settingChanged("search");
  settingChanged("related");
}

const TREE_CLOSED_KEY = "folders-closed"; // "1" once the drawer is closed
const TREE_OPEN_KEY = "folder-tree-open"; // node key -> whether it is unfolded, where that isn't the default

// One drawer with a tree of what Search and Related look in: every corpus folder, and under an osiris
// archive its lanes, and under a lane its kinds of document. The tick boxes are in one column, and
// only the names are indented. An archive's box ticks or unticks everything in it, a lane's box all
// its kinds, and a box shows a mix when only some of what is under it is ticked.
function renderFolderTree(panel, folders, sources) {
  const archives = new Set(sources.map((s) => s.folder));
  const unfolded = store.json(TREE_OPEN_KEY, {});
  const isOpen = (key, byDefault) => unfolded[key] ?? byDefault;
  const count = (n) => n.toLocaleString();
  const rows = [];

  const row = ({ depth, key, label, documents, checked, partial = false, title, onchange, children }) => {
    const input = h("input", { type: "checkbox", "data-key": key, checked, onchange, "aria-label": label });
    input.indeterminate = partial;
    const open = children && isOpen(key, depth === 0);
    // Only the box ticks. A click anywhere else on the line of a node with something under it folds or unfolds it.
    rows.push(h("div", {
      class: `tree-row depth-${depth}${children ? " foldable" : ""}`, style: `--depth: ${depth}`, title,
      onclick: children ? (e) => { if (!e.target.closest("input, button")) foldTreeNode(panel, key, !open); } : undefined,
    },
      input,
      h("span", { class: "tree-name" },
        children
          ? h("button", {
            type: "button", class: "tree-fold", "data-fold": key, "aria-expanded": String(open),
            "aria-label": `${open ? "Fold" : "Unfold"} ${label}`,
            onclick: () => foldTreeNode(panel, key, !open),
          }, open ? "▾" : "▸")
          : h("span", { class: "tree-fold" }),
        h("span", { class: "tree-label" }, label)),
      documents !== undefined ? h("span", { class: "source-count" }, count(documents)) : h("span")));
    if (open) children();
  };

  // Archives first, then the folders with nothing under them.
  const order = folders.length ? [...folders.filter((f) => archives.has(f)), ...folders.filter((f) => !archives.has(f))] : [...archives];
  for (const folder of order) {
    if (!archives.has(folder)) {
      row({
        depth: 0, key: folder, label: folder, checked: !state.skipped.has(folder),
        title: `Include the folder ${folder} in Search and Related. Untick to leave it out of both.`,
        onchange: (e) => setSkipped([folder], !e.target.checked),
      });
      continue;
    }
    const mine = sources.filter((s) => s.folder === folder);
    const off = Boolean(folder) && state.skipped.has(folder);
    const on = new Set(off ? [] : mine.filter((s) => !state.skipped.has(s.id)).map((s) => s.id));
    const turn = (ids, include) => {
      const next = new Set(on);
      for (const id of ids) include ? next.add(id) : next.delete(id);
      setSourcesOn(folder, mine, next);
    };
    const lanes = new Map();
    for (const s of mine) lanes.set(s.lane, [...(lanes.get(s.lane) ?? []), s]);
    row({
      depth: 0, key: `${folder}/*`, label: folder || "Sources", documents: mine.reduce((n, s) => n + s.documents, 0),
      checked: on.size === mine.length, partial: on.size > 0 && on.size < mine.length,
      title: `An osiris archive. Ticked, all of it is in Search and Related. Untick it to leave everything in it out, then tick the sources you want.`,
      onchange: () => setSourcesOn(folder, mine, new Set(on.size === mine.length ? [] : mine.map((s) => s.id))),
      children: () => {
        for (const kinds of lanes.values()) {
          const ids = kinds.map((s) => s.id);
          const ticked = ids.filter((id) => on.has(id)).length;
          const name = kinds[0].lane_name;
          const single = kinds.length === 1 && kinds[0].name === name;
          row({
            depth: 1, key: kinds[0].lane, label: name, documents: kinds.reduce((n, s) => n + s.documents, 0),
            checked: ticked === ids.length, partial: ticked > 0 && ticked < ids.length,
            title: single ? `Include ${name} in Search and Related.` : `Include every kind of document from ${name} in Search and Related.`,
            onchange: () => turn(ids, ticked < ids.length),
            children: single ? undefined : () => {
              for (const s of kinds) {
                row({
                  depth: 2, key: s.id, label: s.name, documents: s.documents, checked: on.has(s.id),
                  title: `Include ${s.name} from ${name} in Search and Related.`,
                  onchange: (e) => turn([s.id], e.target.checked),
                });
              }
            },
          });
        }
      },
    });
  }

  const leaves = [...folders.filter((f) => !archives.has(f)).map((f) => !state.skipped.has(f)),
    ...sources.map((s) => !(s.folder && state.skipped.has(s.folder)) && !state.skipped.has(s.id))];
  const included = leaves.filter(Boolean).length;
  const status = included === leaves.length ? "all" : included ? "some left out" : "none";
  const drawer = h("details", { class: "tune folder-tree" },
    h("summary", {}, h("span", { class: "tune-title" }, archives.size ? "Folders and sources" : "Folders", " ",
      h("span", { class: "tune-count" }, `· ${status}`))),
    h("div", { class: "tree" }, rows));
  drawer.open = store.get(TREE_CLOSED_KEY) !== "1";
  drawer.addEventListener("toggle", () => {
    store.set(TREE_CLOSED_KEY, drawer.open ? "" : "1");
    for (const other of document.querySelectorAll(".folder-tree")) if (other !== drawer) other.open = drawer.open;
  });
  return drawer;
}

// Folds or unfolds a node in both panels' trees, keeping the focus on its button.
function foldTreeNode(panel, key, open) {
  store.set(TREE_OPEN_KEY, JSON.stringify({ ...store.json(TREE_OPEN_KEY, {}), [key]: open }));
  renderFolderPicks(state.folders);
  [...document.querySelectorAll(`#${panel}-folders button[data-fold]`)].find((b) => b.dataset.fold === key)?.focus();
}

// ---------- index status ----------

function formatBytes(bytes) {
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  return bytes < 1e9 ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / 1e9).toFixed(2)} GB`;
}

function statusText(s) {
  if (s.state === "embedding") return `Preparing meaning search. ${s.embedded.toLocaleString()} of ${s.to_embed.toLocaleString()} passages done. You can still use Search.`;
  if (s.state === "error") return `Can't read your documents. ${s.error}. See the server log. This is tried again when a file changes.`;
  if (s.state !== "ready") return "Reading your documents…";
  if (s.models?.length) return ""; // the menu shows the model
  return s.model ?? "Keywords only";
}

// Documents, passages and the size of the index, with the chosen model's vectors.
function countsText(s) {
  const model = s.models?.find((m) => m.id === state.model) ?? s.models?.[0];
  const bytes = model?.index_bytes ?? s.index_bytes;
  return [
    `${s.files.toLocaleString()} document${s.files === 1 ? "" : "s"}`,
    `${s.passages.toLocaleString()} passage${s.passages === 1 ? "" : "s"}`,
    bytes ? `index ${formatBytes(bytes)}` : "",
  ].filter(Boolean).join(" · ");
}

function renderIndexStatus(s) {
  $("#index-status").textContent = statusText(s);
  $("#index-size").textContent = countsText(s);
}

// The meaning model this browser uses, from the models the server offers. The first is the default.
function renderModelPick(models) {
  const ids = models.map((m) => m.id);
  const saved = store.get("model");
  state.model = ids.includes(saved) ? saved : ids[0] ?? null;
  const pick = $("#model-pick");
  pick.replaceChildren(...models.map((m) => h("option", { value: m.id }, m.label)));
  pick.value = state.model ?? "";
  $("#model-name").textContent = pick.selectedOptions[0]?.textContent ?? "";
  $("#model-box").hidden = !models.length;
}

function pickModel(id) {
  state.model = id;
  store.set("model", id);
  $("#model-name").textContent = $("#model-pick").selectedOptions[0]?.textContent ?? id;
  if (state.indexStatus) renderIndexStatus(state.indexStatus);
  runSearch();
  if (state.relatedFor) requestRelated();
}

async function pollStatus() {
  let settled = false; // nothing to wait for: poll slowly
  try {
    const s = await api("GET", "api/status");
    state.indexStatus = s;
    settled = s.state === "ready" || s.state === "error";
    renderIndexStatus(s);
    $("#statusbar").dataset.state = s.state;
    if (state.indexUpdated !== null && s.updated !== state.indexUpdated) {
      loadFiles().catch(fail);
      if (state.relatedFor) requestRelated();
    }
    state.indexUpdated = s.updated;
    if (JSON.stringify(s.sources ?? []) !== JSON.stringify(state.sources)) renderFolderPicks(state.folders, s.sources ?? []);
    const outdated = String(s.page_version) !== state.pageVersion;
    $("#statusbar").classList.toggle("update-ready", outdated);
    $("#reload-page").hidden = !outdated;
  } catch {
    $("#index-status").textContent = "Can't reach the server";
  }
  setTimeout(pollStatus, settled ? 10000 : 2000);
}

// What a reload keeps. A tab keeps this until it closes, so a new tab starts anew.
const RESUME_KEY = "isis.resume";

function saveResume() {
  rememberTaskPlace();
  saveTaskView();
  const resume = {
    ...currentView(), // for the task on show, whatever another tab has saved for it since
    task: state.task,
    taskPlaces: state.taskPlaces,
    panel: state.panel,
    openFolders: [...state.openFolders],
    fileFilter: $("#file-filter").value,
    sideScroll: state.panel === "search" ? null : $(`#panel-${state.panel}`).scrollTop,
  };
  try { sessionStorage.setItem(RESUME_KEY, JSON.stringify(resume)); } catch {}
}

function loadResume() {
  try {
    return JSON.parse(sessionStorage.getItem(RESUME_KEY));
  } catch {
    return null;
  }
}

// A panel's results arrive after the page loads, and only then can it scroll to where it was.
function restoreSideScroll(panel) {
  if (state.sideScroll === null || panel !== state.panel) return;
  $(`#panel-${panel}`).scrollTop = state.sideScroll;
  state.sideScroll = null;
}

// Reloads only once every task is saved, so that no text is lost.
async function reloadPage() {
  const button = $("#reload-page");
  button.disabled = true;
  try {
    await Promise.all([...state.docs.keys()].map((name) => saveTask(name)));
    if (![...state.docs.values()].some((d) => d.dirty || d.saving)) {
      location.reload();
      return;
    }
  } catch {
    // The text below says what to do.
  }
  button.textContent = "COULD NOT SAVE YOUR TASK. PRESS HERE TO TRY AGAIN.";
  button.disabled = false;
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

// The theme menu at the bottom left: light unless dark is chosen, and kept in this browser.
// theme.js applies the kept choice before the page is drawn.
function bindTheme() {
  const pick = $("#theme-pick");
  const show = () => ($("#theme-name").textContent = pick.selectedOptions[0].textContent);
  pick.value = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  show();
  pick.addEventListener("change", () => {
    if (pick.value === "dark") document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    store.set("theme", pick.value);
    show();
  });
}

// Drawers stay open or closed across reloads.
function rememberOpen(drawer, key) {
  drawer.open = store.get(key) === "open";
  drawer.addEventListener("toggle", () => store.set(key, drawer.open ? "open" : "closed"));
}

function bind() {
  bindTheme();
  for (const kind of ["search", "related"]) rememberOpen($(`#${kind}-tune`), `tune-${kind}`);
  rememberOpen($("#related-feedback"), "feedback");
  for (const kind of ["search", "related"]) rememberOpen($(`#${kind}-help`), `help-${kind}`);
  bindResizers();
  $("#new-task").addEventListener("click", () => newTask().catch(fail));
  bindTaskTabs();
  bindHitMenu();
  $("#reader-add").addEventListener("click", addReaderSelection);
  $("#conflict-load").addEventListener("click", () => resolveConflict(false).catch(fail));
  $("#conflict-keep").addEventListener("click", () => resolveConflict(true).catch(fail));
  $("#file-filter").addEventListener("input", debounce(renderFiles, 120));
  reader.scrollDOM.addEventListener("scroll", debounce(rememberReaderScroll, 250));
  window.addEventListener("pagehide", rememberReaderScroll);
  window.addEventListener("pagehide", saveResume);
  $("#search-input").addEventListener("input", runSearch);
  bindSearchHistory();
  $("#search-meaning").addEventListener("click", toggleMeaning);
  $("#search-input").addEventListener("input", () => renderPresets("search"));
  $("#related-refresh").addEventListener("click", () => refreshRelated().then(scrollToBestResults).catch(fail));
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
  // Ctrl/⌘+F finds in the pane last clicked. With the cursor elsewhere, such as a box in the side
  // panel, it stays the browser's own find.
  let lastPane = null;
  const paneOf = (target) => (target?.closest?.("#reader") ? reader : target?.closest?.("#task") ? taskView : null);
  document.addEventListener("pointerdown", (e) => {
    lastPane = paneOf(e.target);
  }, true);
  document.addEventListener("keydown", (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.shiftKey || e.altKey || e.key.toLowerCase() !== "f") return;
    const active = document.activeElement;
    const pane = paneOf(active) ?? (active === document.body ? lastPane : null);
    if (!pane || (pane === taskView && !state.task)) return;
    e.preventDefault();
    openFind(pane);
  }, true);
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      showPanel("search");
      $("#panel-search").scrollTop = 0; // back to the box
    }
  });
  $("#reload-page").addEventListener("click", reloadPage);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "hidden") return;
    saveTaskView();
    saveTask(state.task, { keepalive: true }).catch(() => {});
  });
  window.addEventListener("beforeunload", (e) => {
    // A save still on its way counts as unsaved: closing the page could cancel it.
    if ([...state.docs.values()].some((d) => d.dirty || d.saving)) e.preventDefault();
  });
}

async function init() {
  bind();
  await loadParamSpecs();
  const status = await api("GET", "api/status");
  // The version these files came with. Asking the server now would make an old page look current.
  state.pageVersion = $('meta[name="page-version"]')?.content || String(status.page_version);
  renderFolderPicks(status.folders, status.sources ?? []);
  renderModelPick(status.models ?? []);
  $("#model-pick").addEventListener("change", (e) => pickModel(e.target.value));
  if (!status.model) {
    const button = $("#search-meaning");
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    button.title = "No meaning model is loaded, so search uses only your words.";
  }
  const resume = loadResume();
  state.sideScroll = resume?.sideScroll ?? null;
  state.taskPlaces = resume?.taskPlaces ?? {};
  if (resume) {
    state.openFolders = new Set(resume.openFolders ?? []);
    $("#file-filter").value = resume.fileFilter ?? "";
  }
  showPanel(resume?.panel || store.get("panel") || "search");
  await Promise.all([loadTasks(), loadFiles()]);
  shareOutHiddenDocs();
  const last = resume?.task ?? store.get("task");
  const name = state.tasks.find((t) => t.name === last)?.name ?? state.tasks[0]?.name;
  if (name) {
    await switchTask(name, resume?.task === name ? { ...loadTaskView(name), ...resume } : undefined);
  } else {
    showNoTask();
    if (resume) await showView(resume);
  }
  restoreSideScroll("files"); // opening a document scrolls the list to it
  pollStatus();
  preloadTasks();
}

init().catch(fail);

// Exposed for debugging and end-to-end tests.
window.isis = { state, reader, taskView, openFile, switchTask, addToTask, refreshRelated, renderPresets };
