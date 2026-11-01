// Hard-wrapped text (lines cut at ~70 characters, as in most plain-text books) is shown as
// flowing paragraphs: a line break inside a paragraph is displayed as a space, the way markdown
// renders it. The text itself is unchanged. Markdown structure keeps its line breaks.

const FENCE = /^\s*(```|~~~)/;
const STARTS_BLOCK = /^(#{1,6}(\s|$)|[-*+•◦▪‣]\s|\d+[.)]\s|>|\||<|(?:[-*_]\s*){3,}$|=+\s*$)/; // heading, list item, quote, table, HTML, rule, underline
const ENDS_LINE = /^(#{1,6}(\s|$)|\||(?:[-*_]\s*){3,}$|=+\s*$)|( {2}|\\)$/; // heading, table row, rule, underline, hard break

const DASHES = /^-{3,}\s*$/;
const RULE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const YAML = /^\s*([\w.-]+:(\s|$)|-\s|#)/;
const MAX_ENVELOPE = 200; // lines

// A block enveloped by --- lines, opening at line i: front matter at the top of a file (which may
// have blank lines, and may end with ...), or further down, a --- after a blank line, lines with
// no blank between them, and a closing ---. Markdown would make the text above the closing --- a
// heading. `meta` says the lines read as YAML: they are data, not prose.
export function envelopeAt(lines, i) {
  if (!DASHES.test(lines[i]) || (i > 0 && lines[i - 1].trim())) return null;
  for (let j = i + 1; j < Math.min(lines.length, i + MAX_ENVELOPE); j++) {
    const closes = DASHES.test(lines[j]) || (i === 0 && lines[j].trim() === "...");
    if (closes) {
      if (j === i + 1) return null;
      const meta = lines.slice(i + 1, j).every((line) => !line.trim() || YAML.test(line));
      return { open: i, close: j, meta };
    }
    if (!lines[j].trim() && i > 0) return null;
  }
  return null;
}

// The enveloped blocks of a document, and the other lines that are horizontal rules.
export function structure(lines) {
  const envelopes = [];
  const rules = [];
  let fence = false;
  for (let i = 0; i < lines.length; i++) {
    if (FENCE.test(lines[i])) fence = !fence;
    if (fence) continue;
    const envelope = envelopeAt(lines, i);
    if (envelope) {
      envelopes.push(envelope);
      i = envelope.close;
    } else if (RULE.test(lines[i]) && (i === 0 || !lines[i - 1].trim())) {
      rules.push(i);
    }
  }
  return { envelopes, rules };
}

const ITEM = /^(\s*)([-*+•◦▪‣]|\d+[.)])(\s+)/;
const indentOf = (line) => {
  let width = 0;
  for (const c of /^\s*/.exec(line)[0]) width += c === "\t" ? 4 - (width % 4) : 1;
  return width;
};

// The lists of a document. `items`: each bullet or number, with its nesting depth (from how far it
// is indented compared with the items above it, however many spaces that takes) and where its text
// starts. `continuations`: indented lines that carry on an item, with the depth of that item;
// `joined` means right below it, so it flows on, and not after a blank line. Not code: nothing in a
// fence, or indented that far without a list above it.
export function listStructure(lines) {
  const items = [];
  const continuations = [];
  const edges = new Set(structure(lines).envelopes.flatMap(({ open, close }) => [open, close]));
  let stack = [];
  let fence = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (FENCE.test(line)) {
      fence = !fence;
      stack = [];
      continue;
    }
    if (fence || edges.has(i)) {
      stack = [];
      continue;
    }
    if (!line.trim()) continue;
    const indent = indentOf(line);
    const item = RULE.test(line) ? null : ITEM.exec(line);
    if (item && (stack.length || indent < 4)) {
      while (stack.length && stack[stack.length - 1] > indent) stack.pop();
      if (!stack.length || stack[stack.length - 1] < indent) stack.push(indent);
      items.push({
        line: i,
        depth: stack.length - 1,
        ordered: /\d/.test(item[2]),
        marker: item[2],
        markerFrom: item[1].length,
        markerTo: item[1].length + item[2].length,
        textFrom: item[0].length,
      });
    } else if (stack.length && indent > stack[0]) {
      const depth = Math.max(stack.filter((width) => width < indent).length - 1, 0);
      continuations.push({ line: i, depth, indent: /^\s*/.exec(line)[0].length, joined: Boolean(lines[i - 1].trim()) });
    } else if (stack.length && !lines[i - 1].trim()) {
      stack = []; // plain text after a blank line ends the list
    }
  }
  return { items, continuations };
}

// What Enter does on line `n` at character `ch` when that starts a new item: straight on the next
// line, whether or not the list has blank lines between its items. `prefix` starts the new line;
// `renumber` lists the numbers below it that move up. Null when Enter should do what it does
// elsewhere: outside a list, before the item's text, and on an empty item, which ends the list.
export function enterInList(lines, n, ch) {
  const { items, continuations } = listStructure(lines);
  const onItem = items.find((item) => item.line === n);
  const carriesOn = continuations.some((c) => c.line === n && c.joined);
  const owner = onItem ?? (carriesOn ? items.filter((item) => item.line < n).at(-1) : null);
  if (!owner) return null;
  const text = lines[owner.line];
  if (onItem && (ch < owner.textFrom || !text.slice(owner.textFrom).trim())) return null;
  const box = /^\[[ xX]\]\s+/.exec(text.slice(owner.textFrom));
  const lead = text.slice(0, owner.markerFrom);
  const gap = text.slice(owner.markerTo, owner.textFrom);
  const renumber = [];
  let marker = owner.marker;
  if (owner.ordered) {
    const number = parseInt(owner.marker, 10);
    const close = owner.marker.slice(-1);
    marker = `${number + 1}${close}`;
    let next = number + 1;
    for (const item of items.filter((other) => other.line > owner.line)) {
      if (item.depth < owner.depth) break;
      if (item.depth === owner.depth && item.ordered) {
        next += 1;
        renumber.push({ line: item.line, from: item.markerFrom, to: item.markerTo, text: `${next}${item.marker.slice(-1)}` });
      }
    }
  }
  return { prefix: `${lead}${marker}${gap}${box ? "[ ] " : ""}`, renumber };
}

// Indices i of the lines whose break to line i + 1 shows as a space.
export function softBreaks(lines) {
  const out = [];
  const carriesOn = new Set(listStructure(lines).continuations.filter((c) => c.joined).map((c) => c.line));
  const keep = new Set();
  for (const { open, close, meta } of structure(lines).envelopes) {
    if (meta) for (let i = open; i <= close; i++) keep.add(i);
  }
  let fence = false;
  for (let i = 0; i < lines.length - 1; i++) {
    const line = lines[i];
    if (FENCE.test(line)) {
      fence = !fence;
      continue;
    }
    if (keep.has(i)) continue;
    const next = lines[i + 1];
    if (fence || !line.trim() || !next.trim() || ENDS_LINE.test(line)) continue;
    if ((/^\s/.test(next) && !carriesOn.has(i + 1)) || STARTS_BLOCK.test(next) || FENCE.test(next)) continue; // indented: verse, code
    out.push(i);
  }
  return out;
}

// The paragraph (first and last line index) around line `n`: the run of non-blank lines.
export function paragraphAt(lines, n) {
  let lo = n;
  let hi = n;
  while (lo > 0 && lines[lo - 1].trim()) lo--;
  while (hi < lines.length - 1 && lines[hi + 1].trim()) hi++;
  return [lo, hi];
}
