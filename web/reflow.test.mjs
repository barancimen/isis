import { test } from "node:test";
import assert from "node:assert/strict";
import { enterInList, envelopeAt, linesNotProse, listStructure, paragraphAt, softBreaks, structure } from "./reflow.js";

test("hard-wrapped paragraphs flow", () => {
  const lines = ["What the white whale was to Ahab, has been", "hinted; what, at times, he", "was to me.", "", "Next paragraph."];
  assert.deepEqual(softBreaks(lines), [0, 1]);
});

test("markdown structure keeps its line breaks", () => {
  const lines = [
    "# A heading", "text under it", "",
    "- one", "- two", "",
    "> quoted", "> more", "",
    "| a | b |", "|---|---|", "",
    "a hard break  ", "after it", "",
    "Setext title", "===", "",
    "prose", "    indented verse", "    more verse",
  ];
  assert.deepEqual(softBreaks(lines), []);
  assert.deepEqual(softBreaks(["- a list item that", "continues unindented"]), [0]);
});

test("code fences are left alone", () => {
  const lines = ["```", "a = 1", "b = 2", "```", "prose that", "continues"];
  assert.deepEqual(softBreaks(lines), [4]);
});

test("the paragraph around a line", () => {
  const lines = ["a", "b", "", "c", "d", "e", "", "f"];
  assert.deepEqual(paragraphAt(lines, 4), [3, 5]);
  assert.deepEqual(paragraphAt(lines, 7), [7, 7]);
});

test("front matter is an envelope, even with blank lines, and keeps its lines", () => {
  const lines = ["---", "title: A", "", "author: B", "tags:", "  - x", "---", "", "Prose that", "flows."];
  assert.deepEqual(envelopeAt(lines, 0), { open: 0, close: 6, meta: true });
  assert.deepEqual(softBreaks(lines), [8]);
  assert.equal(envelopeAt(["---", "title: A", "..."], 0).close, 2);
  assert.equal(envelopeAt(["---", "title: A", "no end"], 0), null);
});

test("an envelope further down needs a blank line above and none inside", () => {
  assert.deepEqual(envelopeAt(["text", "", "---", "note: x", "---"], 2), { open: 2, close: 4, meta: true });
  assert.equal(envelopeAt(["text", "---", "note: x", "---"], 1), null); // that first --- makes "text" a heading
  assert.equal(envelopeAt(["text", "", "---", "", "para", "", "---"], 2), null); // two rules around a paragraph
  assert.equal(envelopeAt(["", "---", "---"], 1), null);
});

test("prose in an envelope still flows; YAML in one does not", () => {
  assert.deepEqual(softBreaks(["", "---", "A wrapped", "sentence.", "---"]), [2]);
  assert.deepEqual(envelopeAt(["", "---", "A wrapped", "sentence.", "---"], 1).meta, false);
  assert.deepEqual(softBreaks(["", "---", "a: 1", "b: 2", "---"]), []);
});

test("rules and real headings are told apart", () => {
  const lines = ["Title", "---", "", "---", "", "para", "", "***", "", "```", "---", "x", "---", "```"];
  const { envelopes, rules } = structure(lines);
  assert.deepEqual(rules, [3, 7]); // "Title" above --- is a heading; nothing in the code fence counts
  assert.deepEqual(envelopes, []);
});

const depths = (lines) => listStructure(lines).items.map((item) => [item.line, item.depth]);

test("nesting comes from relative indent, whatever its width", () => {
  assert.deepEqual(depths(["- a", "  - b", "    - c", "- d"]), [[0, 0], [1, 1], [2, 2], [3, 0]]);
  assert.deepEqual(depths(["- a", "    - b", "        - c", "    - d"]), [[0, 0], [1, 1], [2, 2], [3, 1]]);
  assert.deepEqual(depths(["- a", "\t- b", "\t\t- c"]), [[0, 0], [1, 1], [2, 2]]);
  assert.deepEqual(depths(["1. a", "   - b", "10. c"]), [[0, 0], [1, 1], [2, 0]]);
  assert.deepEqual(depths(["- a", "", "  - b"]), [[0, 0], [2, 1]]); // a blank line doesn't end a list
});

test("what each item is", () => {
  const { items } = listStructure(["* a", "+ b", "12. c", "3) d", "-   wide"]);
  assert.deepEqual(items.map((i) => [i.marker, i.ordered, i.textFrom]), [["*", false, 2], ["+", false, 2], ["12.", true, 4], ["3)", true, 3], ["-", false, 4]]);
});

test("not lists: rules, code, fences, or a dash with no space", () => {
  const not = [["---"], ["* * *"], ["- - -"], ["-word"], ["2.5 is a number"], ["", "    - indented code"],
    ["```", "- in a fence", "```"]];
  for (const lines of not) assert.deepEqual(depths(lines), [], lines.join("|"));
  assert.deepEqual(depths(["- a", "```", "- b", "```", "- c"]), [[0, 0], [4, 0]]);
});

test("continuation lines flow on, and a later paragraph keeps its item's indent", () => {
  const lines = ["- hard wrapped", "  and more", "", "  a new paragraph", "- next", "  - sub", "    its second line", "", "plain text", "  - not nested in anything"];
  const { continuations, items } = listStructure(lines);
  assert.deepEqual(continuations.map((c) => [c.line, c.depth, c.joined]), [[1, 0, true], [3, 0, false], [6, 1, true]]);
  assert.deepEqual(softBreaks(lines), [0, 5]);
  assert.deepEqual(items.map((i) => i.line), [0, 4, 5, 9]);
  assert.equal(items[3].depth, 0); // the list above ended at the plain text
});

test("lists inside --- sections, and text that already has bullet characters", () => {
  assert.deepEqual(depths(["Text", "", "---", "- a", "  - b", "---", "", "After"]), [[3, 0], [4, 1]]);
  assert.deepEqual(depths(["---", "title: x", "tags:", "  - one", "---"]), [[3, 0]]);
  assert.deepEqual(depths(["• pasted", "• bullets", "  ◦ nested"]), [[0, 0], [1, 0], [2, 1]]);
  assert.deepEqual(softBreaks(["• pasted", "• bullets"]), []); // each stays on its own line
});

test("Enter in a list starts the next item, in the same style", () => {
  const at = (lines, n, ch = lines[n].length) => enterInList(lines, n, ch);
  assert.equal(at(["- one", "- two"], 1).prefix, "- ");
  assert.equal(at(["* one", "  * two"], 1).prefix, "  * ");
  assert.equal(at(["-   wide"], 0).prefix, "-   ");
  assert.equal(at(["- [x] done"], 0).prefix, "- [ ] ");
  assert.equal(at(["- one", "", "- two"], 2).prefix, "- "); // nothing about the blank line above
  assert.equal(at(["- hard wrapped", "  and more"], 1).prefix, "- "); // from a line that carries on
});

test("Enter leaves other places to the editor", () => {
  assert.equal(enterInList(["plain text"], 0, 3), null);
  assert.equal(enterInList(["- "], 0, 2), null); // an empty item ends the list
  assert.equal(enterInList(["- text"], 0, 1), null); // before the text
  assert.equal(enterInList(["```", "- code", "```"], 1, 6), null);
});

test("numbers below a new item move up, at that level only", () => {
  const { prefix, renumber } = enterInList(["1. one", "2. two", "   - sub", "3. three", "", "- other"], 0, 6);
  assert.equal(prefix, "2. ");
  assert.deepEqual(renumber.map((r) => [r.line, r.text]), [[1, "3."], [3, "4."]]);
  assert.equal(enterInList(["1) a", "2) b"], 1, 4).prefix, "3) ");
});

test("code, headings and table rows are not prose", () => {
  const lines = ["# Title", "Some text", "```", "code", "```", "    indented", "| a | b |", "- item", "    more of the item", "End"];
  assert.deepEqual(linesNotProse(lines, new Set([7, 8])), [0, 2, 3, 4, 5, 6]);
});
