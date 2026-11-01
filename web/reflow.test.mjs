import { test } from "node:test";
import assert from "node:assert/strict";
import { envelopeAt, paragraphAt, softBreaks, structure } from "./reflow.js";

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
