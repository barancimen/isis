import { test } from "node:test";
import assert from "node:assert/strict";
import { paragraphAt, softBreaks } from "./reflow.js";

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
