// Run with: node --test web/insert.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { planInsertion } from "./insert.js";

const apply = (doc, cursor, text) => {
  const p = planInsertion(doc, cursor, text);
  return { doc: doc.slice(0, p.from) + p.insert + doc.slice(p.from), cursor: p.cursorAfter };
};

test("inserts into an empty document", () => {
  assert.equal(apply("", 0, "hello").doc, "hello\n");
});

test("appends as a separate paragraph", () => {
  for (const doc of ["one", "one\n", "one\n\n"]) assert.equal(apply(doc, doc.length, "two").doc, "one\n\ntwo\n");
});

test("inserts between paragraphs at a line start", () => {
  const doc = "one\n\nthree\n";
  assert.equal(apply(doc, doc.indexOf("three"), "two").doc, "one\n\ntwo\n\nthree\n");
});

test("moves a mid-line cursor to the end of the line", () => {
  assert.equal(apply("alpha beta\ngamma\n", 3, "X").doc, "alpha beta\n\nX\n\ngamma\n");
});

test("keeps consecutive inserts in order", () => {
  let s = apply("intro\n\noutro\n", 5, "A");
  s = apply(s.doc, s.cursor, "B");
  s = apply(s.doc, s.cursor, "C");
  assert.equal(s.doc, "intro\n\nA\n\nB\n\nC\n\noutro\n");
});

test("trims surrounding blank lines but keeps inner structure", () => {
  assert.equal(apply("", 0, "\n\n  a\n\nb  \n\n").doc, "  a\n\nb\n");
});
