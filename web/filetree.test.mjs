import { test } from "node:test";
import assert from "node:assert/strict";
import { ancestors, fileTree, visibleRows } from "./filetree.js";

const PATHS = ["a/b/c/deep.md", "a/b/two.md", "a/one.md", "top.md", "z/last.md"];
const names = ({ rows }) => rows.map((r) => `${"  ".repeat(r.depth)}${r.name}`);

test("folders start collapsed: only the top level shows", () => {
  const view = visibleRows(fileTree(PATHS), () => false);
  assert.deepEqual(names(view), ["a", "z", "top.md"]);
  assert.deepEqual(view.rows.map((r) => r.count), [3, 1, undefined]);
});

test("opening a folder shows one more level", () => {
  const open = new Set(["a"]);
  assert.deepEqual(names(visibleRows(fileTree(PATHS), (p) => open.has(p))), ["a", "  b", "  one.md", "z", "top.md"]);
});

test("every level can be open", () => {
  assert.deepEqual(names(visibleRows(fileTree(PATHS), () => true)),
    ["a", "  b", "    c", "      deep.md", "    two.md", "  one.md", "z", "  last.md", "top.md"]);
});

test("the file limit counts what it leaves out", () => {
  const view = visibleRows(fileTree(PATHS), () => true, 2);
  assert.equal(view.rows.filter((r) => r.kind === "file").length, 2);
  assert.equal(view.more, 3);
});

test("the folders above a file", () => {
  assert.deepEqual(ancestors("a/b/c/deep.md"), ["a", "a/b", "a/b/c"]);
  assert.deepEqual(ancestors("top.md"), []);
});
