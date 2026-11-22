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

test("each folder shows its first files and counts the rest", () => {
  const paths = ["big/1.md", "big/2.md", "big/3.md", "big/4.md", "small/a.md", "top.md"];
  const view = visibleRows(fileTree(paths), () => true, 2);
  const shown = view.rows.map((r) => (r.kind === "more" ? `${"  ".repeat(r.depth)}+${r.count}` : `${"  ".repeat(r.depth)}${r.name}`));
  assert.deepEqual(shown, ["big", "  1.md", "  2.md", "  +2", "small", "  a.md", "top.md"]); // the folders below keep their files
});

test("the open file shows even beyond a folder's limit", () => {
  const paths = ["big/1.md", "big/2.md", "big/3.md", "big/4.md"];
  const view = visibleRows(fileTree(paths), () => true, 1, "big/3.md");
  assert.deepEqual(view.rows.map((r) => r.name ?? `+${r.count}`), ["big", "1.md", "3.md", "+2"]);
});

test("the folders above a file", () => {
  assert.deepEqual(ancestors("a/b/c/deep.md"), ["a", "a/b", "a/b/c"]);
  assert.deepEqual(ancestors("top.md"), []);
});
