import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanHistory, forget, remember } from "./history.js";

test("a new search goes first", () => {
  assert.deepEqual(remember(["whale"], "ship"), ["ship", "whale"]);
});

test("a search made again moves to the top instead of showing twice", () => {
  assert.deepEqual(remember(["ship", "whale", "sea"], "whale"), ["whale", "ship", "sea"]);
  assert.deepEqual(remember(["whale"], "  whale "), ["whale"]);
});

test("an empty query is not kept", () => {
  assert.deepEqual(remember(["whale"], "   "), ["whale"]);
});

test("the oldest searches drop off past the limit", () => {
  assert.deepEqual(remember(["b", "c", "d"], "a", 3), ["a", "b", "c"]);
});

test("a search can be removed", () => {
  assert.deepEqual(forget(["a", "b", "c"], "b"), ["a", "c"]);
});

test("a saved list that can't be read starts empty", () => {
  assert.deepEqual(cleanHistory(null), []);
  assert.deepEqual(cleanHistory("whale"), []);
  assert.deepEqual(cleanHistory(["whale", 3, "", " ship ", "whale"]), ["whale", "ship"]);
});
