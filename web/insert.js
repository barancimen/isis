// Where and how a selection goes into a task document: as its own paragraph, at the cursor.
// A cursor in the middle of a line moves to the end of that line, so a line is never split.

export function planInsertion(doc, cursor, text) {
  const body = text.replace(/^\s*\n/, "").replace(/\s+$/, "");

  let from = Math.max(0, Math.min(cursor, doc.length));
  if (from > 0 && doc[from - 1] !== "\n") {
    const eol = doc.indexOf("\n", from);
    from = eol === -1 ? doc.length : eol;
  }

  const before = doc.slice(0, from);
  const after = doc.slice(from);

  let lead = "";
  if (before.length > 0 && !before.endsWith("\n\n")) lead = before.endsWith("\n") ? "\n" : "\n\n";

  let trail;
  if (after.length === 0) trail = "\n";
  else if (after.startsWith("\n\n")) trail = "";
  else if (after.startsWith("\n")) trail = "\n";
  else trail = "\n\n";

  // cursorAfter sits just past the inserted text, so consecutive inserts stay in order.
  return { from, insert: lead + body + trail, cursorAfter: from + lead.length + body.length };
}
