// Hard-wrapped text (lines cut at ~70 characters, as in most plain-text books) is shown as
// flowing paragraphs: a line break inside a paragraph is displayed as a space, the way markdown
// renders it. The text itself is unchanged. Markdown structure keeps its line breaks.

const FENCE = /^\s*(```|~~~)/;
const STARTS_BLOCK = /^(#{1,6}(\s|$)|[-*+]\s|\d+[.)]\s|>|\||<|(?:[-*_]\s*){3,}$|=+\s*$)/; // heading, list item, quote, table, HTML, rule, underline
const ENDS_LINE = /^(#{1,6}(\s|$)|\||(?:[-*_]\s*){3,}$|=+\s*$)|( {2}|\\)$/; // heading, table row, rule, underline, hard break

// Indices i of the lines whose break to line i + 1 shows as a space.
export function softBreaks(lines) {
  const out = [];
  let fence = false;
  for (let i = 0; i < lines.length - 1; i++) {
    const line = lines[i];
    if (FENCE.test(line)) {
      fence = !fence;
      continue;
    }
    const next = lines[i + 1];
    if (fence || !line.trim() || !next.trim() || ENDS_LINE.test(line)) continue;
    if (/^\s/.test(next) || STARTS_BLOCK.test(next) || FENCE.test(next)) continue; // indented: verse, code, list body
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
