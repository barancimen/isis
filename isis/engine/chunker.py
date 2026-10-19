"""Split markdown into passages: the unit that search returns, that gets embedded, and that the
"related" pipeline compares against a task document."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
HAS_WORD = re.compile(r"[^\W_]")

TARGET_WORDS = 100  # short neighbouring paragraphs under one heading are merged up to this size
MAX_WORDS = 180  # longer runs are cut at line boundaries, so most passages fit in 256 model tokens


@dataclass(frozen=True)
class Passage:
    path: str
    heading: str
    start_line: int  # 0-based, inclusive
    end_line: int  # 0-based, inclusive
    text: str

    @property
    def embed_text(self) -> str:
        """What the embedding model sees: the heading gives a short passage its context."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def split_lines(content: str) -> list[str]:
    """Split lines the way CodeMirror does, so line numbers agree with the editor."""
    return re.split(r"\r\n|\r|\n", content)


def _blocks(lines: list[str]):
    """Yield (section, heading, start, end) for each run of non-blank lines.

    Headings and blank lines end a run, except inside code fences. YAML frontmatter is skipped.
    `section` counts headings, so two sections with the same title stay apart.
    """
    start = 0
    if lines and lines[0].strip() == "---":
        for n in range(1, len(lines)):
            if lines[n].strip() in ("---", "..."):
                start = n + 1
                break

    section, heading, run, in_fence = 0, "", None, False
    for n in range(start, len(lines)):
        line = lines[n]
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            m = HEADING.match(line)
            if m or not line.strip():
                if run is not None:
                    yield section, heading, run, n - 1
                    run = None
                if m:
                    section, heading = section + 1, m.group(1)
                continue
        if run is None:
            run = n
    if run is not None:
        yield section, heading, run, len(lines) - 1


def _cut(lines: list[str], words: list[int], start: int, end: int, max_words: int):
    """Cut lines [start, end] into pieces of at most max_words, trimming blank lines at the edges."""
    pieces, s, total = [], start, 0
    for n in range(start, end + 1):
        if total and total + words[n] > max_words:
            pieces.append((s, n - 1))
            s, total = n, 0
        total += words[n]
    pieces.append((s, end))

    for s, e in pieces:
        while s < e and not lines[s].strip():
            s += 1
        while e > s and not lines[e].strip():
            e -= 1
        yield s, e


def chunk_markdown(
    path: str, content: str, target_words: int = TARGET_WORDS, max_words: int = MAX_WORDS
) -> list[Passage]:
    """Split a markdown file into passages with 0-based line ranges.

    Paragraphs under the same heading are merged while the total stays within `target_words`
    (pass 0 to keep every paragraph separate), and anything over `max_words` is cut.
    """
    lines = split_lines(content)
    words = [len(line.split()) for line in lines]

    groups: list[list] = []  # [section, heading, start, end, words]
    for section, heading, start, end in _blocks(lines):
        if not any(HAS_WORD.search(line) for line in lines[start : end + 1]):
            continue  # rules, table borders and the like
        size = sum(words[start : end + 1])
        last = groups[-1] if groups else None
        if last and last[0] == section and last[4] + size <= target_words:
            last[3], last[4] = end, last[4] + size
        else:
            groups.append([section, heading, start, end, size])

    passages = []
    for _, heading, start, end, _ in groups:
        for s, e in _cut(lines, words, start, end, max_words):
            text = "\n".join(lines[s : e + 1])
            if HAS_WORD.search(text):
                passages.append(Passage(path, heading, s, e, text))
    return passages
