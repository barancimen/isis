"""Download a few public-domain books from Project Gutenberg and split them into one markdown
file per chapter, to have a corpus to try isis on.

    python scripts/fetch_sample_corpus.py [target_dir]     # default: ./sample_corpus

The books overlap in theme (the sea, voyages, natural history), which makes the "bridges"
between parts of a task document easier to see.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

BOOKS = {
    "moby-dick": (2701, r"^CHAPTER \d+\..*$"),
    "origin-of-species": (1228, r"^CHAPTER [IVXLC]+\..*$"),
    "voyage-of-the-beagle": (944, r"^CHAPTER [IVXLC]+\b.*$"),
}


def fetch(book_id: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    text = text.replace("\r\n", "\n")
    start = re.search(r"^\*\*\* ?START OF.*$", text, re.M)
    end = re.search(r"^\*\*\* ?END OF.*$", text, re.M)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def chapters(text: str, pattern: str) -> list[tuple[str, str]]:
    heads = list(re.finditer(pattern, text, re.M))
    if not heads:
        return [("Full text", text)]
    # The table of contents repeats the chapter headings; the body starts at the last
    # occurrence of the first chapter number.
    number = lambda m: m.group().split(".")[0].split()[:2]
    first = number(heads[0])
    body_start = max(m.start() for m in heads if number(m) == first)
    heads = [m for m in heads if m.start() >= body_start]

    out = []
    for i, m in enumerate(heads):
        body = text[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)].strip("\n")
        title = m.group().strip()
        if title.rstrip(".").count(" ") <= 1:  # "CHAPTER IV." with the title on the next line
            lines = body.split("\n")
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines:
                title = f"{title} {lines.pop(0).strip()}"
            body = "\n".join(lines).strip("\n")
        out.append((title, body))
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_corpus")
    for name, (book_id, pattern) in BOOKS.items():
        folder = target / name
        folder.mkdir(parents=True, exist_ok=True)
        parts = chapters(fetch(book_id), pattern)
        for n, (title, body) in enumerate(parts, 1):
            (folder / f"{n:03d}-{slug(title)}.md").write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        print(f"{name}: {len(parts)} chapters")


if __name__ == "__main__":
    main()
