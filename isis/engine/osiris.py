"""Archives made by osiris (https://github.com/barancimen/osiris): where each document came from.

osiris saves every document as md/<id[:2]>/<id>.md, with YAML frontmatter whose `source:` says
how its text was got (`youtube:transcript:auto`, `pdf:wayback`, `hackernews`...), and records in
manifest.sqlite which lane fetched each URL. A document's kind is its lane, split further by its
source where that tells documents of one lane apart, such as PDFs from web pages, or videos with
human captions from those with automatic ones. How a text was fetched (directly, from the Wayback
Machine, with a browser) is not a kind: the same page can come either way.

A source id is "<folder>/<kind>", such as "md/youtube:human", where the folder is the archive's
name among the corpus folders (empty when it is served alone). A request may also leave out a
whole lane by its name, such as "md/youtube"."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "manifest.sqlite"
OTHER = "other"

# Lanes in the order osiris lists them, with its names for them.
LANES = {
    "generic": "Web pages and PDFs",
    "localfile": "Local files",
    "youtube": "YouTube",
    "github": "GitHub",
    "reddit": "Reddit",
    "hackernews": "Hacker News",
    "lesswrong": "LessWrong",
    "stackexchange": "Stack Exchange",
    "arxiv": "arXiv",
    "twitter": "X (Twitter)",
    "singlefile": "Saved pages",
    "chat": "AI chats",
}

# The kinds of each lane that has more than one: (kind, name, the sources that are of it). A source
# matches an entry that is equal to it or starts it followed by ":". The first match wins.
KINDS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "generic": [
        ("web", "Web pages", ("html", "scry:crawl")),
        ("pdf", "PDFs", ("pdf",)),
        ("text", "Plain text", ("text",)),
        ("doi", "Paper abstracts", ("doi",)),
    ],
    "localfile": [
        ("pdf", "PDFs", ("file:pdf",)),
        ("epub", "EPUB books", ("file:epub",)),
        ("kindle", "Kindle books", ("file:kindle",)),
        ("docx", "Word documents", ("file:docx",)),
        ("html", "Web pages", ("file:html",)),
        ("text", "Text files", ("file:text",)),
    ],
    "youtube": [
        ("human", "Videos with human captions", ("youtube:transcript:human",)),
        ("auto", "Videos with automatic captions", ("youtube:transcript:auto",)),
        ("nocaptions", "Videos without captions", ("youtube:metadata",)),
        ("playlist", "Playlists", ("youtube:playlist",)),
        ("channel", "Channels", ("youtube:channel",)),
    ],
    "github": [
        ("repo", "Repositories", ("github:api",)),
        ("gist", "Gists", ("github:gist",)),
    ],
    "arxiv": [
        ("abstract", "Abstracts only", ("arxiv_abstract",)),
        ("full", "Full text", ("pdf:arxiv", "html:arxiv", "fulltext", "pdf")),
    ],
    "singlefile": [
        ("singlefile", "SingleFile pages", ("singlefile",)),
        ("html", "Other saved pages", ("html_file",)),
    ],
    # Exports from the chat sites and the extension, and the logs of coding agents (Claude Code's prompt
    # history among them), which osiris saves in one lane.
    "chat": [
        ("chatbot", "Chatbot conversations", ("chat:claude", "chat:chatgpt", "chat:deepseek")),
        ("session", "Coding agent sessions", ("chat:claude-code", "chat:omp", "chat:pi")),
    ],
}

# The lane of a document the manifest doesn't name, from its source alone. A lane's own fetch has a
# source of its own; a page fetched again another way (from the Wayback Machine, say) looks like a
# web page, which only the manifest knows better.
SOURCE_LANES = (
    ("youtube", "youtube"), ("hackernews", "hackernews"), ("lesswrong", "lesswrong"), ("reddit", "reddit"),
    ("stackexchange", "stackexchange"), ("tweet", "twitter"), ("github", "github"), ("arxiv_abstract", "arxiv"),
    ("pdf:arxiv", "arxiv"), ("html:arxiv", "arxiv"), ("fulltext:scry", "arxiv"), ("file", "localfile"),
    ("singlefile", "singlefile"), ("html_file", "singlefile"), ("chat", "chat"),
)


def _starts(source: str, prefix: str) -> bool:
    return source == prefix or source.startswith(f"{prefix}:")


def lane_of_source(source: str) -> str:
    return next((lane for prefix, lane in SOURCE_LANES if _starts(source, prefix)), "generic")


def kind_of(lane: str, source: str) -> str:
    """The kind of a document: its lane, then the part of it that its source points to, as in
    "youtube:auto". A lane with one kind is just its name."""
    kinds = KINDS.get(lane)
    if not kinds:
        return lane
    for kind, _, sources in kinds:
        if any(_starts(source, s) for s in sources):
            return f"{lane}:{kind}"
    return f"{lane}:{OTHER}"


def describe(kind: str) -> tuple[str, str, str]:
    """(lane, lane name, kind name) of a kind, for the page."""
    lane, _, part = kind.partition(":")
    lane_name = LANES.get(lane, lane)
    if not part:
        return lane, lane_name, lane_name
    name = next((n for k, n, _ in KINDS.get(lane, []) if k == part), "Other")
    return lane, lane_name, name


def order(kind: str) -> tuple[int, int]:
    """Where a kind is listed: by lane as osiris lists them, then as in KINDS."""
    lane, _, part = kind.partition(":")
    lanes = list(LANES)
    parts = [k for k, _, _ in KINDS.get(lane, [])]
    return (lanes.index(lane) if lane in lanes else len(lanes),
            parts.index(part) if part in parts else len(parts))


def archive_folders(path: Path) -> tuple[Path, Path | None]:
    """The folder of documents and the manifest of an osiris archive, given either the archive's
    folder (osiris's data/) or its md/ folder. The manifest is None if there is none."""
    path = path.resolve()
    if (path / "md").is_dir():
        md, manifest = path / "md", path / MANIFEST
    else:
        md, manifest = path, path.parent / MANIFEST
    return md, manifest if manifest.is_file() else None


@dataclass
class Manifest:
    """Which lane fetched each document of an archive, from osiris's manifest.sqlite, read again
    whenever osiris has written to it. Without one, each lane is worked out from the source."""

    path: Path | None
    lanes: dict[str, str]
    _stamp: tuple = ()
    _failed: str = ""

    @classmethod
    def at(cls, path: Path | None) -> Manifest:
        manifest = cls(path, {})
        manifest.refresh()
        return manifest

    def _changed(self) -> tuple:
        stamp = []
        for suffix in ("", "-wal"):
            try:
                st = os.stat(f"{self.path}{suffix}")
                stamp.append((st.st_mtime_ns, st.st_size))
            except OSError:
                stamp.append(None)
        return tuple(stamp)

    def refresh(self) -> bool:
        """Read the manifest again if it has changed. Returns whether it did."""
        if self.path is None:
            return False
        stamp = self._changed()
        if stamp == self._stamp:
            return False
        try:
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=5)) as db:
                rows = db.execute("SELECT doc_id, lane FROM urls WHERE doc_id IS NOT NULL AND lane != 'skip'").fetchall()
        except sqlite3.Error as e:
            if str(e) != self._failed:
                self._failed = str(e)
                log.warning("could not read %s (%s): the kind of each document comes from its source instead", self.path, e)
            return False
        self._failed = ""
        self._stamp = stamp
        lanes = {}
        for doc_id, lane in rows:
            lanes.setdefault(doc_id, lane)
        changed = lanes != self.lanes
        self.lanes = lanes
        return changed

    def kind(self, path: str, source: str) -> str:
        """The kind of the document at this corpus path, whose frontmatter has this source."""
        doc_id = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return kind_of(self.lanes.get(doc_id) or lane_of_source(source), source)
