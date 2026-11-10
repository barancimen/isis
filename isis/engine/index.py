"""The corpus index: passages, BM25 and embeddings for every markdown file, kept up to date
incrementally and cached on disk.

Several embedding models can be in use at once: each passage has a vector from each, and a search
names the model it wants. The first model is the default.

The cache is plain data, JSON for the text and .npz files of arrays read with allow_pickle=False, so
a cache that came with someone else's corpus can't run code when it is loaded. Each model's vectors
have a file of their own in .isis/models/, found again by each passage's text: a model left out of
a run keeps them, and an edit made meanwhile only costs the passages it changed."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat as statmode
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .chunker import MAX_FRONT_MATTER, MAX_WORDS, TARGET_WORDS, Passage, chunk_markdown, document_site, split_lines
from .lexical import B, BM25, K1, Vocabulary, term_counts
from .text import STOPWORDS, WORD, tokenize_with_surface
from .typos import WordView, Words

log = logging.getLogger(__name__)

INDEX_VERSION = 2
FRONT_MATTER_CHARS = 65536  # enough to hold the frontmatter of any file
CHUNKER_VERSION = 2  # raise it when files split differently: they are read again, and keep their embeddings
TERMS_VERSION = 2  # raise it when text becomes terms differently: cached passages are split into terms again
# (1: Snowball stems; 2: lowercase words)
MARKDOWN = (".md", ".markdown")
NO_EMBED = ".no-embed"  # a file by this name: this folder and those below it are searched by keywords only
EMBED_GROUP = 256  # passages per embedding call while indexing
PUBLISH_EVERY = 20.0  # seconds between snapshot updates while a large corpus is being embedded
SAVE_EVERY = 120.0  # seconds between cache writes during a long embedding run
REPORT_EVERY = 10.0  # seconds between progress messages during a long embedding run
SAMPLE_ABOVE = 50_000  # embedded passages; above this, "typical" similarity comes from a sample
TYPICAL_SAMPLE = 20_000


def configure_logging() -> None:
    """What `isis` prints: its own progress messages, and only warnings from other libraries."""
    logging.basicConfig(level=logging.WARNING, format="isis: %(message)s")
    logging.getLogger("isis").setLevel(logging.INFO)


def corpus_exclude(folders: list[Path], tasks_dir: Path) -> tuple[Path, ...]:
    """Task folders stay out of the index: the one in use, and the default place in each other
    corpus folder (which holds its tasks when it is served on its own)."""
    return (tasks_dir, *(Path(d) / "_tasks" for d in folders[1:]))


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s" if seconds < 10 else f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _text_key(text: str) -> int:
    """A passage's text as a number, to find its vector among the embeddings kept for another model."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def _changes(new: list[str], modified: list[str], removed: list[str]) -> str:
    """'100 new files, 2 changed, 1 removed', naming the files when there are only a few."""
    parts = []
    for paths, label in ((new, _count(len(new), "new file")), (modified, f"{len(modified):,} changed"),
                         (removed, f"{len(removed):,} removed")):
        if paths:
            names = f" ({', '.join(sorted(paths))})" if len(new) + len(modified) + len(removed) <= 5 else ""
            parts.append(label + names)
    return ", ".join(parts)


def _model_id(embedder) -> str:
    """How a request names a model, such as "MongoDB/mdbr-leaf-ir"."""
    return getattr(embedder, "model_id", embedder.name)


@dataclass
class FileRecord:
    mtime_ns: int
    size: int
    passages: list[Passage]
    indptr: np.ndarray  # this file's rows of the BM25 matrix, as CSR pieces
    term_ids: np.ndarray
    counts: np.ndarray
    vectors: dict[str, np.ndarray] = field(default_factory=dict)  # model name -> float16, one row per passage
    reuse: dict[str, dict[str, np.ndarray]] | None = None  # model name -> embed_text -> vector kept from before; not saved
    site: str | None = None  # the host of `url:` in the frontmatter; None if not known yet
    _keys: np.ndarray | None = field(default=None, repr=False)

    def keys(self) -> np.ndarray:
        """Each passage's text as a number: how the vectors of every model are saved."""
        if self._keys is None:
            self._keys = np.fromiter((_text_key(p.embed_text) for p in self.passages), np.uint64, len(self.passages))
        return self._keys


@dataclass
class Snapshot:
    """An immutable view of the index. A search uses one snapshot throughout, so a refresh
    running in another thread never changes data under it."""

    passages: list[Passage]
    bm25: BM25
    vocab: Vocabulary  # shared with the engine and only ever grows: look terms up with term_id()
    sorted_terms: list[str]  # terms that occur in the corpus, for prefix expansion
    words: WordView  # every word as written, with its term, for typo-tolerant search
    vectors: np.ndarray  # float32 [n, dim] from one model; zero rows where has_vector is False
    has_vector: np.ndarray  # bool [n]
    file_rows: dict[str, tuple[int, int]]  # path -> [start, end) rows
    file_sites: dict[str, str] = field(default_factory=dict)  # path -> host, for documents that have one
    _bm25_variants: dict = field(default_factory=dict, repr=False)
    _pattern_counts: dict = field(default_factory=dict, repr=False)
    _folder_rows: dict = field(default_factory=dict, repr=False)
    near_words: dict = field(default_factory=dict, repr=False)  # for search.near_words
    excerpt_matches: dict = field(default_factory=dict, repr=False)  # for related.find_excerpts
    theme_scores: dict = field(default_factory=dict, repr=False)  # for related.theme_scores
    _typical: dict = field(default_factory=dict, repr=False)
    _typical_vectors: dict = field(default_factory=dict, repr=False)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # requests share snapshots
    # The same index seen through each model, by model id. They share everything but the vectors
    # and what is worked out from them.
    views: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.passages)

    def bm25_for(self, k1: float, b: float) -> BM25:
        """BM25 with other parameters, built on first use; the last few are kept."""
        key = (round(float(k1), 4), round(float(b), 4))
        if key == (self.bm25.k1, self.bm25.b):
            return self.bm25
        return self.cached(self._bm25_variants, key, 4, lambda: self.bm25.with_params(*key))

    def pattern_counts(self, pattern: re.Pattern, rows: np.ndarray | None = None) -> np.ndarray:
        """How often `pattern` matches in each passage, worked out for `rows` (default: every
        passage; the others may read 0). Counts are kept per passage for the last few patterns, so
        a search in one folder scans only that folder, ticking another later scans only that one,
        and a regex in a steer line is not rescanned on every refresh."""
        n = len(self.passages)
        counts, done = self.cached(self._pattern_counts, (pattern.pattern, pattern.flags), 32,
                                   lambda: (np.zeros(n, np.float32), np.zeros(n, bool)))
        todo = np.flatnonzero(~done) if rows is None else rows[~done[rows]]
        if len(todo):
            passages = self.passages  # a pattern that runs out of time raises before anything is marked done
            counts[todo] = np.fromiter((sum(1 for m in pattern.finditer(passages[i].embed_text) if m.end() > m.start())
                                        for i in todo.tolist()), np.float32, len(todo))
            done[todo] = True
        return counts

    def cached(self, cache: dict, key, size: int, make: Callable):
        """cache[key], made on first use; the oldest entries go when there are `size`."""
        with self._cache_lock:
            value = cache.get(key)
        if value is None:
            value = make()  # outside the lock: a slow regex scan doesn't hold up other requests
            with self._cache_lock:
                while len(cache) >= size:
                    cache.pop(next(iter(cache)))
                cache[key] = value
        return value

    def typical_rows(self) -> np.ndarray:
        """The passages that "typical" similarity is worked out from: all the embedded ones, or
        in a large corpus a fixed sample, which estimates a median about as well for far less."""

        def choose():
            rows = np.flatnonzero(self.has_vector)
            if len(rows) <= SAMPLE_ABOVE:
                return rows
            return np.sort(np.random.default_rng(0).choice(rows, TYPICAL_SAMPLE, replace=False))

        return self.cached(self._typical, "rows", 1, choose)

    def typical_vectors(self) -> np.ndarray:
        """The vectors of typical_rows, kept, for comparing a few folders without the others."""
        return self.cached(self._typical_vectors, "vectors", 1, lambda: self.vectors[self.typical_rows()])

    def folder_rows(self) -> dict[str, tuple[int, int]]:
        """The rows of each corpus folder (the first part of a path): one block each, as passages
        are in the order of their paths."""

        def spans() -> dict[str, tuple[int, int]]:
            out: dict[str, tuple[int, int]] = {}
            for path, (start, end) in self.file_rows.items():
                name = path.partition("/")[0]
                out[name] = (out[name][0], end) if name in out else (start, end)
            return out

        return self.cached(self._folder_rows, "spans", 1, spans)

    def outside_folders(self, skip) -> np.ndarray:
        """Which passages are not in these corpus folders."""
        ok = np.ones(len(self.passages), bool)
        for name in skip:
            if (span := self.folder_rows().get(name)) is not None:
                ok[span[0]:span[1]] = False
        return ok

    def at_sites(self, sites) -> np.ndarray:
        """Which passages are in documents from these hosts, or from their subdomains."""
        ok = np.zeros(len(self.passages), bool)
        for path, host in self.file_sites.items():
            if any(host == site or host.endswith(f".{site}") for site in sites):
                start, end = self.file_rows[path]
                ok[start:end] = True
        return ok

    def term_id(self, term: str) -> int | None:
        i = self.vocab.get(term)
        return i if i is not None and i < self.bm25.n_terms else None


@dataclass
class IndexStatus:
    state: str = "starting"  # "scanning", "embedding", "ready" or "error"
    error: str = ""  # why the last refresh failed
    files: int = 0
    passages: int = 0
    embedded: int = 0  # progress of the current embedding run
    to_embed: int = 0
    unembedded: int = 0  # passages without embeddings
    updated: float = 0.0  # time.time() when the snapshot last changed


class Engine:
    """Owns the index of one or more corpus folders. `refresh()` brings it up to date with the
    files on disk; `snapshot` is what searches read. With several folders, a corpus path starts
    with its folder's name, and each folder keeps its own cache, so it can be served alone later."""

    def __init__(
        self,
        corpus_dir: str | Path | list[str | Path],
        *,
        embedder=None,  # an embedder, or a list of them; the first is the default
        index_dir: str | Path | None = None,
        exclude: tuple[str | Path, ...] = (),
        target_words: int = TARGET_WORDS,
        max_words: int = MAX_WORDS,
    ):
        dirs = list(dict.fromkeys(Path(d).resolve() for d in (corpus_dir if isinstance(corpus_dir, (list, tuple)) else [corpus_dir])))
        self.root = dirs[0]
        self.roots: dict[str, Path] = {"": dirs[0]} if len(dirs) == 1 else self._named(dirs)  # path prefix -> folder
        self.index_dir = Path(index_dir).resolve() if index_dir else None
        self.embedders = [] if embedder is None else list(embedder) if isinstance(embedder, (list, tuple)) else [embedder]
        self.exclude = {Path(p).resolve() for p in exclude}
        self.chunking = (target_words, max_words)
        self.vocab = Vocabulary()
        self.words = Words()
        self._cache_unsaved = False  # words listed, or terms made again, from a cache loaded the old way
        self.files: dict[str, FileRecord] = {}
        self.status = IndexStatus()
        self._lock = threading.Lock()
        self._chunker_from: dict[str, int] = {}  # folders whose cache still has files split the old way
        self._unread: list[str] = []
        self.no_embed: set[str] = set()  # corpus-relative folders with a .no-embed file ("" is the whole corpus)
        self.links: set[str] = set()  # corpus-relative paths of the folder links the last scan met
        self._said: set[str] = set()  # one-time messages about links, so rescans stay quiet
        self._refreshed = False
        self._load()
        self._publish()

    @staticmethod
    def _named(dirs: list[Path]) -> dict[str, Path]:
        """Each folder by its name, numbered when two share one."""
        named: dict[str, Path] = {}
        for d in dirs:
            name, n = d.name or "root", 1
            while name in named:
                n += 1
                name = f"{d.name}-{n}"
            named[name] = d
        return named

    @property
    def embedder(self):
        """The default model; None when searching by keywords only."""
        return self.embedders[0] if self.embedders else None

    def embedder_for(self, model: str | None):
        """The embedder with this model id, or the default one for None or an id it doesn't know."""
        return next((e for e in self.embedders if _model_id(e) == model), self.embedder)

    # ---- files ----

    def resolve(self, path: str) -> Path:
        """The file for a corpus-relative path. Refuses a path that climbs out of the corpus.
        Links inside the corpus are followed wherever they point (that is how a folder from
        elsewhere joins it), as long as what they reach is markdown too."""
        rel = Path(path)
        full = self._located(path) if not rel.is_absolute() and ".." not in rel.parts else None
        if full is None or full.suffix.lower() not in MARKDOWN or full.resolve().suffix.lower() not in MARKDOWN:
            raise ValueError(f"not a markdown file in the corpus: {path}")
        return full

    def _located(self, path: str) -> Path | None:
        """Where a corpus path is on disk; None if it starts with no folder's name."""
        if "" in self.roots:
            return self.root / path
        head, _, rest = path.partition("/")
        return self.roots[head] / rest if head in self.roots and rest else None

    def read_lines(self, path: str) -> list[str]:
        return split_lines(self.resolve(path).read_text(encoding="utf-8", errors="replace"))

    def _say_once(self, level: int, message: str) -> None:
        if message not in self._said:
            self._said.add(message)
            log.log(level, message)

    def _scan(self) -> dict[str, os.stat_result]:
        """Every markdown file under the corpus folders, following symlinks. The real folders come
        first and links after them, and each folder or file is taken once (by device and inode), so
        a link loop ends and something reachable both ways keeps its real path. Folders it could
        not read go in self._unread."""
        found: dict[str, os.stat_result] = {}
        unread: list[str] = []
        folders, files = set(), set()
        folder_links: list[tuple[Path, Path, str]] = []
        file_links: list[tuple[Path, str]] = []
        no_embed: set[str] = set()

        def label(path: Path, root: Path, prefix: str) -> str:
            rel = path.relative_to(root).as_posix()
            rel = "" if rel == "." else rel
            return f"{prefix}/{rel}".rstrip("/") if prefix else rel

        def excluded(path: Path) -> bool:
            real = path.resolve()
            return real in self.exclude or any(parent in self.exclude for parent in real.parents)

        def first_visit(path: Path) -> bool:
            st = path.stat()
            key = (st.st_dev, st.st_ino)
            if key in folders:
                return False
            folders.add(key)
            return True

        def walk(top: Path, root: Path, prefix: str) -> None:
            def failed(e: OSError) -> None:
                where = label(Path(e.filename), root, prefix) if e.filename else label(top, root, prefix)
                unread.append(where)
                self._say_once(logging.WARNING, f"could not read {where or 'the corpus'}: {e.strerror or e}; its files stay in the index")

            for dirpath, dirnames, filenames in os.walk(top, onerror=failed):
                here = Path(dirpath)
                keep = []
                for d in sorted(dirnames):
                    path = here / d
                    if d.startswith(".") or path.resolve() in self.exclude:
                        continue
                    try:
                        if path.is_symlink():
                            folder_links.append((path, root, prefix))
                        elif first_visit(path):
                            keep.append(d)
                    except OSError:  # deleted since os.walk listed it
                        pass
                dirnames[:] = keep
                for name in filenames:
                    path = here / name
                    if name == NO_EMBED:
                        no_embed.add(label(here, root, prefix))
                    if name.startswith("."):
                        continue
                    if not name.lower().endswith(MARKDOWN):
                        if path.is_symlink() and not path.exists():  # a linked folder that has gone away
                            self._say_once(logging.WARNING, f"link {label(path, root, prefix)} skipped: its target doesn't exist")
                        continue
                    try:
                        st = path.lstat()
                    except OSError:
                        continue
                    if statmode.S_ISLNK(st.st_mode):
                        file_links.append((path, label(path, root, prefix)))
                    else:
                        found[label(path, root, prefix)] = st
                        files.add((st.st_dev, st.st_ino))

        for prefix, root in self.roots.items():
            try:
                if first_visit(root):
                    walk(root, root, prefix)
                else:
                    self._say_once(logging.INFO, f"{root} skipped: it is already inside another folder being scanned")
            except OSError as e:
                unread.append(label(root, root, prefix))
                self._say_once(logging.WARNING, f"{root} skipped: {e.strerror or e}")
        links = set()
        while folder_links:  # walking a linked folder can find more links
            link, root, prefix = folder_links.pop(0)
            rel = label(link, root, prefix)
            links.add(rel)
            try:
                if excluded(link):
                    continue
                if not first_visit(link):
                    self._say_once(logging.INFO, f"link {rel} skipped: its folder is already indexed")
                    continue
            except OSError as e:
                unread.append(rel)
                self._say_once(logging.WARNING, f"link {rel} skipped: {e.strerror or e}")
                continue
            self._say_once(logging.INFO, f"following link {rel} -> {link.resolve()}")
            walk(link, root, prefix)
        self.links = links
        for link, rel in file_links:
            try:
                st = link.stat()
            except OSError as e:
                self._say_once(logging.WARNING, f"link {rel} skipped: {e.strerror or e}")
                continue
            if link.resolve().suffix.lower() not in MARKDOWN or excluded(link):
                self._say_once(logging.INFO, f"link {rel} skipped: it doesn't point to a markdown file of the corpus")
            elif (st.st_dev, st.st_ino) not in files:
                found[rel] = st
                files.add((st.st_dev, st.st_ino))
        self.no_embed = no_embed
        self._unread = unread
        return found

    def _is_unread(self, path: str) -> bool:
        """Whether this file is under a folder the last scan could not read. It may only be away
        for now (an unmounted drive, changed permissions), and embeddings can take hours to redo."""
        return any(not d or path == d or path.startswith(f"{d}/") for d in self._unread)

    def _embeds(self, path: str) -> bool:
        """Whether this file gets embeddings: not under a folder with a .no-embed file."""
        return not any(not d or path.startswith(f"{d}/") for d in self.no_embed)

    def _read(self, path: str, stat: os.stat_result) -> FileRecord:
        text = self._located(path).read_text(encoding="utf-8", errors="replace")
        passages = chunk_markdown(path, text, *self.chunking)
        site = document_site(split_lines(text)[:MAX_FRONT_MATTER])
        return FileRecord(stat.st_mtime_ns, stat.st_size, passages, *self._terms_of(passages), site=site)

    def _terms_of(self, passages: list[Passage]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The BM25 rows (CSR pieces) of these passages. Their words join the word list."""
        pairs = [tokenize_with_surface(p.heading) + tokenize_with_surface(p.text) for p in passages]
        rows = term_counts([[t for t, _ in ps] for ps in pairs], self.vocab)
        for term, word in {pair for ps in pairs for pair in ps}:
            self.words.add(word, self.vocab.ids[term])
        return rows

    def _site_on_disk(self, path: str) -> str | None:
        """The site of a file in a cache from before sites were kept, without reading all of it."""
        try:
            with self._located(path).open(encoding="utf-8", errors="replace") as f:
                return document_site(split_lines(f.read(FRONT_MATTER_CHARS))[:MAX_FRONT_MATTER])
        except (OSError, AttributeError):  # gone, or its folder is away: known next time
            return None

    # ---- refresh ----

    def refresh(self, on_progress: Callable[[IndexStatus], None] | None = None) -> dict:
        """Re-read changed files, drop deleted ones, embed what isn't embedded yet."""
        with self._lock:
            self.status.error = ""
            try:
                return self._refresh(on_progress)
            except Exception as e:
                self.status.state, self.status.error = "error", f"{type(e).__name__}: {e}"
                raise

    def _refresh(self, on_progress) -> dict:
        started = time.perf_counter()
        first = not self._refreshed
        self._refreshed = True
        self._set_state("scanning", on_progress)
        if first:
            log.info("scanning %s", ", ".join(map(str, self.roots.values())))
        found = self._scan()
        removed = [p for p in self.files if p not in found and not self._is_unread(p)]
        changed = [
            p
            for p, st in found.items()
            if (r := self.files.get(p)) is None or (r.mtime_ns, r.size) != (st.st_mtime_ns, st.st_size)
        ]
        new = [p for p in changed if p not in self.files]
        modified = [p for p in changed if p in self.files]
        if changed or removed:
            total = f"{_count(len(found), 'markdown file')}: " if first else ""
            log.info("%s%s", total, _changes(new, modified, removed))
        elif first:
            log.info("%s, no changes since the last run" if found else "no markdown files found in %s",
                     _count(len(found), "markdown file") if found else ", ".join(map(str, self.roots.values())))
        for path in removed:
            del self.files[path]
        for path in changed:
            try:
                record = self._read(path, found[path])
            except OSError as e:  # deleted or unreadable since the scan
                log.warning("skipping %s: %s", path, e)
                self.files.pop(path, None)
                continue
            old = self.files.get(path)
            if old is not None:  # an edit changes a few passages: the others keep their vectors, for every model
                reuse = {name: dict(zip((p.embed_text for p in old.passages), vectors)) for name, vectors in old.vectors.items()}
                for name, kept in (old.reuse or {}).items():  # vectors loaded for it, not yet taken up
                    reuse[name] = {**kept, **reuse.get(name, {})}
                record.reuse = reuse or None
            self.files[path] = record
        for folder in sorted(self.no_embed):
            where = folder or "the corpus"
            self._say_once(logging.INFO, f"keywords only in {where}: it has a {NO_EMBED} file")
        dropped = 0
        for path, record in self.files.items():
            if not self._embeds(path):
                record.reuse = None
                if record.vectors:  # the folder got its .no-embed file after it was embedded
                    record.vectors = {}
                    dropped += 1
        touched = bool(removed or changed or dropped)
        if touched:
            self._publish()

        embedded, filled = self._embed_missing(on_progress)
        if touched or filled or self._cache_unsaved:
            self._publish()
            self._save()
            self._cache_unsaved = False
        self._set_state("ready", on_progress)
        if first or touched or filled:
            meaning = ", keywords only" if self.embedder is None else (
                f", {self.status.unembedded:,} of them keywords only" if self.status.unembedded else "")
            log.info("index ready: %s, %s%s (%s)", _count(len(self.files), "file"), _count(len(self.snapshot), "passage"),
                     meaning, _duration(time.perf_counter() - started))
        return {
            "files": len(self.files),
            "new": len(new),
            "changed": len(changed),
            "removed": len(removed),
            "embedded": embedded,
            "seconds": round(time.perf_counter() - started, 2),
        }

    def _embed_missing(self, on_progress) -> tuple[int, int]:
        """Embeds, with each model in turn, the passages that have no vector from it yet. Returns
        how many passages it embedded, and how many files got vectors (some may have taken all of
        theirs from the ones kept for that model)."""
        work = []
        for embedder in self.embedders:
            name = embedder.name
            pending = [r for path, r in self.files.items()
                       if self._embeds(path) and not self._is_unread(path) and name not in r.vectors]
            todo = {id(r): [p.embed_text for p in r.passages if p.embed_text not in (r.reuse or {}).get(name, {})]
                    for r in pending}
            work.append((embedder, pending, todo))
        self.status.embedded = 0
        self.status.to_embed = sum(len(texts) for _, _, todo in work for texts in todo.values())
        if self.status.to_embed:
            self._set_state("embedding", on_progress)
        begun = last_report = last_publish = last_save = time.monotonic()

        for embedder, pending, todo in work:
            name, dim = embedder.name, embedder.dim
            n = sum(map(len, todo.values()))
            if n:
                kept = sum(len(r.passages) for r in pending) - n
                log.info("embedding %s with %s%s", _count(n, "passage"), _model_id(embedder),
                         f" ({kept:,} unchanged kept their embeddings)" if kept else "")
            group: list[FileRecord] = []

            def flush():
                texts = [t for r in group for t in todo[id(r)]]
                fresh = iter(embedder.embed(texts) if texts else np.zeros((0, dim), np.float32))
                for r in group:
                    kept = (r.reuse or {}).pop(name, {})
                    rows = [kept[p.embed_text] if p.embed_text in kept else next(fresh) for p in r.passages]
                    r.vectors[name] = np.array(rows, np.float16).reshape(len(r.passages), dim)
                    r.reuse = r.reuse or None
                self.status.embedded += len(texts)
                group.clear()

            for record in pending:
                group.append(record)
                if sum(len(todo[id(r)]) for r in group) < EMBED_GROUP:
                    continue
                flush()
                if on_progress:
                    on_progress(self.status)
                now = time.monotonic()
                if now - last_report > REPORT_EVERY and self.status.embedded:
                    last_report = now
                    done, total = self.status.embedded, self.status.to_embed
                    left = (now - begun) / done * (total - done)
                    log.info("embedded %s / %s passages (%d%%), about %s left", f"{done:,}", f"{total:,}", 100 * done / total,
                             _duration(left))
                if now - last_publish > PUBLISH_EVERY:  # let searches use what is embedded so far
                    self._publish()
                    last_publish = now
                if now - last_save > SAVE_EVERY:
                    self._save()
                    last_save = now
            if group:
                flush()
        return self.status.to_embed, sum(len(pending) for _, pending, _ in work)

    def _set_state(self, state: str, on_progress) -> None:
        self.status.state = state
        if on_progress:
            on_progress(self.status)

    # ---- snapshot ----

    def _publish(self) -> None:
        self.snapshot = self._compile()
        self.status.files = len(self.files)
        self.status.passages = len(self.snapshot)
        self.status.unembedded = len(self.snapshot) - int(self.snapshot.has_vector.sum())
        self.status.updated = time.time()

    def _compile(self) -> Snapshot:
        passages: list[Passage] = []
        file_rows = {}
        indptrs, term_ids, counts = [np.zeros(1, np.int64)], [], []
        vectors = {e.name: ([], []) for e in self.embedders}  # model name -> (arrays, has)
        nnz = 0
        for path in sorted(self.files):
            r = self.files[path]
            n = len(r.passages)
            file_rows[path] = (len(passages), len(passages) + n)
            passages.extend(r.passages)
            indptrs.append(r.indptr[1:] + nnz)
            nnz += int(r.indptr[-1])
            term_ids.append(r.term_ids)
            counts.append(r.counts)
            for e in self.embedders:
                v = r.vectors.get(e.name)
                ok = v is not None and v.shape == (n, e.dim)
                vectors[e.name][0].append(v if ok else np.zeros((n, e.dim), np.float16))
                vectors[e.name][1].append(np.full(n, ok))

        def of(e) -> tuple[np.ndarray, np.ndarray]:
            if e is None:
                return np.zeros((len(passages), 0), np.float32), np.zeros(len(passages), bool)
            arrays, has = vectors[e.name]
            return (np.concatenate(arrays).astype(np.float32) if arrays else np.zeros((0, e.dim), np.float32),
                    np.concatenate(has) if has else np.zeros(0, bool))

        bm25 = BM25(
            np.concatenate(indptrs),
            np.concatenate(term_ids) if term_ids else np.zeros(0, np.int32),
            np.concatenate(counts) if counts else np.zeros(0, np.float32),
            n_terms=len(self.vocab),
            k1=K1,
            b=B,
        )
        default_vectors, default_has = of(self.embedder)
        snapshot = Snapshot(
            passages=passages,
            bm25=bm25,
            vocab=self.vocab,
            sorted_terms=sorted(self.vocab.terms[i] for i in np.flatnonzero(bm25.doc_freq)),
            words=self.words.view(),
            vectors=default_vectors,
            has_vector=default_has,
            file_rows=file_rows,
            file_sites={path: r.site for path, r in self.files.items() if r.site},
        )
        for e in self.embedders:
            if e is self.embedder:
                snapshot.views[_model_id(e)] = snapshot
            else:  # its own vectors, and its own caches of what depends on them
                v, has = of(e)
                snapshot.views[_model_id(e)] = replace(snapshot, vectors=v, has_vector=has, excerpt_matches={},
                                                       theme_scores={}, _typical={}, _typical_vectors={})
        return snapshot

    def view(self, embedder) -> Snapshot:
        """The current snapshot as this model sees it."""
        snapshot = self.snapshot
        return snapshot.views.get(_model_id(embedder), snapshot) if embedder is not None else snapshot

    # ---- cache ----

    def _cache_dir(self, prefix: str) -> Path:
        """Each corpus folder has its own cache, in its .isis folder."""
        if self.index_dir is not None:
            return self.index_dir / prefix if prefix else self.index_dir
        return self.roots[prefix] / ".isis"

    def index_bytes(self, embedder=None) -> int:
        """The size of the index on disk: the text and keywords, and this model's vectors, or every
        model's without one. This model's are counted as embedded so far, which is what its file
        will hold once saved: during a long embedding run the file is only rewritten now and then."""
        total = 0
        for prefix in self.roots:
            folder = self._cache_dir(prefix)
            vectors = [] if embedder is not None else (folder / "models").glob("*.npz")
            for path in [folder / "index.json", folder / "index.npz", *vectors]:
                with suppress(OSError):  # not saved yet, or being replaced
                    total += path.stat().st_size
        if embedder is not None:  # a float16 vector and an 8-byte text key per passage
            embedded = sum(len(r.passages) for r in list(self.files.values()) if embedder.name in r.vectors)
            total += embedded * (2 * embedder.dim + 8)
        return total

    def _vectors_path(self, prefix: str, name: str) -> Path:
        """Where a model's vectors are saved, one file per model."""
        slug = re.sub(r"[^\w.-]+", "_", name)[:150]
        return self._cache_dir(prefix) / "models" / f"{slug}.npz"

    def _write_vectors(self, prefix: str, name: str, keys: np.ndarray, vectors: np.ndarray) -> None:
        path = self._vectors_path(prefix, name)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}-{threading.get_ident()}.tmp.npz")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(tmp, name=np.array(name), keys=keys, vectors=vectors)
            os.replace(tmp, path)
        except OSError as e:
            log.warning("could not save the embeddings of %s: %s", name, e)
        finally:  # a full disk must not also keep a copy
            tmp.unlink(missing_ok=True)

    def _save_vectors(self, prefix: str, embedder, records: list[FileRecord]) -> None:
        """Saves one model's vectors for these files, with those loaded for it but not yet taken up."""
        keys, vectors = [], []
        for r in records:
            if (v := r.vectors.get(embedder.name)) is not None:
                keys.append(r.keys())
                vectors.append(v)
            elif kept := (r.reuse or {}).get(embedder.name):
                rows = [i for i, p in enumerate(r.passages) if p.embed_text in kept]
                if rows:
                    keys.append(r.keys()[rows])
                    vectors.append(np.stack([kept[r.passages[i].embed_text] for i in rows]).astype(np.float16))
        if keys:
            self._write_vectors(prefix, embedder.name, np.concatenate(keys), np.concatenate(vectors))
        elif self._vectors_path(prefix, embedder.name).exists():  # none left, as when a .no-embed file came
            self._write_vectors(prefix, embedder.name, np.zeros(0, np.uint64), np.zeros((0, embedder.dim), np.float16))

    def _load_vectors(self, prefix: str, embedder, files: dict[str, FileRecord]) -> None:
        """Gives each file its vectors from this model's saved ones. A file with some passages
        missing (changed while the model was not in use) keeps the ones found, to embed only the rest."""
        path = self._vectors_path(prefix, embedder.name)
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["name"]) != embedder.name or data["vectors"].shape[1:] != (embedder.dim,):
                    return
                keys, vectors = data["keys"], data["vectors"]
        except Exception as e:
            log.warning("ignoring unreadable embeddings in %s: %s", path, e)
            return
        row_of = dict(zip(keys.tolist(), range(len(keys))))
        for r in files.values():
            rows = [row_of.get(k) for k in r.keys().tolist()]
            if None not in rows:
                r.vectors[embedder.name] = vectors[rows]
            elif found := {p.embed_text: vectors[i] for p, i in zip(r.passages, rows) if i is not None}:
                r.reuse = {**(r.reuse or {}), embedder.name: found}

    def _save(self) -> None:
        for prefix in self.roots:
            self._save_folder(prefix)

    def _save_folder(self, prefix: str) -> None:
        folder = self._cache_dir(prefix)
        folder.mkdir(parents=True, exist_ok=True)
        lead = f"{prefix}/" if prefix else ""
        paths = sorted(p for p in self.files if p.startswith(lead))
        records = [self.files[p] for p in paths]
        term_ids = np.concatenate([r.term_ids for r in records]) if records else np.zeros(0, np.int32)
        used, local_ids = np.unique(term_ids, return_inverse=True)  # the cache names only its own terms
        words, word_terms, word_masks = self.words.of_terms(used)
        meta = {
            "version": INDEX_VERSION,
            "chunking": list(self.chunking),
            "chunker": self._chunker_from.get(prefix, CHUNKER_VERSION),
            "terms": TERMS_VERSION,
            "vocab": [self.vocab.terms[i] for i in used],
            "words": words,  # as written, each with its term in word_terms
            "files": [
                {
                    "path": path[len(lead):],
                    "mtime_ns": r.mtime_ns,
                    "size": r.size,
                    "site": r.site,
                    "passages": [[p.heading, p.start_line, p.end_line, p.text] for p in r.passages],
                }
                for path, r in zip(paths, records)
            ],
        }
        arrays = {
            "indptr": np.concatenate([r.indptr for r in records]) if records else np.zeros(0, np.int64),
            "term_ids": local_ids.astype(np.int32),
            "counts": np.concatenate([r.counts for r in records]) if records else np.zeros(0, np.float32),
            "vectors": np.zeros((0, 0), np.float16),  # each model's are in a file of their own
            "word_terms": word_terms,
            "word_masks": word_masks,  # their letters, so loading needn't work them out again
        }
        # The arrays go first: a cache is only used when its JSON matches them (see _load).
        unique = f"{os.getpid()}-{threading.get_ident()}"  # another process may share the folder
        tmp_npz, tmp_json = folder / f"index.{unique}.tmp.npz", folder / f"index.{unique}.tmp"
        try:
            np.savez(tmp_npz, **arrays)
            os.replace(tmp_npz, folder / "index.npz")
            tmp_json.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_json, folder / "index.json")
        finally:  # a full disk must not also keep a copy of the index
            tmp_npz.unlink(missing_ok=True)
            tmp_json.unlink(missing_ok=True)
        for embedder in self.embedders:
            self._save_vectors(prefix, embedder, records)

    def _load(self) -> None:
        for prefix in self.roots:
            self._load_folder(prefix)

    def _load_folder(self, prefix: str) -> None:
        folder = self._cache_dir(prefix)
        json_path, npz_path = folder / "index.json", folder / "index.npz"
        if not json_path.exists() or not npz_path.exists():
            return
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if meta.get("version") != INDEX_VERSION or tuple(meta.get("chunking", ())) != self.chunking:
                return
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in ("indptr", "term_ids", "counts", "vectors")}
                word_terms = data["word_terms"] if "word_terms" in data else None
                word_masks = data["word_masks"] if "word_masks" in data else None
            legacy = meta.get("embedder")  # a cache from before each model had a file of its own
            lead = f"{prefix}/" if prefix else ""
            if meta.get("terms", 1) != TERMS_VERSION:  # its terms are made the old way: make them again
                files, legacy_vectors = self._records(meta["files"], arrays, lead)
                self._make_terms_again(files)  # which lists the words too
            else:
                terms = list(meta["vocab"])
                ids = np.fromiter((self.vocab.add(t) for t in terms), np.int32, len(terms))  # into the shared vocabulary
                arrays["term_ids"] = ids[arrays["term_ids"]] if len(ids) else arrays["term_ids"]
                files, legacy_vectors = self._records(meta["files"], arrays, lead)
                if word_terms is not None and word_masks is not None and "words" in meta:
                    if not len(meta["words"]) == len(word_terms) == len(word_masks):
                        raise ValueError("the word list doesn't match its terms")
                    self.words.extend(meta["words"], ids[word_terms].tolist(), word_masks)
                else:
                    self._list_words(files)
        except Exception as e:
            log.warning("ignoring unreadable index cache in %s: %s", folder, e)
            return
        for path, record in files.items():
            if record.site is None:
                record.site = self._site_on_disk(path)
        self.files.update(files)
        log.info("loaded the index cache%s: %s, %s", f" of {prefix}" if prefix else "", _count(len(files), "file"),
                 _count(sum(len(r.passages) for r in files.values()), "passage"))
        if legacy and legacy_vectors and not self._vectors_path(prefix, legacy).exists():
            self._write_vectors(prefix, legacy, np.concatenate([files[path].keys() for path in legacy_vectors]),
                                np.concatenate(list(legacy_vectors.values())))
        for embedder in self.embedders:
            self._load_vectors(prefix, embedder, files)
        chunker = meta.get("chunker", 1)
        if chunker != CHUNKER_VERSION:
            if not self.embedders and (folder / "models").is_dir():
                self._chunker_from[prefix] = chunker  # reading them again now would lose their embeddings: next run
            else:
                log.info("files are split differently now: reading them again (unchanged passages keep their embeddings)")
                for record in files.values():
                    record.mtime_ns = 0  # looks changed to the next refresh

    def _make_terms_again(self, files: dict[str, FileRecord]) -> None:
        """New BM25 rows for a cache whose terms were made differently (stems, before), from the
        passages it holds: the files aren't read again and keep their embeddings."""
        n = sum(len(r.passages) for r in files.values())
        log.info("splitting %s into words again, as search no longer uses word stems (only this once)", _count(n, "passage"))
        for record in files.values():
            record.indptr, record.term_ids, record.counts = self._terms_of(record.passages)
        self._cache_unsaved = True

    def _list_words(self, files: dict[str, FileRecord]) -> None:
        """The words of a cache from before words were kept, from the passages it holds, so the
        files don't have to be read again."""
        n = sum(len(r.passages) for r in files.values())
        log.info("listing the words of %s for typo-tolerant search (only this once)", _count(n, "passage"))
        found: set[str] = set()
        for record in files.values():
            for p in record.passages:
                found.update(WORD.findall(p.heading))
                found.update(WORD.findall(p.text))
        words = sorted({w.lower() for w in found} - STOPWORDS)
        words = [w for w in words if len(w) > 1]
        found_terms = [(w, self.vocab.get(w)) for w in words]
        self.words.extend([w for w, i in found_terms if i is not None], [i for _, i in found_terms if i is not None])
        self._cache_unsaved = True

    @staticmethod
    def _records(entries: list[dict], arrays: dict, lead: str = "") -> tuple[dict[str, FileRecord], dict[str, np.ndarray]]:
        """FileRecords from a cache's JSON entries and the arrays they index into, in order, and the
        vectors an older cache kept in the same arrays, by path."""
        files, legacy = {}, {}
        at_indptr = at_nnz = at_vec = 0
        for e in entries:
            n = len(e["passages"])
            path = lead + e["path"]
            indptr = arrays["indptr"][at_indptr : at_indptr + n + 1]
            nnz = int(indptr[-1])
            if e.get("embedded"):
                legacy[path] = arrays["vectors"][at_vec : at_vec + n]
                at_vec += n
            files[path] = FileRecord(
                e["mtime_ns"],
                e["size"],
                [Passage(path, h, s, end, t) for h, s, end, t in e["passages"]],
                indptr.astype(np.int64),
                arrays["term_ids"][at_nnz : at_nnz + nnz],
                arrays["counts"][at_nnz : at_nnz + nnz],
                site=e.get("site"),
            )
            at_indptr += n + 1
            at_nnz += nnz
        if (at_indptr, at_nnz, at_vec) != (len(arrays["indptr"]), len(arrays["term_ids"]), len(arrays["vectors"])):
            raise ValueError("the index arrays don't match the file list")
        return files, legacy

    # ---- queries ----

    def _named_folders(self, skip) -> set[str]:
        return {name for name in skip if name and name in self.roots}

    def search(self, query: str, *, limit: int = 50, mode: str = "lexical", context_lines: int = 2, config=None,
               skip_folders=(), skip_files=(), model: str | None = None):
        from .search import search

        embedder = self.embedder_for(model)
        return search(
            self.view(embedder),
            query,
            limit=limit,
            mode=mode,
            embedder=embedder,
            context_lines=context_lines,
            read_lines=self.read_lines,
            config=config,
            skip_folders=self._named_folders(skip_folders),
            skip_files=skip_files,
        )

    def related(self, task_text: str, *, prefs=None, config=None, seed: int = 0, skip_folders=(), model: str | None = None):
        from .related import related

        embedder = self.embedder_for(model)
        return related(
            self.view(embedder),
            task_text,
            embedder=embedder,
            prefs=prefs,
            config=config,
            seed=seed,
            read_lines=self.read_lines,
            skip_folders=self._named_folders(skip_folders),
        )
