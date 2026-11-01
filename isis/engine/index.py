"""The corpus index: passages, BM25 and embeddings for every markdown file, kept up to date
incrementally and cached on disk.

The cache is plain data, JSON for the text and an .npz of arrays read with allow_pickle=False, so
a cache that came with someone else's corpus can't run code when it is loaded."""

from __future__ import annotations

import json
import logging
import os
import re
import stat as statmode
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .chunker import MAX_WORDS, TARGET_WORDS, Passage, chunk_markdown, split_lines
from .lexical import BM25, Vocabulary, term_counts
from .text import tokenize

log = logging.getLogger(__name__)


def configure_logging() -> None:
    """What `isis` prints: its own progress messages, and only warnings from other libraries."""
    logging.basicConfig(level=logging.WARNING, format="isis: %(message)s")
    logging.getLogger("isis").setLevel(logging.INFO)


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s" if seconds < 10 else f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _changes(new: list[str], modified: list[str], removed: list[str]) -> str:
    """'100 new files, 2 changed, 1 removed', naming the files when there are only a few."""
    parts = []
    for paths, label in ((new, _count(len(new), "new file")), (modified, f"{len(modified):,} changed"),
                         (removed, f"{len(removed):,} removed")):
        if paths:
            names = f" ({', '.join(sorted(paths))})" if len(new) + len(modified) + len(removed) <= 5 else ""
            parts.append(label + names)
    return ", ".join(parts)

INDEX_VERSION = 2
MARKDOWN = (".md", ".markdown")
EMBED_GROUP = 256  # passages per embedding call while indexing
PUBLISH_EVERY = 20.0  # seconds between snapshot updates while a large corpus is being embedded
SAVE_EVERY = 120.0  # seconds between cache writes during a long embedding run
REPORT_EVERY = 10.0  # seconds between progress messages during a long embedding run


@dataclass
class FileRecord:
    mtime_ns: int
    size: int
    passages: list[Passage]
    indptr: np.ndarray  # this file's rows of the BM25 matrix, as CSR pieces
    term_ids: np.ndarray
    counts: np.ndarray
    vectors: np.ndarray | None = None  # float16, one row per passage; None until embedded
    reuse: dict | None = None  # embed_text -> vector kept from before an edit; not saved


@dataclass
class Snapshot:
    """An immutable view of the index. A search uses one snapshot throughout, so a refresh
    running in another thread never changes data under it."""

    passages: list[Passage]
    bm25: BM25
    vocab: Vocabulary  # shared with the engine and only ever grows: look terms up with term_id()
    sorted_terms: list[str]  # terms that occur in the corpus, for prefix expansion
    vectors: np.ndarray  # float32 [n, dim]; zero rows where has_vector is False
    has_vector: np.ndarray  # bool [n]
    file_rows: dict[str, tuple[int, int]]  # path -> [start, end) rows
    _bm25_variants: dict = field(default_factory=dict, repr=False)
    _pattern_counts: dict = field(default_factory=dict, repr=False)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # requests share snapshots

    def __len__(self) -> int:
        return len(self.passages)

    def bm25_for(self, k1: float, b: float) -> BM25:
        """BM25 with other parameters, built on first use; the last few are kept."""
        key = (round(float(k1), 4), round(float(b), 4))
        if key == (self.bm25.k1, self.bm25.b):
            return self.bm25
        return self._cached(self._bm25_variants, key, 4, lambda: self.bm25.with_params(*key))

    def pattern_counts(self, pattern: re.Pattern) -> np.ndarray:
        """How often `pattern` matches in each passage. The last few patterns are kept, so a
        regex in a steer line is not rescanned on every refresh."""
        return self._cached(self._pattern_counts, (pattern.pattern, pattern.flags), 32, lambda: np.fromiter(
            (sum(1 for m in pattern.finditer(p.embed_text) if m.end() > m.start()) for p in self.passages),
            np.float32, len(self.passages),
        ))

    def _cached(self, cache: dict, key, size: int, make: Callable):
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
    updated: float = 0.0  # time.time() when the snapshot last changed


class Engine:
    """Owns the index of one corpus folder. `refresh()` brings it up to date with the files on
    disk; `snapshot` is what searches read."""

    def __init__(
        self,
        corpus_dir: str | Path,
        *,
        embedder=None,
        index_dir: str | Path | None = None,
        exclude: tuple[str | Path, ...] = (),
        target_words: int = TARGET_WORDS,
        max_words: int = MAX_WORDS,
    ):
        self.root = Path(corpus_dir).resolve()
        self.index_dir = Path(index_dir).resolve() if index_dir else self.root / ".isis"
        self.embedder = embedder
        self.exclude = {Path(p).resolve() for p in exclude}
        self.chunking = (target_words, max_words)
        self.vocab = Vocabulary()
        self.files: dict[str, FileRecord] = {}
        self.status = IndexStatus()
        self._lock = threading.Lock()
        # The model the cached vectors came from. Without a model they are kept as they are, so a
        # keyword-only run never throws away embeddings that took hours to compute.
        self._vectors_from = getattr(embedder, "name", None)
        self.links: set[str] = set()  # corpus-relative paths of the folder links the last scan met
        self._said: set[str] = set()  # one-time messages about links, so rescans stay quiet
        self._refreshed = False
        self._load()
        self._publish()

    @property
    def dim(self) -> int:
        return self.embedder.dim if self.embedder is not None else 0

    # ---- files ----

    def resolve(self, path: str) -> Path:
        """The file for a corpus-relative path. Refuses a path that climbs out of the corpus.
        Links inside the corpus are followed wherever they point (that is how a folder from
        elsewhere joins it), as long as what they reach is markdown too."""
        rel = Path(path)
        full = self.root / rel
        if (rel.is_absolute() or ".." in rel.parts or full.suffix.lower() not in MARKDOWN
                or full.resolve().suffix.lower() not in MARKDOWN):
            raise ValueError(f"not a markdown file in the corpus: {path}")
        return full

    def read_lines(self, path: str) -> list[str]:
        return split_lines(self.resolve(path).read_text(encoding="utf-8", errors="replace"))

    def _say_once(self, level: int, message: str) -> None:
        if message not in self._said:
            self._said.add(message)
            log.log(level, message)

    def _scan(self) -> dict[str, os.stat_result]:
        """Every markdown file under the corpus, following symlinks. The real folders come first and
        links after them, and each folder or file is taken once (by device and inode), so a link
        loop ends and something reachable both ways keeps its real path."""
        found: dict[str, os.stat_result] = {}
        folders, files = set(), set()
        folder_links: list[Path] = []
        file_links: list[Path] = []

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

        def walk(top: Path) -> None:
            for dirpath, dirnames, filenames in os.walk(top):
                here = Path(dirpath)
                keep = []
                for d in sorted(dirnames):
                    path = here / d
                    if d.startswith(".") or (path.resolve() in self.exclude):
                        continue
                    try:
                        if path.is_symlink():
                            folder_links.append(path)
                        elif first_visit(path):
                            keep.append(d)
                    except OSError:  # deleted since os.walk listed it
                        pass
                dirnames[:] = keep
                for name in filenames:
                    path = here / name
                    if name.startswith("."):
                        continue
                    if not name.lower().endswith(MARKDOWN):
                        if path.is_symlink() and not path.exists():  # a linked folder that has gone away
                            rel = path.relative_to(self.root).as_posix()
                            self._say_once(logging.WARNING, f"link {rel} skipped: its target doesn't exist")
                        continue
                    try:
                        st = path.lstat()
                    except OSError:
                        continue
                    if statmode.S_ISLNK(st.st_mode):
                        file_links.append(path)
                    else:
                        found[path.relative_to(self.root).as_posix()] = st
                        files.add((st.st_dev, st.st_ino))

        first_visit(self.root)
        walk(self.root)
        links = set()
        while folder_links:  # walking a linked folder can find more links
            link = folder_links.pop(0)
            rel = link.relative_to(self.root).as_posix()
            links.add(rel)
            try:
                if excluded(link):
                    continue
                if not first_visit(link):
                    self._say_once(logging.INFO, f"link {rel} skipped: its folder is already indexed")
                    continue
            except OSError as e:
                self._say_once(logging.WARNING, f"link {rel} skipped: {e.strerror or e}")
                continue
            self._say_once(logging.INFO, f"following link {rel} -> {link.resolve()}")
            walk(link)
        self.links = links
        for link in file_links:
            rel = link.relative_to(self.root).as_posix()
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
        return found

    def _read(self, path: str, stat: os.stat_result) -> FileRecord:
        text = (self.root / path).read_text(encoding="utf-8", errors="replace")
        passages = chunk_markdown(path, text, *self.chunking)
        tokens = [tokenize(p.heading) + tokenize(p.text) for p in passages]
        return FileRecord(stat.st_mtime_ns, stat.st_size, passages, *term_counts(tokens, self.vocab))

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
            log.info("scanning %s", self.root)
        found = self._scan()
        removed = [p for p in self.files if p not in found]
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
                     _count(len(found), "markdown file") if found else self.root)
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
            if old is not None and old.vectors is not None and old.vectors.shape[1:] == (self.dim,) and self.dim:
                # An edit changes a few passages: the others keep their vectors.
                record.reuse = {p.embed_text: v for p, v in zip(old.passages, old.vectors)}
            self.files[path] = record
        if removed or changed:
            self._publish()

        embedded = self._embed_missing(on_progress)
        if removed or changed or embedded:
            self._publish()
            self._save()
        self._set_state("ready", on_progress)
        if first or removed or changed or embedded:
            meaning = "" if self.embedder is not None else ", keywords only"
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

    def _embed_missing(self, on_progress) -> int:
        if self.embedder is None:
            return 0
        pending = [r for r in self.files.values() if r.vectors is None or r.vectors.shape[1:] != (self.dim,)]
        todo = {id(r): [p.embed_text for p in r.passages if p.embed_text not in (r.reuse or {})] for r in pending}
        self.status.embedded, self.status.to_embed = 0, sum(map(len, todo.values()))
        if self.status.to_embed:
            self._set_state("embedding", on_progress)
            kept = sum(len(r.passages) for r in pending) - self.status.to_embed
            log.info("embedding %s%s", _count(self.status.to_embed, "passage"),
                     f" ({kept:,} unchanged kept their embeddings)" if kept else "")

        begun = last_report = last_publish = last_save = time.monotonic()
        group: list[FileRecord] = []

        def flush():
            texts = [t for r in group for t in todo[id(r)]]
            fresh = iter(self.embedder.embed(texts) if texts else np.zeros((0, self.dim), np.float32))
            for r in group:
                rows = [r.reuse[p.embed_text] if p.embed_text in (r.reuse or {}) else next(fresh) for p in r.passages]
                r.vectors = np.array(rows, np.float16).reshape(len(r.passages), self.dim)
                r.reuse = None
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
        return self.status.to_embed

    def _set_state(self, state: str, on_progress) -> None:
        self.status.state = state
        if on_progress:
            on_progress(self.status)

    # ---- snapshot ----

    def _publish(self) -> None:
        self.snapshot = self._compile()
        self.status.files = len(self.files)
        self.status.passages = len(self.snapshot)
        self.status.updated = time.time()

    def _compile(self) -> Snapshot:
        passages: list[Passage] = []
        file_rows = {}
        indptrs, term_ids, counts, vectors, has = [np.zeros(1, np.int64)], [], [], [], []
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
            ok = r.vectors is not None and r.vectors.shape == (n, self.dim)
            vectors.append(r.vectors if ok else np.zeros((n, self.dim), np.float16))
            has.append(np.full(n, ok))

        bm25 = BM25(
            np.concatenate(indptrs),
            np.concatenate(term_ids) if term_ids else np.zeros(0, np.int32),
            np.concatenate(counts) if counts else np.zeros(0, np.float32),
            n_terms=len(self.vocab),
        )
        return Snapshot(
            passages=passages,
            bm25=bm25,
            vocab=self.vocab,
            sorted_terms=sorted(self.vocab.terms[i] for i in np.flatnonzero(bm25.doc_freq)),
            vectors=np.concatenate(vectors).astype(np.float32) if vectors else np.zeros((0, self.dim), np.float32),
            has_vector=np.concatenate(has) if has else np.zeros(0, bool),
            file_rows=file_rows,
        )

    # ---- cache ----

    @property
    def _cache_paths(self) -> tuple[Path, Path]:
        return self.index_dir / "index.json", self.index_dir / "index.npz"

    def index_bytes(self) -> int:
        """The size of the index on disk."""
        total = 0
        for path in self._cache_paths:
            try:
                total += path.stat().st_size
            except OSError:
                pass  # not saved yet, or being replaced
        return total

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(self.files)
        records = [self.files[p] for p in paths]
        embedded = [r for r in records if r.vectors is not None]
        meta = {
            "version": INDEX_VERSION,
            "chunking": list(self.chunking),
            "embedder": self._vectors_from,
            "vocab": self.vocab.terms,
            "files": [
                {
                    "path": path,
                    "mtime_ns": r.mtime_ns,
                    "size": r.size,
                    "embedded": r.vectors is not None,
                    "passages": [[p.heading, p.start_line, p.end_line, p.text] for p in r.passages],
                }
                for path, r in zip(paths, records)
            ],
        }
        dim = embedded[0].vectors.shape[1] if embedded else 0
        arrays = {
            "indptr": np.concatenate([r.indptr for r in records]) if records else np.zeros(0, np.int64),
            "term_ids": np.concatenate([r.term_ids for r in records]) if records else np.zeros(0, np.int32),
            "counts": np.concatenate([r.counts for r in records]) if records else np.zeros(0, np.float32),
            "vectors": np.concatenate([r.vectors for r in embedded]) if embedded else np.zeros((0, dim), np.float16),
        }
        json_path, npz_path = self._cache_paths
        # The arrays go first: a cache is only used when its JSON matches them (see _load).
        unique = f"{os.getpid()}-{threading.get_ident()}"  # another process may share the folder
        tmp = npz_path.with_name(f"index.{unique}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, npz_path)
        tmp = json_path.with_name(f"index.{unique}.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, json_path)
        old = self.index_dir / "index.pkl"
        if old.exists():  # the version 1 cache, which this one replaces
            old.unlink()

    def _load(self) -> None:
        json_path, npz_path = self._cache_paths
        if (self.index_dir / "index.pkl").exists() and not json_path.exists():
            log.info("ignoring the old index cache (index.pkl); the index will be rebuilt")
        if not json_path.exists() or not npz_path.exists():
            return
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if meta.get("version") != INDEX_VERSION or tuple(meta.get("chunking", ())) != self.chunking:
                return
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in ("indptr", "term_ids", "counts", "vectors")}
            files = self._records(meta["files"], arrays)
            vocab = Vocabulary(meta["vocab"])
            saved = meta.get("embedder")
        except Exception as e:
            log.warning("ignoring unreadable index cache: %s", e)
            return
        self.vocab = vocab
        self.files = files
        log.info("loaded the index cache: %s, %s", _count(len(files), "file"),
                 _count(sum(len(r.passages) for r in files.values()), "passage"))
        if self.embedder is None:
            self._vectors_from = saved
        elif saved != self.embedder.name:  # a different model: keep the text, re-embed
            log.info("the embedding model changed: every passage will be embedded again")
            for record in self.files.values():
                record.vectors = None

    @staticmethod
    def _records(entries: list[dict], arrays: dict) -> dict[str, FileRecord]:
        """FileRecords from the cache's JSON entries and the arrays they index into, in order."""
        files = {}
        at_indptr = at_nnz = at_vec = 0
        for e in entries:
            n = len(e["passages"])
            indptr = arrays["indptr"][at_indptr : at_indptr + n + 1]
            nnz = int(indptr[-1])
            vectors = None
            if e["embedded"]:
                vectors = arrays["vectors"][at_vec : at_vec + n]
                at_vec += n
            files[e["path"]] = FileRecord(
                e["mtime_ns"],
                e["size"],
                [Passage(e["path"], h, s, end, t) for h, s, end, t in e["passages"]],
                indptr.astype(np.int64),
                arrays["term_ids"][at_nnz : at_nnz + nnz],
                arrays["counts"][at_nnz : at_nnz + nnz],
                vectors,
            )
            at_indptr += n + 1
            at_nnz += nnz
        if (at_indptr, at_nnz, at_vec) != (len(arrays["indptr"]), len(arrays["term_ids"]), len(arrays["vectors"])):
            raise ValueError("the index arrays don't match the file list")
        return files

    # ---- queries ----

    def search(self, query: str, *, limit: int = 50, mode: str = "lexical", context_lines: int = 2, config=None):
        from .search import search

        return search(
            self.snapshot,
            query,
            limit=limit,
            mode=mode,
            embedder=self.embedder,
            context_lines=context_lines,
            read_lines=self.read_lines,
            config=config,
        )

    def related(self, task_text: str, *, cursor_line: int | None = None, prefs=None, config=None, seed: int = 0):
        from .related import related

        return related(
            self.snapshot,
            task_text,
            embedder=self.embedder,
            cursor_line=cursor_line,
            prefs=prefs,
            config=config,
            seed=seed,
            read_lines=self.read_lines,
        )
