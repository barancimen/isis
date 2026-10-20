"""The corpus index: passages, BM25 and embeddings for every markdown file, kept up to date
incrementally and cached on disk."""

from __future__ import annotations

import logging
import os
import pickle
import re
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

INDEX_VERSION = 1
MARKDOWN = (".md", ".markdown")
EMBED_GROUP = 256  # passages per embedding call while indexing
PUBLISH_EVERY = 20.0  # seconds between snapshot updates while a large corpus is being embedded
SAVE_EVERY = 120.0  # seconds between cache writes during a long embedding run


@dataclass
class FileRecord:
    mtime_ns: int
    size: int
    passages: list[Passage]
    indptr: np.ndarray  # this file's rows of the BM25 matrix, as CSR pieces
    term_ids: np.ndarray
    counts: np.ndarray
    vectors: np.ndarray | None = None  # float16, one row per passage; None until embedded


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

    def __len__(self) -> int:
        return len(self.passages)

    def bm25_for(self, k1: float, b: float) -> BM25:
        """BM25 with other parameters, built on first use; the last few are kept."""
        key = (round(float(k1), 4), round(float(b), 4))
        if key == (self.bm25.k1, self.bm25.b):
            return self.bm25
        variant = self._bm25_variants.get(key)
        if variant is None:
            while len(self._bm25_variants) >= 4:
                self._bm25_variants.pop(next(iter(self._bm25_variants)))
            variant = self._bm25_variants[key] = self.bm25.with_params(*key)
        return variant

    def pattern_counts(self, pattern: re.Pattern) -> np.ndarray:
        """How often `pattern` matches in each passage. The last few patterns are kept, so a
        regex in a steer line is not rescanned on every refresh."""
        key = (pattern.pattern, pattern.flags)
        counts = self._pattern_counts.get(key)
        if counts is None:
            while len(self._pattern_counts) >= 32:
                self._pattern_counts.pop(next(iter(self._pattern_counts)))
            counts = np.fromiter(
                (sum(1 for m in pattern.finditer(p.embed_text) if m.end() > m.start()) for p in self.passages),
                np.float32, len(self.passages),
            )
            self._pattern_counts[key] = counts
        return counts

    def term_id(self, term: str) -> int | None:
        i = self.vocab.get(term)
        return i if i is not None and i < self.bm25.n_terms else None


@dataclass
class IndexStatus:
    state: str = "starting"  # "scanning", "embedding" or "ready"
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
        self._load()
        self._publish()

    @property
    def dim(self) -> int:
        return self.embedder.dim if self.embedder is not None else 0

    # ---- files ----

    def resolve(self, path: str) -> Path:
        """The file for a corpus-relative path. Refuses anything outside the corpus."""
        full = (self.root / path).resolve()
        if self.root not in full.parents or full.suffix.lower() not in MARKDOWN:
            raise ValueError(f"not a markdown file in the corpus: {path}")
        return full

    def read_lines(self, path: str) -> list[str]:
        return split_lines(self.resolve(path).read_text(encoding="utf-8", errors="replace"))

    def _scan(self) -> dict[str, os.stat_result]:
        found = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            here = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and (here / d).resolve() not in self.exclude
            ]
            for name in filenames:
                if name.lower().endswith(MARKDOWN) and not name.startswith("."):
                    try:
                        found[(here / name).relative_to(self.root).as_posix()] = (here / name).stat()
                    except OSError:
                        pass
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
            started = time.perf_counter()
            self._set_state("scanning", on_progress)
            found = self._scan()
            removed = [p for p in self.files if p not in found]
            changed = [
                p
                for p, st in found.items()
                if (r := self.files.get(p)) is None or (r.mtime_ns, r.size) != (st.st_mtime_ns, st.st_size)
            ]
            for path in removed:
                del self.files[path]
            for path in changed:
                try:
                    self.files[path] = self._read(path, found[path])
                except OSError as e:  # deleted or unreadable since the scan
                    log.warning("skipping %s: %s", path, e)
                    self.files.pop(path, None)
            if removed or changed:
                self._publish()

            embedded = self._embed_missing(on_progress)
            if removed or changed or embedded:
                self._publish()
                self._save()
            self._set_state("ready", on_progress)
            return {
                "files": len(self.files),
                "changed": len(changed),
                "removed": len(removed),
                "embedded": embedded,
                "seconds": round(time.perf_counter() - started, 2),
            }

    def _embed_missing(self, on_progress) -> int:
        if self.embedder is None:
            return 0
        pending = [r for r in self.files.values() if r.vectors is None or r.vectors.shape[1:] != (self.dim,)]
        self.status.embedded, self.status.to_embed = 0, sum(len(r.passages) for r in pending)
        if pending:
            self._set_state("embedding", on_progress)

        last_publish = last_save = time.monotonic()
        group: list[FileRecord] = []

        def flush():
            texts = [p.embed_text for r in group for p in r.passages]
            vectors = self.embedder.embed(texts) if texts else np.zeros((0, self.dim), np.float32)
            offset = 0
            for r in group:
                r.vectors = vectors[offset : offset + len(r.passages)].astype(np.float16)
                offset += len(r.passages)
            self.status.embedded += len(texts)
            group.clear()

        for record in pending:
            group.append(record)
            if sum(len(r.passages) for r in group) < EMBED_GROUP:
                continue
            flush()
            if on_progress:
                on_progress(self.status)
            now = time.monotonic()
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
    def _cache_path(self) -> Path:
        return self.index_dir / "index.pkl"

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    "version": INDEX_VERSION,
                    "chunking": self.chunking,
                    "embedder": getattr(self.embedder, "name", None),
                    "vocab": self.vocab.terms,
                    "files": self.files,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, self._cache_path)

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            log.warning("ignoring unreadable index cache: %s", e)
            return
        if data.get("version") != INDEX_VERSION or tuple(data.get("chunking", ())) != self.chunking:
            return
        self.vocab = Vocabulary(data["vocab"])
        self.files = data["files"]
        if data.get("embedder") != getattr(self.embedder, "name", None):
            for record in self.files.values():  # a different model: keep the text, re-embed
                record.vectors = None

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
