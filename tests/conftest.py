import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from isis.engine.index import Engine
from isis.engine.semantic import HashEmbedder
from isis.server.app import create_app
from isis.server.tasks import TaskStore

FILES = {
    "whales.md": "# Whales\n\nThe sperm whale dives deep for squid.\n\nAmbergris is found in whales.\n",
    "ships/pequod.md": "# The Pequod\n\nThe Pequod was a whaling ship.\n\nAhab was her captain, and he hunted the white whale.\n",
    "finches.md": "# Finches\n\nThe finches of the Galapagos differ in the shape of their beaks.\n",
    ".hidden/secret.md": "# Hidden\n\nAmbergris here should never be indexed.\n",
    "notes.txt": "Ambergris in a text file is not indexed either.\n",
}


def write_corpus(root: Path, files: dict[str, str] = FILES) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def legacy_cache(root: Path) -> list[Path]:
    """Rewrites every index cache under `root` the way caches were saved before the passages' text
    had a file of its own (version 2: all of it in index.json), and returns their index.json files."""
    from isis.engine.index import UNKNOWN, _split

    written = []
    for json_path in sorted(root.rglob("index.json")):
        folder, meta = json_path.parent, json.loads(json_path.read_text())
        cache, counts = folder / f"cache.{meta['token']}", meta["counts"]
        a = {path.stem: np.load(path) for path in cache.glob("*.npy")}
        texts = (cache / "texts.bin").read_bytes()
        headings, starts = _split(a["headings"], counts["headings"]), a["text_starts"]
        files, at = [], 0
        for path, mtime, size, n, site, source in zip(
            _split(a["file_paths"], counts["files"]), a["file_mtime"].tolist(), a["file_size"].tolist(),
            a["file_passages"].tolist(), _split(a["file_sites"], counts["files"]), _split(a["file_sources"], counts["files"]),
        ):
            passages = [[headings[a["heading_ids"][i]], int(a["start_lines"][i]), int(a["end_lines"][i]),
                         texts[starts[i]:starts[i + 1]].decode()] for i in range(at, at + n)]
            files.append({"path": path, "mtime_ns": mtime, "size": size, "site": None if site == UNKNOWN else site,
                          "source": None if source == UNKNOWN else source, "passages": passages})
            at += n
        old = {"version": 2, "chunking": meta["chunking"], "chunker": meta["chunker"], "terms": meta["terms"],
               "vocab": _split(a["vocab"], counts["vocab"]), "words": _split(a["words"], counts["words"]), "files": files}
        np.savez(folder / "index.npz", indptr=a["indptr"], term_ids=a["term_ids"], counts=a["counts"],
                 vectors=np.zeros((0, 0), np.float16), word_terms=a["word_terms"], word_masks=a["word_masks"])
        json_path.write_text(json.dumps(old))
        shutil.rmtree(cache)
        written.append(json_path)
    return written


def paths(hits) -> list[str]:
    return [h.path for h in hits]


@pytest.fixture
def corpus(tmp_path) -> Path:
    return write_corpus(tmp_path / "corpus")


@pytest.fixture
def engine(corpus, tmp_path) -> Engine:
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=tmp_path / "index", target_words=0)
    e.refresh()
    return e


@pytest.fixture
def client(engine, corpus):
    with TestClient(create_app(engine, TaskStore(corpus / "_tasks"), background=False, watch=False)) as c:
        yield c
